#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>

using namespace pxr;

// Comparison operators for GfVec2f
inline bool operator<(const GfVec2f& lhs, const GfVec2f& rhs)
{
    if (lhs[0] != rhs[0])
        return lhs[0] < rhs[0];
    return lhs[1] < rhs[1];
}

inline bool operator>(const GfVec2f& lhs, const GfVec2f& rhs)
{
    return rhs < lhs;
}

// Comparison operators for GfVec3f
inline bool operator<(const GfVec3f& lhs, const GfVec3f& rhs)
{
    if (lhs[0] != rhs[0])
        return lhs[0] < rhs[0];
    if (lhs[1] != rhs[1])
        return lhs[1] < rhs[1];
    return lhs[2] < rhs[2];
}

inline bool operator>(const GfVec3f& lhs, const GfVec3f& rhs)
{
    return rhs < lhs;
}

#include <algorithm>

#include "nodes/core/socket_trait.inl"
template<>
struct ValueTrait<pxr::GfVec3f> {
    static constexpr bool has_min = true;
    static constexpr bool has_max = true;
    static constexpr bool has_default = true;
};
template<>
struct ValueTrait<pxr::GfVec2f> {
    static constexpr bool has_min = true;
    static constexpr bool has_max = true;
    static constexpr bool has_default = true;
};

#include <spdlog/spdlog.h>

#include "../source/renderTLAS.h"
#include "GPUContext/raytracing_context.hpp"
#include "nodes/core/def/node_def.hpp"
#include "nvrhi/nvrhi.h"
#include "nvrhi/utils.h"
#include "render_node_base.h"
#include "utils/math.h"

// Material BRDF Analyzer Node
// Visualizes BRDF eval, pdf, and sample distribution for a given material

NODE_DEF_OPEN_SCOPE

// Environment prefilter data structure (from path_tracing)
struct EnvironmentPrefilterData {
    float4x4 u_envMatrix;
    float3 u_envLightIntensity;
    int u_envRadianceMips;
    int u_envIrradianceMips;
};

// Parameters for BRDF analysis
struct BRDFAnalysisParams {
    pxr::GfVec3f incident_direction;  // Wi in world space
    pxr::GfVec2f uv_coord;            // UV coordinate
    unsigned material_id;             // Material to analyze
    unsigned resolution;              // Output texture resolution
    unsigned num_samples;             // Number of samples for Monte Carlo
};

NODE_DECLARATION_FUNCTION(material_brdf_analyzer)
{
    b.add_input<float>("Incident Direction X").default_val(0.0f);
    b.add_input<float>("Incident Direction Y").default_val(0.0f);
    b.add_input<float>("Incident Direction Z").default_val(1.0f);
    b.add_input<float>("UV X").default_val(0.5f);
    b.add_input<float>("UV Y").default_val(0.5f);
    b.add_input<int>("Material ID").default_val(0).min(0).max(100);
    b.add_input<int>("Resolution").min(32).max(2048).default_val(512);
    b.add_input<int>("Num Samples").min(100).max(1000000).default_val(10000);

    b.add_output<nvrhi::TextureHandle>("BRDF Eval");
    b.add_output<nvrhi::TextureHandle>("PDF");
    b.add_output<nvrhi::TextureHandle>("Sample Distribution");
}

NODE_EXECUTION_FUNCTION(material_brdf_analyzer)
{
    using namespace nvrhi;

    // Get input parameters
    float incident_x = params.get_input<float>("Incident Direction X");
    float incident_y = params.get_input<float>("Incident Direction Y");
    float incident_z = params.get_input<float>("Incident Direction Z");
    float uv_x = params.get_input<float>("UV X");
    float uv_y = params.get_input<float>("UV Y");
    int material_id = params.get_input<int>("Material ID");
    int resolution = params.get_input<int>("Resolution");
    int num_samples = params.get_input<int>("Num Samples");

    // Normalize incident direction
    float3 incident_dir = float3(incident_x, incident_y, incident_z);
    float length = std::sqrt(incident_dir.x * incident_dir.x + 
                            incident_dir.y * incident_dir.y + 
                            incident_dir.z * incident_dir.z);
    if (length > 0.0f) {
        incident_dir.x /= length;
        incident_dir.y /= length;
        incident_dir.z /= length;
    }

    // Create output textures
    TextureDesc tex_desc;
    tex_desc.width = resolution;
    tex_desc.height = resolution;
    tex_desc.format = Format::RGBA32_FLOAT;
    tex_desc.isUAV = true;
    tex_desc.debugName = "BRDF_Eval_Output";
    tex_desc.initialState = ResourceStates::UnorderedAccess;
    tex_desc.keepInitialState = true;

    auto eval_output = resource_allocator.create(tex_desc);
    MARK_DESTROY_NVRHI_RESOURCE(eval_output);

    tex_desc.debugName = "PDF_Output";
    auto pdf_output = resource_allocator.create(tex_desc);
    MARK_DESTROY_NVRHI_RESOURCE(pdf_output);

    tex_desc.debugName = "Sample_Distribution";
    auto sample_output = resource_allocator.create(tex_desc);
    MARK_DESTROY_NVRHI_RESOURCE(sample_output);

    // Get materials and setup callable shaders
    auto& materials = global_payload.get_materials();
    std::unordered_map<unsigned, std::string> callable_shaders;

    for (auto material : materials) {
        auto location = material.second->GetMaterialLocation();
        if (location == -1) {
            continue;
        }
        callable_shaders[location] = material.second->GetMaterialName();
    }

    // ===== BRDF Eval Shader =====
    ProgramDesc eval_program_desc;
    eval_program_desc.set_path("shaders/material_brdf_eval.slang");
    eval_program_desc.shaderType = ShaderType::AllRayTracing;
    eval_program_desc.nvapi_support = true;
    eval_program_desc.define(
        "FALCOR_MATERIAL_INSTANCE_SIZE",
        std::to_string(c_FalcorMaterialInstanceSize));

    for (auto material : materials) {
        auto location = material.second->GetMaterialLocation();
        if (location == -1) {
            continue;
        }
        eval_program_desc.add_source_code(
            material.second->GetShader(shader_factory));
    }

    auto eval_compiled = resource_allocator.create(eval_program_desc);
    MARK_DESTROY_NVRHI_RESOURCE(eval_compiled);
    CHECK_PROGRAM_ERROR(eval_compiled);

    // Setup eval shader parameters
    struct AnalysisConstants {
        float3 incident_direction;
        float _pad0;
        float2 uv_coord;
        uint32_t material_id;
        uint32_t resolution;
    };

    AnalysisConstants analysis_constants;
    analysis_constants.incident_direction = incident_dir;
    analysis_constants.uv_coord = float2(uv_x, uv_y);
    analysis_constants.material_id = static_cast<uint32_t>(material_id);
    analysis_constants.resolution = static_cast<uint32_t>(resolution);

    auto analysis_cb = create_constant_buffer(params, analysis_constants);
    MARK_DESTROY_NVRHI_RESOURCE(analysis_cb);

    // Create sampler
    SamplerDesc sampler_desc;
    sampler_desc.addressU = SamplerAddressMode::Wrap;
    sampler_desc.addressV = SamplerAddressMode::Wrap;
    sampler_desc.addressW = SamplerAddressMode::Wrap;
    auto sampler = resource_allocator.create(sampler_desc);
    MARK_DESTROY_NVRHI_RESOURCE(sampler);

    // Setup program vars for eval shader
    ProgramVars eval_vars(resource_allocator, eval_compiled);
    
    // Debug: Check reflection info
    spdlog::info("=== Eval shader reflection info ===");
    auto& reflection = eval_compiled->get_reflection_info();
    auto& layouts = reflection.get_binding_layout_descs();
    for (size_t space_idx = 0; space_idx < layouts.size(); ++space_idx) {
        spdlog::info("Binding space {}: {} bindings", space_idx, layouts[space_idx].bindings.size());
        for (size_t bind_idx = 0; bind_idx < layouts[space_idx].bindings.size(); ++bind_idx) {
            auto& bind = layouts[space_idx].bindings[bind_idx];
            spdlog::info("  Binding {}: slot={}, type={}, size={}", 
                bind_idx, bind.slot, (int)bind.type, bind.size);
        }
    }
    
    eval_vars["SceneBVH"] = instance_collection->get_tlas();
    eval_vars["eval_output"] = eval_output;
    eval_vars["pdf_output"] = pdf_output;
    eval_vars["analysis_params"] = analysis_cb;

    for (int i = 0; i < 9; ++i) {
        eval_vars["samplers"][i] = sampler;
    }
    
    // Bind MaterialX environment samplers (from mx_environment_prefilter.slang)
    eval_vars["u_envRadiance_sampler"] = sampler;
    eval_vars["u_envIrradiance_sampler"] = sampler;

    eval_vars["instanceDescBuffer"] = instance_collection->instance_pool.get_device_buffer();
    eval_vars["meshDescBuffer"] = instance_collection->mesh_pool.get_device_buffer();
    eval_vars["materialBlobBuffer"] = instance_collection->material_pool.get_device_buffer();
    eval_vars["materialHeaderBuffer"] = instance_collection->material_header_pool.get_device_buffer();

    eval_vars.set_descriptor_table(
        "t_BindlessBuffers",
        instance_collection->bindlessData.bufferDescriptorTableManager->GetDescriptorTable(),
        instance_collection->bindlessData.bufferBindlessLayout);

    eval_vars.set_descriptor_table(
        "t_BindlessTextures",
        instance_collection->bindlessData.textureDescriptorTableManager->GetDescriptorTable(),
        instance_collection->bindlessData.textureBindlessLayout);

    eval_vars.finish_setting_vars();

    // Create raytracing context for eval shader
    RaytracingContext eval_context(resource_allocator, eval_vars);
    eval_context.announce_raygeneration("EvalPdfRayGen");
    eval_context.announce_hitgroup("EvalHit", "", "", 0);
    eval_context.announce_miss("EvalMiss", 0);

    for (auto& callable : callable_shaders) {
        eval_context.announce_callable(callable.second, callable.first);
    }

    eval_context.finish_announcing_shader_names();

    // Execute eval shader
    eval_context.begin();
    eval_context.trace_rays({}, eval_vars, resolution, resolution, 1);
    eval_context.finish();

    // ===== BRDF Sample Shader =====
    ProgramDesc sample_program_desc;
    sample_program_desc.set_path("shaders/material_brdf_sample.slang");
    sample_program_desc.shaderType = ShaderType::AllRayTracing;
    sample_program_desc.nvapi_support = true;
    sample_program_desc.define(
        "FALCOR_MATERIAL_INSTANCE_SIZE",
        std::to_string(c_FalcorMaterialInstanceSize));

    for (auto material : materials) {
        auto location = material.second->GetMaterialLocation();
        if (location == -1) {
            continue;
        }
        sample_program_desc.add_source_code(
            material.second->GetShader(shader_factory));
    }

    auto sample_compiled = resource_allocator.create(sample_program_desc);
    MARK_DESTROY_NVRHI_RESOURCE(sample_compiled);
    CHECK_PROGRAM_ERROR(sample_compiled);

    // Setup sample shader parameters
    struct SampleConstants {
        float3 incident_direction;
        float _pad0;
        float2 uv_coord;
        uint32_t material_id;
        uint32_t resolution;
        uint32_t num_samples;
    };

    SampleConstants sample_constants;
    sample_constants.incident_direction = incident_dir;
    sample_constants.uv_coord = float2(uv_x, uv_y);
    sample_constants.material_id = static_cast<uint32_t>(material_id);
    sample_constants.resolution = static_cast<uint32_t>(resolution);
    sample_constants.num_samples = static_cast<uint32_t>(num_samples);

    auto sample_cb = create_constant_buffer(params, sample_constants);
    MARK_DESTROY_NVRHI_RESOURCE(sample_cb);

    // Create RNG buffer
    BufferDesc rng_desc;
    rng_desc.byteSize = sizeof(uint32_t) * resolution * resolution;
    rng_desc.structStride = sizeof(uint32_t);  // RWStructuredBuffer<uint> in shader
    rng_desc.canHaveUAVs = true;
    rng_desc.debugName = "RNG_Seeds";
    rng_desc.initialState = ResourceStates::UnorderedAccess;
    rng_desc.keepInitialState = true;

    auto rng_buffer = resource_allocator.create(rng_desc);
    MARK_DESTROY_NVRHI_RESOURCE(rng_buffer);

    // Setup program vars for sample shader
    ProgramVars sample_vars(resource_allocator, sample_compiled);
    sample_vars["SceneBVH"] = instance_collection->get_tlas();
    sample_vars["sample_output"] = sample_output;
    sample_vars["sample_params"] = sample_cb;
    sample_vars["rng_seeds"] = rng_buffer;

    for (int i = 0; i < 9; ++i) {
        sample_vars["samplers"][i] = sampler;
    }
    
    // Bind MaterialX environment samplers
    sample_vars["u_envRadiance_sampler"] = sampler;
    sample_vars["u_envIrradiance_sampler"] = sampler;

    sample_vars["instanceDescBuffer"] = instance_collection->instance_pool.get_device_buffer();
    sample_vars["meshDescBuffer"] = instance_collection->mesh_pool.get_device_buffer();
    sample_vars["materialBlobBuffer"] = instance_collection->material_pool.get_device_buffer();
    sample_vars["materialHeaderBuffer"] = instance_collection->material_header_pool.get_device_buffer();

    sample_vars.set_descriptor_table(
        "t_BindlessBuffers",
        instance_collection->bindlessData.bufferDescriptorTableManager->GetDescriptorTable(),
        instance_collection->bindlessData.bufferBindlessLayout);

    sample_vars.set_descriptor_table(
        "t_BindlessTextures",
        instance_collection->bindlessData.textureDescriptorTableManager->GetDescriptorTable(),
        instance_collection->bindlessData.textureBindlessLayout);

    sample_vars.finish_setting_vars();

    // Create raytracing context for sample shader
    RaytracingContext sample_context(resource_allocator, sample_vars);
    sample_context.announce_raygeneration("SampleRayGen");
    sample_context.announce_hitgroup("SampleHit", "", "", 0);
    sample_context.announce_miss("SampleMiss", 0);

    for (auto& callable : callable_shaders) {
        sample_context.announce_callable(callable.second, callable.first);
    }

    sample_context.finish_announcing_shader_names();

    // Execute sample shader
    sample_context.begin();
    sample_context.trace_rays({}, sample_vars, resolution * resolution, 1, 1);
    sample_context.finish();

    // Set outputs
    params.set_output("BRDF Eval", eval_output);
    params.set_output("PDF", pdf_output);
    params.set_output("Sample Distribution", sample_output);

    return true;
}

NODE_DECLARATION_REQUIRED(material_brdf_analyzer)
NODE_DECLARATION_UI(material_brdf_analyzer);
NODE_DEF_CLOSE_SCOPE


#include <spdlog/spdlog.h>

#include <memory>

#include "../source/renderTLAS.h"
#include "GPUContext/raytracing_context.hpp"
#include "nodes/core/def/node_def.hpp"
#include "nvrhi/nvrhi.h"
#include "nvrhi/utils.h"
#include "render_node_base.h"
#include "shaders/shaders/utils/HitObject.h"
#include "utils/math.h"

#include "Scene/MaterialParamsBuffer.slang"

// A traditional path tracing node

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(path_tracing)
{
    b.add_input<nvrhi::BufferHandle>("Pixel Target");
    b.add_input<nvrhi::BufferHandle>("Rays");
    b.add_input<nvrhi::BufferHandle>("Random Seeds");

    b.add_output<nvrhi::TextureHandle>("Output");

    // Function content omitted
}

struct EnvironmentPrefilterData {
    float4x4 u_envMatrix;
    float3 u_envLightIntensity;
    int u_envRadianceMips;
    int u_envIrradianceMips;
};

NODE_EXECUTION_FUNCTION(path_tracing)
{
    using namespace nvrhi;

    ProgramDesc program_desc;
    program_desc.set_path("shaders/path_tracing.slang");
    program_desc.shaderType = nvrhi::ShaderType::AllRayTracing;
    program_desc.nvapi_support = true;

    auto& materials = global_payload.get_materials();

    std::unordered_map<unsigned, std::string> callable_shaders;

    for (auto material : materials) {
        if (material.second == nullptr) {
            spdlog::warn(
                "Null material found in path tracing node, {}",
                material.first.GetText());
            continue;
        }
        auto location = material.second->GetMaterialLocation();
        if (location == -1) {
            continue;
        }

        program_desc.add_source_code(
            material.second->GetShader(shader_factory));

        auto callable = material.second->GetShader(shader_factory);
        callable_shaders[location] = material.second->GetMaterialName();

        // combined_desc.add_component(callable->get_linked_program());
    }

    auto raytrace_compiled = resource_allocator.create(program_desc);
    MARK_DESTROY_NVRHI_RESOURCE(raytrace_compiled);
    CHECK_PROGRAM_ERROR(raytrace_compiled);

    auto m_CommandList = resource_allocator.create(CommandListDesc{});
    MARK_DESTROY_NVRHI_RESOURCE(m_CommandList);

    auto output =
        create_default_render_target(params, nvrhi::Format::RGBA32_FLOAT);

    ProgramVars program_vars(resource_allocator, raytrace_compiled);

    SamplerDesc sampler_desc;
    sampler_desc.addressU = nvrhi::SamplerAddressMode::Wrap;
    sampler_desc.addressV = nvrhi::SamplerAddressMode::Wrap;

    auto sampler = resource_allocator.create(sampler_desc);
    MARK_DESTROY_NVRHI_RESOURCE(sampler);

    auto random_seeds = params.get_input<nvrhi::BufferHandle>("Random Seeds");

    program_vars["SceneBVH"] = params.get_global_payload<RenderGlobalPayload&>()
                                   .InstanceCollection->get_tlas();
    program_vars["inPixelTarget"] =
        params.get_input<nvrhi::BufferHandle>("Pixel Target");
    program_vars["output"] = output;
    program_vars["random_seeds"] = random_seeds;
    for (int i = 0; i < 9; ++i) {
        program_vars["samplers"][i] = sampler;
    }

    auto rays = params.get_input<nvrhi::BufferHandle>("Rays");
    program_vars["rays"] = rays;

    auto env_prefilter_data = EnvironmentPrefilterData{};
    auto env_prefilter_cb = create_constant_buffer(params, env_prefilter_data);
    MARK_DESTROY_NVRHI_RESOURCE(env_prefilter_cb);
    program_vars["u_envPrefilterData"] = env_prefilter_cb;

    nvrhi::BufferDesc material_params_desc;
    // Each pixel should be able to store 288 bytes

    material_params_desc.byteSize =
        rays->getDesc().byteSize / sizeof(RayInfo) * sizeof(MaterialParams);
    material_params_desc.structStride = sizeof(MaterialParams);
    material_params_desc.canHaveUAVs = true;
    material_params_desc.initialState = nvrhi::ResourceStates::ShaderResource;
    material_params_desc.debugName = "materialParamsBuffer";
    material_params_desc.keepInitialState = true;
    auto material_params_buffer =
        resource_allocator.create(material_params_desc);

    MARK_DESTROY_NVRHI_RESOURCE(material_params_buffer);

    auto u_envRadiance = create_empty_texture(params, { 4, 4 });
    auto u_envIrradiance = create_empty_texture(params, { 4, 4 });
    MARK_DESTROY_NVRHI_RESOURCE(u_envRadiance);
    MARK_DESTROY_NVRHI_RESOURCE(u_envIrradiance);
    program_vars["u_envRadiance"] = u_envRadiance;
    program_vars["u_envIrradiance"] = u_envIrradiance;

    auto refraction_twosided = create_constant_buffer(params, 0u);
    MARK_DESTROY_NVRHI_RESOURCE(refraction_twosided);
    program_vars["u_refractionTwoSided"] = refraction_twosided;

    program_vars["u_envRadiance_sampler"] = sampler;
    program_vars["u_envIrradiance_sampler"] = sampler;

    //    program_vars["cb"] = create_constant_buffer(params, 1);

    program_vars["instanceDescBuffer"] =
        instance_collection->instance_pool.get_device_buffer();
    program_vars["meshDescBuffer"] =
        instance_collection->mesh_pool.get_device_buffer();

    program_vars["materialBlobBuffer"] =
        instance_collection->material_pool.get_device_buffer();
    program_vars["materialHeaderBuffer"] =
        instance_collection->material_header_pool.get_device_buffer();
    program_vars["materialParamsBuffer"] = material_params_buffer;

    // Bind light buffer - only include lights with valid paths
    auto& all_lights = global_payload.get_lights();
    std::vector<Hd_USTC_CG_Light*> valid_lights;

    for (auto* light : all_lights) {
        // Only include lights with non-empty paths (not fallback lights)
        if (light && !light->GetId().IsEmpty()) {
            valid_lights.push_back(light);
        }
    }

    uint32_t lightCount = static_cast<uint32_t>(valid_lights.size());

    instance_collection->light_pool.compress();
    program_vars["lightBuffer"] =
        instance_collection->light_pool.get_device_buffer();

    // Pass light count
    auto lightCountBuffer = create_constant_buffer(params, lightCount);
    MARK_DESTROY_NVRHI_RESOURCE(lightCountBuffer);
    program_vars["lightCount"] = lightCountBuffer;

    program_vars.set_descriptor_table(
        "t_BindlessBuffers",
        instance_collection->bindlessData.bufferDescriptorTableManager
            ->GetDescriptorTable(),
        instance_collection->bindlessData.bufferBindlessLayout);

    program_vars.set_descriptor_table(
        "t_BindlessTextures",
        instance_collection->bindlessData.textureDescriptorTableManager
            ->GetDescriptorTable(),
        instance_collection->bindlessData.textureBindlessLayout);
    program_vars.finish_setting_vars();

    RaytracingContext context(resource_allocator, program_vars);

    context.announce_raygeneration("RayGen");
    context.announce_hitgroup(
        "ClosestHit", "", "", 0);  // Primary ray hit group at index 0
    context.announce_hitgroup(
        "ShadowHit", "", "", 1);       // Shadow ray hit group at index 1
    context.announce_miss("Miss", 0);  // Primary ray miss shader at index 0
    context.announce_miss(
        "ShadowMiss", 1);  // Shadow ray miss shader at index 1

    // Register shared material evaluation callables at fixed indices
    // Pass nullptr for local root signature since these callables use global resources
    context.announce_callable(
        "eval_standard_surface",
        0,
        nullptr);  // shader_type_id = 0
    context.announce_callable(
        "eval_preview_surface", 1, nullptr);  // shader_type_id = 1
    context.announce_callable("eval_fallback", 2, nullptr);
    // Register per-material data fetch callables starting from index 2
    for (auto& callable : callable_shaders) {
        context.announce_callable(callable.second, 3 + callable.first, nullptr);
    }

    context.finish_announcing_shader_names();

    auto buffer_size = rays->getDesc().byteSize / sizeof(RayInfo);

    if (buffer_size > 0) {
        context.begin();
        context.trace_rays({}, program_vars, buffer_size, 1, 1);
        context.finish();
    }

    params.set_output("Output", output);

    return true;
}

NODE_DECLARATION_UI(path_tracing);
NODE_DEF_CLOSE_SCOPE

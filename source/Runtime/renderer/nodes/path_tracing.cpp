
#include "../source/renderTLAS.h"
#include "nodes/core/def/node_def.hpp"
#include "nvrhi/nvrhi.h"
#include "nvrhi/utils.h"
#include "render_node_base.h"
#include "renderer/raytracing_context.hpp"
#include "shaders/shaders/utils/HitObject.h"
#include "utils/math.h"
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
    program_desc.set_path(std::filesystem::path("shaders/path_tracing.slang"));
    program_desc.shaderType = nvrhi::ShaderType::AllRayTracing;
    program_desc.nvapi_support = true;
    program_desc.define(
        "FALCOR_MATERIAL_INSTANCE_SIZE",
        std::to_string(c_FalcorMaterialInstanceSize));

    auto &materials = global_payload.get_materials();

    std::unordered_map<unsigned, std::string> callable_shaders;

    for (auto material : materials) {
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
    // Function content omitted

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

    program_vars["SceneBVH"] =
        params.get_global_payload<RenderGlobalPayload &>()
            .InstanceCollection->get_tlas();
    program_vars["inPixelTarget"] =
        params.get_input<nvrhi::BufferHandle>("Pixel Target");
    program_vars["output"] = output;
    program_vars["random_seeds"] = random_seeds;
    program_vars["sampler"] = sampler;
    program_vars["rays"] = params.get_input<nvrhi::BufferHandle>("Rays");

    auto env_prefilter_data = EnvironmentPrefilterData{};
    auto env_prefilter_cb = create_constant_buffer(params, env_prefilter_data);
    MARK_DESTROY_NVRHI_RESOURCE(env_prefilter_cb);
    program_vars["u_envPrefilterData"] = env_prefilter_cb;

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

    program_vars["materialBuffer"] =
        instance_collection->material_pool.get_device_buffer();

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
    context.announce_hitgroup("ClosestHit");
    context.announce_hitgroup("ShadowHit", "", "", 1);
    context.announce_miss("Miss");
    context.announce_miss("ShadowMiss", 1);

    for (auto &callable : callable_shaders) {
        context.announce_callable(callable.second, callable.first);
    }

    context.finish_announcing_shader_names();

    auto rays = params.get_input<nvrhi::BufferHandle>("Rays");

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

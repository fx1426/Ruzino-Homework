
#include <spdlog/spdlog.h>

#include <memory>

#include "../source/renderTLAS.h"
#include "GPUContext/program_vars.hpp"
#include "GPUContext/raytracing_context.hpp"
#include "RHI/internal/resources.hpp"
#include "Scene/MaterialParamsBuffer.slang"
#include "nodes/core/def/node_def.hpp"
#include "nvrhi/nvrhi.h"
#include "nvrhi/utils.h"
#include "render_node_base.h"
#include "shaders/shaders/utils/HitObject.h"
#include "utils/math.h"

// A traditional path tracing node

using namespace USTC_CG;
struct PathTracingStorage {
    constexpr static bool has_storage = false;

    // Cached resources
    ProgramHandle raytracing_program;
    nvrhi::SamplerHandle sampler;
    nvrhi::BufferHandle material_params_buffer;
    nvrhi::BufferHandle light_count_buffer;
    nvrhi::TextureHandle output_texture;

    // Cached pipeline state - these are expensive to create
    std::unique_ptr<ProgramVars> cached_program_vars;
    std::unique_ptr<RaytracingContext> cached_rt_context;

    size_t last_ray_buffer_size = 0;
    uint32_t last_light_count = 0;
    nvrhi::TextureDesc last_output_desc;
    bool pipeline_dirty = true;  // Mark pipeline needs rebuild

    GfVec2i last_frame_size = GfVec2i(0, 0);

    ResourceAllocator* rs;

    // Destructor to clean up resources
    ~PathTracingStorage()
    {
        rs->destroy(raytracing_program);
        raytracing_program = nullptr;
        rs->destroy(sampler);
        sampler = nullptr;
        rs->destroy(material_params_buffer);
        material_params_buffer = nullptr;
        rs->destroy(light_count_buffer);
        light_count_buffer = nullptr;
        rs->destroy(output_texture);
        output_texture = nullptr;
        cached_program_vars.reset();
        cached_rt_context.reset();
    }
};

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(path_tracing)
{
    b.add_input<nvrhi::BufferHandle>("Pixel Target");
    b.add_input<nvrhi::BufferHandle>("Rays");
    b.add_input<nvrhi::BufferHandle>("Random Seeds");

    b.add_output<nvrhi::TextureHandle>("Output");

    // Function content omitted
}

NODE_EXECUTION_FUNCTION(path_tracing)
{
    using namespace nvrhi;

    auto& storage = params.get_storage<PathTracingStorage&>();
    storage.rs = &resource_allocator;

    // Check if program needs to be rebuilt
    bool materials_dirty = global_payload.is_dirty(
        RenderGlobalPayload::SceneDirtyBits::DirtyMaterials);
    bool geometry_dirty = global_payload.is_dirty(
        RenderGlobalPayload::SceneDirtyBits::DirtyGeometry);

    bool program_changed = !storage.raytracing_program || materials_dirty;

    // Geometry changes require pipeline rebuild due to potential buffer layout
    // changes
    if (geometry_dirty) {
        storage.pipeline_dirty = true;
    }

    std::unordered_map<unsigned, std::string> callable_shaders;

    if (program_changed) {
        // Build material callable shaders map
        auto& materials = global_payload.get_materials();

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
            callable_shaders[location] = material.second->GetMaterialName();
        }

        // Clean up old program if exists
        if (storage.raytracing_program) {
            resource_allocator.destroy(storage.raytracing_program);
        }

        ProgramDesc program_desc;
        program_desc.set_path("shaders/path_tracing.slang");
        program_desc.shaderType = nvrhi::ShaderType::AllRayTracing;
        program_desc.nvapi_support = true;

        for (auto material : materials) {
            if (material.second == nullptr) {
                continue;
            }
            auto location = material.second->GetMaterialLocation();
            if (location == -1) {
                continue;
            }
            program_desc.add_source_code(
                material.second->GetShader(shader_factory));
        }

        storage.raytracing_program = resource_allocator.create(program_desc);
        CHECK_PROGRAM_ERROR(storage.raytracing_program);
        storage.pipeline_dirty = true;  // Mark pipeline needs rebuild
    }

    // Create or reuse sampler
    if (!storage.sampler) {
        SamplerDesc sampler_desc;
        sampler_desc.addressU = nvrhi::SamplerAddressMode::Wrap;
        sampler_desc.addressV = nvrhi::SamplerAddressMode::Wrap;
        storage.sampler = resource_allocator.create(sampler_desc);
    }

    // Check output texture
    auto camera = get_free_camera(params);
    auto size = camera->dataWindow.GetSize();

    bool frame_size_changed =
        (storage.last_frame_size[0] != size[0] ||
         storage.last_frame_size[1] != size[1]);

    if (frame_size_changed) {
        storage.last_frame_size = GfVec2i(size[0], size[1]);
        storage.pipeline_dirty = true;
    }

    if (!storage.output_texture || storage.last_output_desc.width != size[0] ||
        storage.last_output_desc.height != size[1]) {
        nvrhi::TextureDesc output_desc;
        output_desc.width = size[0];
        output_desc.height = size[1];
        output_desc.format = nvrhi::Format::RGBA32_FLOAT;
        output_desc.isRenderTarget = true;
        output_desc.isUAV = true;
        output_desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        output_desc.keepInitialState = true;
        output_desc.debugName = "PathTracingOutput";
        if (storage.output_texture) {
            resource_allocator.destroy(storage.output_texture);
        }
        storage.output_texture = resource_allocator.create(output_desc);
        storage.last_output_desc = output_desc;
    }

    auto rays = params.get_input<nvrhi::BufferHandle>("Rays");
    size_t ray_buffer_size =
        rays->getDesc().byteSize / sizeof(RayInfo) * sizeof(MaterialParams);

    // Check if material params buffer needs resize
    if (!storage.material_params_buffer ||
        storage.last_ray_buffer_size != ray_buffer_size) {
        if (storage.material_params_buffer) {
            resource_allocator.destroy(storage.material_params_buffer);
        }

        nvrhi::BufferDesc material_params_desc;
        material_params_desc.byteSize = ray_buffer_size;
        material_params_desc.structStride = sizeof(MaterialParams);
        material_params_desc.canHaveUAVs = true;
        material_params_desc.initialState =
            nvrhi::ResourceStates::ShaderResource;
        material_params_desc.debugName = "materialParamsBuffer";
        material_params_desc.keepInitialState = true;

        storage.material_params_buffer =
            resource_allocator.create(material_params_desc);
        storage.last_ray_buffer_size = ray_buffer_size;
    }

    // Count valid lights
    auto& all_lights = global_payload.get_lights();
    uint32_t lightCount = 0;
    for (auto* light : all_lights) {
        if (light && !light->GetId().IsEmpty()) {
            lightCount++;
        }
    }

    // Check if light count buffer needs update
    if (!storage.light_count_buffer || storage.last_light_count != lightCount) {
        if (storage.light_count_buffer) {
            resource_allocator.destroy(storage.light_count_buffer);
        }
        storage.light_count_buffer = create_constant_buffer(params, lightCount);
        storage.last_light_count = lightCount;
    }

    // Setup or reuse program vars and context
    if (storage.pipeline_dirty || !storage.cached_program_vars ||
        !storage.cached_rt_context) {
        // Rebuild everything when pipeline is dirty
        storage.cached_program_vars = std::make_unique<ProgramVars>(
            resource_allocator, storage.raytracing_program);
        auto& program_vars = *storage.cached_program_vars;

        auto random_seeds =
            params.get_input<nvrhi::BufferHandle>("Random Seeds");

        program_vars["SceneBVH"] =
            params.get_global_payload<RenderGlobalPayload&>()
                .InstanceCollection->get_tlas();
        program_vars["inPixelTarget"] =
            params.get_input<nvrhi::BufferHandle>("Pixel Target");
        program_vars["output"] = storage.output_texture;
        program_vars["random_seeds"] = random_seeds;

        for (int i = 0; i < 9; ++i) {
            program_vars["samplers"][i] = storage.sampler;
        }

        program_vars["rays"] = rays;

        program_vars["instanceDescBuffer"] =
            instance_collection->instance_pool.get_device_buffer();
        program_vars["meshDescBuffer"] =
            instance_collection->mesh_pool.get_device_buffer();

        program_vars["materialBlobBuffer"] =
            instance_collection->material_pool.get_device_buffer();
        program_vars["materialHeaderBuffer"] =
            instance_collection->material_header_pool.get_device_buffer();
        program_vars["materialParamsBuffer"] = storage.material_params_buffer;

        instance_collection->light_pool.compress();
        program_vars["lightBuffer"] =
            instance_collection->light_pool.get_device_buffer();
        program_vars["lightCount"] = storage.light_count_buffer;

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

        storage.cached_rt_context = std::make_unique<RaytracingContext>(
            resource_allocator, program_vars);
        auto& context = *storage.cached_rt_context;

        context.announce_raygeneration("RayGen");
        context.announce_hitgroup("ClosestHit", "", "", 0);
        context.announce_hitgroup("ShadowHit", "", "", 1);
        context.announce_miss("Miss", 0);
        context.announce_miss("ShadowMiss", 1);

        context.announce_callable("eval_standard_surface", 0, nullptr);
        context.announce_callable("eval_preview_surface", 1, nullptr);
        context.announce_callable("eval_fallback", 2, nullptr);

        for (auto& callable : callable_shaders) {
            context.announce_callable(
                callable.second, 3 + callable.first, nullptr);
        }

        context.finish_announcing_shader_names();
        storage.pipeline_dirty = false;
    }
    // else: Fast path - reuse everything from cache, no updates needed

    // Use the cached context and program vars
    auto& context = *storage.cached_rt_context;
    auto& program_vars = *storage.cached_program_vars;

    auto buffer_size = rays->getDesc().byteSize / sizeof(RayInfo);

    if (buffer_size > 0) {
        context.begin();
        context.trace_rays({}, program_vars, buffer_size, 1, 1);
        context.finish();
    }

    params.set_output("Output", storage.output_texture);

    return true;
}

NODE_DECLARATION_UI(path_tracing);
NODE_DEF_CLOSE_SCOPE

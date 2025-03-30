#include "raytracing_context.hpp"

USTC_CG_NAMESPACE_OPEN_SCOPE
RaytracingContext::RaytracingContext(ResourceAllocator& r, ProgramVars& vars)
    : GPUContext(r, vars)
{
    program = vars.get_programs()[0];
}

RaytracingContext::~RaytracingContext()
{
    resource_allocator_.destroy(ray_generation_shader);
    for (auto& hitgroup : hit_group_shaders) {
        resource_allocator_.destroy(std::get<0>(hitgroup));
        resource_allocator_.destroy(std::get<1>(hitgroup));
        resource_allocator_.destroy(std::get<2>(hitgroup));
    }
    for (auto& callable : callable_shaders) {
        resource_allocator_.destroy(callable);
    }
    for (auto& miss : miss_shaders) {
        resource_allocator_.destroy(miss);
    }
    resource_allocator_.destroy(raytracing_pipeline);
    resource_allocator_.destroy(sbt);
    resource_allocator_.destroy(program);
}

void RaytracingContext::begin()
{
    GPUContext::begin();
}

void RaytracingContext::finish()
{
    GPUContext::finish();
}

void RaytracingContext::trace_rays(
    const RaytracingState& state,
    const ProgramVars& program_vars,
    uint32_t width,
    uint32_t height,
    uint32_t depth) const
{
    nvrhi::rt::State rt_state;
    rt_state.bindings = program_vars.get_binding_sets();
    rt_state.shaderTable = sbt;

    commandList_->setRayTracingState(rt_state);

    nvrhi::rt::DispatchRaysArguments args;
    args.width = width;
    args.height = height;
    args.depth = depth;
    commandList_->dispatchRays(args);
}

void RaytracingContext::announce_raygeneration(
    const std::string& name,
    ProgramVarHandle handle)
{
    raygeneration_name = name;
    ray_generation_program = handle;
}

void RaytracingContext::announce_hitgroup(
    const std::string& closesthit,
    const std::string& anyhit,
    const std::string& intercestion,
    unsigned position,
    ProgramVarHandle handle_hg)
{
    if (hitgroup_names.size() <= position) {
        hitgroup_names.resize(position + 1);
    }
    hitgroup_names[position] =
        std::make_tuple(closesthit, anyhit, intercestion);

    if (hitgroup_programs.size() <= position) {
        hitgroup_programs.resize(position + 1);
    }

    hitgroup_programs[position] = handle_hg;
}

void RaytracingContext::announce_callable(
    const std::string& name,
    unsigned position,
    ProgramVarHandle handle)
{
    if (callable_names.size() <= position) {
        callable_names.resize(position + 1);
    }
    callable_names[position] = name;

    if (callable_programs.size() <= position) {
        callable_programs.resize(position + 1);
    }
    callable_programs[position] = handle;
}

void RaytracingContext::announce_miss(
    const std::string& name,
    unsigned position,
    ProgramVarHandle handle)
{
    if (miss_names.size() <= position) {
        miss_names.resize(position + 1);
    }
    miss_names[position] = name;

    if (miss_programs.size() <= position) {
        miss_programs.resize(position + 1);
    }
    miss_programs[position] = handle;
}

void RaytracingContext::finish_announcing_shader_names()
{
    // prepare the shaders
    resource_allocator_.destroy(ray_generation_shader);
    for (auto& hitgroup : hit_group_shaders) {
        resource_allocator_.destroy(std::get<0>(hitgroup));
        resource_allocator_.destroy(std::get<1>(hitgroup));
        resource_allocator_.destroy(std::get<2>(hitgroup));
    }
    for (auto& callable : callable_shaders) {
        resource_allocator_.destroy(callable);
    }

    for (auto& miss : miss_shaders) {
        resource_allocator_.destroy(miss);
    }

    hit_group_shaders.clear();
    callable_shaders.clear();
    miss_shaders.clear();

    auto local_program = program;

    if (ray_generation_program)
        local_program = ray_generation_program->get_programs()[0];

    nvrhi::ShaderDesc raygen_shader_desc;
    raygen_shader_desc.entryName = raygeneration_name.c_str();
    raygen_shader_desc.shaderType = nvrhi::ShaderType::RayGeneration;
    raygen_shader_desc.debugName = std::to_string(
        reinterpret_cast<long long>(local_program->getBufferPointer()));
    ray_generation_shader = resource_allocator_.create(
        raygen_shader_desc,
        local_program->getBufferPointer(),
        local_program->getBufferSize());

    for (int i = 0; i < hitgroup_names.size(); ++i) {
        auto hitgroup = hitgroup_names[i];

        local_program = program;
        if (hitgroup_programs[i])
            local_program = hitgroup_programs[i]->get_programs()[0];

        nvrhi::ShaderDesc chs_desc;
        chs_desc.entryName = std::get<0>(hitgroup).c_str();
        chs_desc.shaderType = nvrhi::ShaderType::ClosestHit;
        chs_desc.debugName = std::to_string(
            reinterpret_cast<long long>(local_program->getBufferPointer()));
        assert(!chs_desc.entryName.empty());
        auto chs_shader = resource_allocator_.create(
            chs_desc,
            local_program->getBufferPointer(),
            local_program->getBufferSize());

        nvrhi::ShaderDesc ahs_desc;
        ahs_desc.entryName = std::get<1>(hitgroup).c_str();
        ahs_desc.shaderType = nvrhi::ShaderType::AnyHit;
        ahs_desc.debugName = std::to_string(
            reinterpret_cast<long long>(local_program->getBufferPointer()));

        nvrhi::ShaderHandle ahs_shader = nullptr;
        if (!ahs_desc.entryName.empty()) {
            ahs_shader = resource_allocator_.create(
                ahs_desc,
                local_program->getBufferPointer(),
                local_program->getBufferSize());
        }

        nvrhi::ShaderDesc is_desc;
        is_desc.entryName = std::get<2>(hitgroup).c_str();
        is_desc.shaderType = nvrhi::ShaderType::Intersection;
        is_desc.debugName = std::to_string(
            reinterpret_cast<long long>(local_program->getBufferPointer()));

        nvrhi::ShaderHandle is_shader = nullptr;
        if (!is_desc.entryName.empty()) {
            is_shader = resource_allocator_.create(
                is_desc,
                local_program->getBufferPointer(),
                local_program->getBufferSize());
        }

        hit_group_shaders.push_back(
            std::make_tuple(chs_shader, ahs_shader, is_shader));
    }

    for (int i = 0; i < callable_names.size(); ++i) {
        auto callable = callable_names[i];

        bool is_placeholder = callable.empty();

        if (is_placeholder) {
            callable_shaders.push_back(nullptr);
            continue;
        }

        local_program = program;

        if (callable_programs[i])
            local_program = callable_programs[i]->get_programs()[0];

        nvrhi::ShaderDesc callable_desc;
        callable_desc.entryName = callable.c_str();
        callable_desc.shaderType = nvrhi::ShaderType::Callable;
        callable_desc.debugName = std::to_string(
            reinterpret_cast<long long>(local_program->getBufferPointer()));
        auto callable_shader = resource_allocator_.create(
            callable_desc,
            local_program->getBufferPointer(),
            local_program->getBufferSize());
        callable_shaders.push_back(callable_shader);
    }

    for (int i = 0; i < miss_names.size(); ++i) {
        auto miss = miss_names[i];
        local_program = program;

        if (miss_programs[i])
            local_program = miss_programs[i]->get_programs()[0];

        nvrhi::ShaderDesc miss_desc;
        miss_desc.entryName = miss.c_str();
        miss_desc.shaderType = nvrhi::ShaderType::Miss;
        miss_desc.debugName = std::to_string(
            reinterpret_cast<long long>(local_program->getBufferPointer()));
        auto miss_shader = resource_allocator_.create(
            miss_desc,
            local_program->getBufferPointer(),
            local_program->getBufferSize());
        miss_shaders.push_back(miss_shader);
    }

    // create the pipeline
    nvrhi::rt::PipelineDesc pipeline_desc;
    pipeline_desc.maxPayloadSize = 16 * sizeof(float);
    pipeline_desc.globalBindingLayouts = vars_.get_binding_layout();
    pipeline_desc.maxRecursionDepth = 31;
    pipeline_desc.maxAttributeSize = 4 * sizeof(float);

    pipeline_desc.shaders = { { "Raygen", ray_generation_shader, nullptr } };

    for (size_t i = 0; i < hit_group_shaders.size(); ++i) {
        std::string hit_group_export_name = "HitGroup" + std::to_string(i);
        pipeline_desc.hitGroups.push_back({ hit_group_export_name,
                                            std::get<0>(hit_group_shaders[i]),
                                            std::get<1>(hit_group_shaders[i]),
                                            std::get<2>(hit_group_shaders[i]),
                                            nullptr

        });
    }

    // callable shaders
    for (size_t i = 0; i < callable_shaders.size(); ++i) {
        std::string callable_export_name = "Callable" + std::to_string(i);
        pipeline_desc.shaders.push_back(
            { callable_export_name, callable_shaders[i], nullptr

            });
    }

    // miss shaders
    for (size_t i = 0; i < miss_shaders.size(); ++i) {
        std::string miss_export_name = "Miss" + std::to_string(i);
        pipeline_desc.shaders.push_back(
            { miss_export_name, miss_shaders[i], nullptr });
    }

    resource_allocator_.destroy(raytracing_pipeline);
    raytracing_pipeline = resource_allocator_.create(pipeline_desc);

    sbt = raytracing_pipeline->createShaderTable();
    sbt->setRayGenerationShader("Raygen");
    for (size_t i = 0; i < hit_group_shaders.size(); ++i) {
        std::string hit_group_export_name = "HitGroup" + std::to_string(i);
        sbt->addHitGroup(hit_group_export_name.c_str());
    }

    for (size_t i = 0; i < callable_shaders.size(); ++i) {
        std::string callable_export_name = "Callable" + std::to_string(i);
        sbt->addCallableShader(callable_export_name.c_str());
    }

    for (size_t i = 0; i < miss_shaders.size(); ++i) {
        std::string miss_export_name = "Miss" + std::to_string(i);
        sbt->addMissShader(miss_export_name.c_str());
    }
}

USTC_CG_NAMESPACE_CLOSE_SCOPE
#pragma once
#include <nvrhi/nvrhi.h>
#include <pxr/base/gf/matrix4f.h>
#include <pxr/base/gf/vec2i.h>

#include <hd_RUZINO/render_global_payload.hpp>

#include "api.h"
#include "nodes/core/node_exec.hpp"
#include "utils/resource_cleaner.hpp"
RUZINO_NAMESPACE_OPEN_SCOPE
class Hd_RUZINO_Camera;
HD_RUZINO_API Hd_RUZINO_Camera* get_free_camera(
    ExeParams& params,
    const std::string& camera_name = "Camera");

#define global_payload      params.get_global_payload<RenderGlobalPayload&>()
#define instance_collection global_payload.InstanceCollection
#define resource_allocator  get_resource_allocator(params)
#define shader_factory      get_shader_factory(params)

HD_RUZINO_API ResourceAllocator& get_resource_allocator(ExeParams& params);
HD_RUZINO_API ShaderFactory& get_shader_factory(ExeParams& params);
HD_RUZINO_API BufferHandle get_free_camera_planarview_cb(
    ExeParams& params,
    const std::string& camera_name = "Camera");
HD_RUZINO_API BufferHandle
get_model_buffer(ExeParams& params, const pxr::GfMatrix4f& matrix);

template<typename T>
inline BufferHandle create_buffer(
    ExeParams& params,
    size_t count,
    bool is_constant_buffer = false,
    bool is_uav_buffer = false,
    bool isVertexBuffer = false)
{
    nvrhi::BufferDesc buffer_desc = nvrhi::BufferDesc();
    buffer_desc.byteSize = count * sizeof(T);
    buffer_desc.isVertexBuffer = isVertexBuffer;
    buffer_desc.initialState = nvrhi::ResourceStates::ShaderResource;
    buffer_desc.debugName = type_name<T>().data();
    buffer_desc.structStride = sizeof(T);
    buffer_desc.keepInitialState = true;

    if (is_constant_buffer) {
        buffer_desc.isConstantBuffer = true;
        buffer_desc.initialState = nvrhi::ResourceStates::ConstantBuffer;
        buffer_desc.cpuAccess = nvrhi::CpuAccessMode::Write;
    }

    if (is_uav_buffer) {
        buffer_desc.canHaveUAVs = true;
        buffer_desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
    }

    auto buffer = resource_allocator.create(buffer_desc);

    return buffer;
}

template<typename T>
inline BufferHandle create_buffer(
    ExeParams& params,
    size_t count,
    const T& init_value,
    bool is_constant_buffer = false,
    bool is_uav_buffer = false)
{
    auto buffer =
        create_buffer<T>(params, count, is_constant_buffer, is_uav_buffer);

    // fill the buffer with default values
    std::vector<T> cpu_data(count, init_value);
    auto ptr = resource_allocator.device->mapBuffer(
        buffer, nvrhi::CpuAccessMode::Write);

    memcpy(ptr, cpu_data.data(), cpu_data.size() * sizeof(T));

    resource_allocator.device->unmapBuffer(buffer);

    return buffer;
}

template<typename T>
inline BufferHandle create_constant_buffer(ExeParams& params, const T& value)
{
    return create_buffer<T>(params, 1, value, true);
}

template<typename T>
inline BufferHandle create_uav_buffer(  // unordered access view
    ExeParams& params,
    size_t count,
    const T& init_value)
{
    return create_buffer<T>(params, count, init_value, false, true);
}

template<typename T>
inline std::tuple<nvrhi::BufferHandle, nvrhi::BufferHandle>
create_counter_buffer(ExeParams& params, size_t max_size)
{
    nvrhi::BufferDesc storage_buffer = nvrhi::BufferDesc();
    storage_buffer.byteSize = max_size * sizeof(T);
    storage_buffer.initialState = nvrhi::ResourceStates::UnorderedAccess;
    storage_buffer.debugName = type_name<T>().data();
    storage_buffer.cpuAccess = nvrhi::CpuAccessMode::Write;
    storage_buffer.canHaveUAVs = true;
    storage_buffer.structStride = sizeof(T);
    auto buffer = resource_allocator.create(storage_buffer);

    nvrhi::BufferDesc counter_buffer = nvrhi::BufferDesc();
    counter_buffer.byteSize = sizeof(uint32_t);
    counter_buffer.isVertexBuffer = true;
    counter_buffer.initialState = nvrhi::ResourceStates::UnorderedAccess;
    counter_buffer.debugName = "counterBuffer";
    counter_buffer.cpuAccess = nvrhi::CpuAccessMode::Write;
    counter_buffer.structStride = sizeof(uint32_t);
    counter_buffer.canHaveUAVs = true;
    auto counter = resource_allocator.create(counter_buffer);

    // fill the counter with 0
    uint32_t zero = 0;
    auto ptr = resource_allocator.device->mapBuffer(
        counter, nvrhi::CpuAccessMode::Write);
    memcpy(ptr, &zero, sizeof(uint32_t));
    resource_allocator.device->unmapBuffer(counter);

    return { buffer, counter };
}

HD_RUZINO_API unsigned counter_read_out(
    ExeParams& params,
    nvrhi::IBuffer* counter);

HD_RUZINO_API TextureHandle create_default_render_target(
    ExeParams& params,
    nvrhi::Format format = nvrhi::Format::RGBA16_FLOAT);

HD_RUZINO_API void initialize_texture(
    ExeParams& params,
    nvrhi::ITexture* texture,
    const nvrhi::Color& color);

HD_RUZINO_API TextureHandle create_default_depth_stencil(ExeParams& params);

HD_RUZINO_API TextureHandle create_empty_texture(
    ExeParams& params,
    const pxr::GfVec2i& size,
    nvrhi::Format format = nvrhi::Format::RGBA16_FLOAT);

HD_RUZINO_API pxr::GfVec2i get_size(ExeParams& params);

#define CHECK_PROGRAM_ERROR(program)              \
    if (!program->get_error_string().empty()) {   \
        spdlog::error(                            \
            "Failed to create shader {}: {}",     \
            #program,                             \
            program->get_error_string().c_str()); \
        resource_allocator.destroy(program);      \
        (program) = nullptr;                      \
        return false;                             \
    }

#ifdef _DEBUG
#define PROFILE_SCOPE(node_name) auto scope = log::profile_scope(#node_name)
#else
#define PROFILE_SCOPE(node_name)
#endif

RUZINO_NAMESPACE_CLOSE_SCOPE

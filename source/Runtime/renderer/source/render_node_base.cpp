#include <hd_RUZINO/render_node_base.h>
#include <nvrhi/nvrhi.h>
#include <pxr/base/vt/array.h>

#include "RHI/ResourceManager/resource_allocator.hpp"
#include "RHI/internal/resources.hpp"
#include "camera.h"
#include "hd_RUZINO/render_global_payload.hpp"
#include "light.h"
#include "nodes/core/node_exec.hpp"
#include "utils/cam_to_view_contants.h"
#include "utils/view_cb.h"

// Undefine macros to avoid conflicts with member variable names
#undef resource_allocator
#undef shader_factory

namespace Ruzino {
Hd_RUZINO_Camera* get_free_camera(
    ExeParams& params,
    const std::string& camera_name)
{
    auto& cameras =
        params.get_global_payload<RenderGlobalPayload&>().get_cameras();

    Hd_RUZINO_Camera* free_camera = nullptr;
    for (auto camera : cameras) {
        if (camera->GetId() != SdfPath::EmptyPath()) {
            free_camera = camera;
            break;
        }
    }
    return free_camera;
}

ResourceAllocator& get_resource_allocator(ExeParams& params)
{
    return params.get_global_payload<RenderGlobalPayload&>().resource_allocator;
}

ShaderFactory& get_shader_factory(ExeParams& params)
{
    return params.get_global_payload<RenderGlobalPayload&>().shader_factory;
}

BufferHandle get_free_camera_planarview_cb(
    ExeParams& params,
    const std::string& camera_name)
{
    auto free_camera = get_free_camera(params, camera_name);
    auto& resource_allocator = get_resource_allocator(params);
    auto constant_buffer = resource_allocator.create(
        nvrhi::BufferDesc{ .byteSize = sizeof(PlanarViewConstants),
                           .debugName = "constantBuffer",
                           .isConstantBuffer = true,
                           .initialState =
                               nvrhi::ResourceStates::ConstantBuffer,
                           .cpuAccess = nvrhi::CpuAccessMode::Write });

    PlanarViewConstants view_constant = camera_to_view_constants(free_camera);
    auto mapped_constant_buffer = resource_allocator.device->mapBuffer(
        constant_buffer, nvrhi::CpuAccessMode::Write);
    memcpy(mapped_constant_buffer, &view_constant, sizeof(PlanarViewConstants));
    resource_allocator.device->unmapBuffer(constant_buffer);
    return constant_buffer;
}

BufferHandle get_model_buffer(ExeParams& params, const pxr::GfMatrix4f& matrix)
{
    auto& resource_allocator = get_resource_allocator(params);
    auto desc = nvrhi::BufferDesc{ .byteSize = sizeof(pxr::GfMatrix4f),
                                   .debugName = "modelBuffer",
                                   .isConstantBuffer = true,
                                   .initialState =
                                       nvrhi::ResourceStates::ConstantBuffer,
                                   .cpuAccess = nvrhi::CpuAccessMode::Write };
    desc.structStride = sizeof(pxr::GfMatrix4f);
    auto model_buffer = resource_allocator.create(desc);

    auto mapped_model_buffer = resource_allocator.device->mapBuffer(
        model_buffer, nvrhi::CpuAccessMode::Write);
    memcpy(mapped_model_buffer, &matrix, sizeof(pxr::GfMatrix4f));
    resource_allocator.device->unmapBuffer(model_buffer);
    return model_buffer;
}

unsigned counter_read_out(ExeParams& params, nvrhi::IBuffer* counter)
{
    auto& resource_allocator = get_resource_allocator(params);
    uint32_t count = 0;
    resource_allocator.device->waitForIdle();
    auto ptr = resource_allocator.device->mapBuffer(
        counter, nvrhi::CpuAccessMode::Read);
    memcpy(&count, ptr, sizeof(uint32_t));
    resource_allocator.device->unmapBuffer(counter);
    return count;
}

TextureHandle create_default_render_target(
    ExeParams& params,
    nvrhi::Format format)
{
    auto camera = get_free_camera(params);
    auto size = camera->dataWindow.GetSize();
    auto& resource_allocator = get_resource_allocator(params);
    // Output texture
    nvrhi::TextureDesc desc =
        nvrhi::TextureDesc{}
            .setWidth(size[0])
            .setHeight(size[1])
            .setFormat(format)
            .setIsUAV(true)
            .setInitialState(nvrhi::ResourceStates::RenderTarget)
            .setKeepInitialState(true)
            .setClearValue(nvrhi::Color(0.0f, 0.0f, 0.0f, 1.0f))
            .setUseClearValue(true)
            .setIsRenderTarget(true);
    auto output_texture = resource_allocator.create(desc);
    return output_texture;
}

void initialize_texture(
    ExeParams& params,
    nvrhi::ITexture* texture,
    const nvrhi::Color& color)
{
    auto& resource_allocator = get_resource_allocator(params);
    auto command_list = resource_allocator.create(nvrhi::CommandListDesc{});
    command_list->open();
    command_list->clearTextureFloat(texture, {}, color);
    command_list->close();
    resource_allocator.device->executeCommandList(command_list);
    resource_allocator.destroy(command_list);
}

TextureHandle create_default_depth_stencil(ExeParams& params)
{
    auto camera = get_free_camera(params);
    auto size = camera->dataWindow.GetSize();
    auto& resource_allocator = get_resource_allocator(params);
    // Depth texture
    nvrhi::TextureDesc depth_desc =
        nvrhi::TextureDesc{}
            .setWidth(size[0])
            .setHeight(size[1])
            .setFormat(nvrhi::Format::D32)
            .setIsRenderTarget(true)
            .setInitialState(nvrhi::ResourceStates::DepthWrite)
            .setKeepInitialState(true);
    auto depth_stencil_texture = resource_allocator.create(depth_desc);
    return depth_stencil_texture;
}

TextureHandle create_empty_texture(
    ExeParams& params,
    const pxr::GfVec2i& size,
    nvrhi::Format format)
{
    auto& resource_allocator = get_resource_allocator(params);
    nvrhi::TextureDesc desc =
        nvrhi::TextureDesc{}
            .setWidth(size[0])
            .setHeight(size[1])
            .setFormat(format)
            .setInitialState(nvrhi::ResourceStates::ShaderResource)
            .setKeepInitialState(true);
    auto texture = resource_allocator.create(desc);
    return texture;
}

GfVec2i get_size(ExeParams& params)
{
    auto camera = get_free_camera(params);
    auto size = camera->dataWindow.GetSize();
    return size;
}
}  // namespace Ruzino
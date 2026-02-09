#include <cstring>

#include "hd_RUZINO/render_node_base.h"
#include "nodes/core/def/node_def.hpp"

#if RUZINO_WITH_CUDA
#include "RHI/internal/cuda_extension.hpp"
#include "hd_RUZINO/render_global_payload.hpp"

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(nvrhi_to_cuda)
{
    b.add_input<nvrhi::TextureHandle>("Texture");
    b.add_output<Ruzino::cuda::CUDALinearBufferHandle>("Buffer");
}

NODE_EXECUTION_FUNCTION(nvrhi_to_cuda)
{
    auto texture = params.get_input<nvrhi::TextureHandle>("Texture");

    if (!texture) {
        throw std::runtime_error("Invalid input texture");
    }

    auto device = global_payload.nvrhi_device;

    const auto& desc = texture->getDesc();

    // Determine element size based on format
    uint32_t element_size = 0;
    switch (desc.format) {
        case nvrhi::Format::RGBA32_FLOAT:
        case nvrhi::Format::RGBA32_UINT:
        case nvrhi::Format::RGBA32_SINT: element_size = 16; break;
        case nvrhi::Format::RGB32_FLOAT:
        case nvrhi::Format::RGB32_UINT:
        case nvrhi::Format::RGB32_SINT: element_size = 12; break;
        case nvrhi::Format::RG32_FLOAT:
        case nvrhi::Format::RG32_UINT:
        case nvrhi::Format::RG32_SINT: element_size = 8; break;
        case nvrhi::Format::R32_FLOAT:
        case nvrhi::Format::R32_UINT:
        case nvrhi::Format::R32_SINT: element_size = 4; break;
        default:
            element_size = 16;  // Default to RGBA32
            break;
    }

    // Convert texture to CUDA linear buffer
    auto buffer = Ruzino::cuda::copy_texture_to_linear_buffer_with_cleanup(
        device, texture.Get(), element_size);

    params.set_output("Buffer", buffer);
    return true;
}

NODE_DEF_CLOSE_SCOPE
#endif  // RUZINO_WITH_CUDA

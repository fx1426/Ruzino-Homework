#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// Framework includes
#include <spdlog/spdlog.h>

#include "GCore/GOP.h"
#include "GCore/algorithms/intersection.h"
#include "RHI/rhi.hpp"
#include "cmdparser.hpp"
#include "nodes/system/node_system.hpp"
#include "stage/stage.hpp"
#include "usd_nodejson.hpp"


// USD includes
#include "pxr/usd/usd/primRange.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usdGeom/camera.h"

// Hydra includes
#include "pxr/base/gf/camera.h"
#include "pxr/base/gf/frustum.h"
#include "pxr/imaging/hd/driver.h"
#include "pxr/imaging/hd/tokens.h"
#include "pxr/imaging/hdx/tokens.h"
#include "pxr/imaging/hgi/blitCmdsOps.h"
#include "pxr/imaging/hgi/tokens.h"
#include "pxr/usdImaging/usdImagingGL/engine.h"

// NVRHI includes
#include "nvrhi/nvrhi.h"

// Image saving
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "pxr/imaging/garch/glApi.h"
#include "stb_image_write.h"

// USD Hio for HDR/EXR support
#include "pxr/imaging/hio/image.h"
#include "pxr/imaging/hio/types.h"

#ifdef _WIN32
#include <gl/GL.h>
#include <windows.h>

#endif

using namespace USTC_CG;
using namespace pxr;

// USD utilities
UsdGeomCamera GetCamera(
    const UsdStageRefPtr& stage,
    const std::string& camera_path)
{
    // If camera_path is specified, try to use it
    if (!camera_path.empty()) {
        SdfPath path(camera_path);
        UsdPrim prim = stage->GetPrimAtPath(path);
        if (prim && prim.IsA<UsdGeomCamera>()) {
            spdlog::info("Using specified camera: {}", camera_path);
            return UsdGeomCamera(prim);
        }
        else {
            spdlog::warn(
                "Specified camera path '{}' not found or not a camera, falling "
                "back to first camera",
                camera_path);
        }
    }

    // Fall back to first camera
    for (const UsdPrim& prim : stage->Traverse()) {
        if (prim.IsA<UsdGeomCamera>()) {
            spdlog::info("Found camera: {}", prim.GetPath().GetString());
            return UsdGeomCamera(prim);
        }
    }
    return UsdGeomCamera();
}

// Graphics context initialization
void CreateGLContext()
{
#ifdef _WIN32
    HDC hdc = GetDC(GetConsoleWindow());
    PIXELFORMATDESCRIPTOR pfd = {};
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 24;

    int pixelFormat = ChoosePixelFormat(hdc, &pfd);
    SetPixelFormat(hdc, pixelFormat, &pfd);

    HGLRC hglrc = wglCreateContext(hdc);
    wglMakeCurrent(hdc, hglrc);
#endif
}

// Image utilities
std::string GetFileExtension(const std::string& filename)
{
    size_t pos = filename.find_last_of('.');
    if (pos == std::string::npos) {
        return "";
    }
    std::string ext = filename.substr(pos + 1);
    // Convert to lowercase
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext;
}

bool SaveImageToFile(
    const std::string& filename,
    int width,
    int height,
    const std::vector<uint8_t>& data)
{
    const float* float_data = reinterpret_cast<const float*>(data.data());
    std::string ext = GetFileExtension(filename);

    // Check if it's a HDR/EXR format
    if (ext == "exr" || ext == "hdr") {
        spdlog::info("Saving as HDR format: {}", ext);

        // Create flipped float data
        std::vector<float> flipped_data(width * height * 4);
        for (int y = 0; y < height; ++y) {
            int flipped_y = height - 1 - y;
            for (int x = 0; x < width; ++x) {
                for (int c = 0; c < 4; ++c) {
                    int src_idx = (y * width + x) * 4 + c;
                    int dst_idx = (flipped_y * width + x) * 4 + c;
                    flipped_data[dst_idx] = float_data[src_idx];
                }
            }
        }

        // Use USD Hio to save HDR/EXR
        HioImage::StorageSpec storage;
        storage.width = width;
        storage.height = height;
        storage.format = HioFormatFloat32Vec4;
        storage.flipped = false;  // Already flipped above
        storage.data = flipped_data.data();

        auto image = HioImage::OpenForWriting(filename);
        if (!image) {
            spdlog::error("Could not create image output for {}", filename);
            return false;
        }

        if (!image->Write(storage)) {
            spdlog::error("Failed to write HDR image to {}", filename);
            return false;
        }

        spdlog::info("Successfully saved HDR image to {}", filename);
        return true;
    }
    else {
        // PNG and other LDR formats - use STB
        spdlog::info("Saving as LDR format (PNG)");

        std::vector<uint8_t> rgba_data(width * height * 4);

        // Flip the image vertically while converting from float to uint8
        for (int y = 0; y < height; ++y) {
            int flipped_y = height - 1 - y;
            for (int x = 0; x < width; ++x) {
                for (int c = 0; c < 4; ++c) {
                    int src_idx = (y * width + x) * 4 + c;
                    int dst_idx = (flipped_y * width + x) * 4 + c;
                    rgba_data[dst_idx] = static_cast<uint8_t>(
                        std::clamp(float_data[src_idx] * 255.0f, 0.0f, 255.0f));
                }
            }
        }

        return stbi_write_png(
                   filename.c_str(),
                   width,
                   height,
                   4,
                   rgba_data.data(),
                   width * 4) != 0;
    }
}
// Texture reading methods
bool ReadTextureDirectly(
    UsdImagingGLEngine* renderer,
    int width,
    int height,
    std::vector<uint8_t>& texture_data)
{
    auto hacked_handle =
        renderer->GetRendererSetting(pxr::TfToken("VulkanColorAov"));
    if (!hacked_handle.IsHolding<const void*>()) {
        return false;
    }

    spdlog::info("Using direct texture copy method...");

    auto bare_pointer = hacked_handle.Get<const void*>();
    auto texture =
        *static_cast<nvrhi::ITexture**>(const_cast<void*>(bare_pointer));

    texture_data.resize(width * height * 4 * sizeof(float));

    // Create staging texture once and reuse command list
    static nvrhi::StagingTextureHandle staging_texture;
    static nvrhi::CommandListHandle command_list;

    if (!command_list) {
        command_list = RHI::get_device()->createCommandList();
    }

    if (!staging_texture || staging_texture->getDesc().width != width ||
        staging_texture->getDesc().height != height) {
        nvrhi::TextureDesc staging_desc;
        staging_desc.debugName = "headless_staging";
        staging_desc.width = width;
        staging_desc.height = height;
        staging_desc.format = texture->getDesc().format;
        staging_desc.initialState = nvrhi::ResourceStates::CopyDest;

        staging_texture = RHI::get_device()->createStagingTexture(
            staging_desc, nvrhi::CpuAccessMode::Read);
    }

    // Single command list operation
    command_list->open();
    command_list->copyTexture(staging_texture, {}, texture, {});
    command_list->close();
    RHI::get_device()->executeCommandList(command_list.Get());
    RHI::get_device()->waitForIdle();

    // Direct memory copy without row-by-row iteration
    size_t pitch;
    auto mapped = RHI::get_device()->mapStagingTexture(
        staging_texture, {}, nvrhi::CpuAccessMode::Read, &pitch);

    size_t row_size = width * 4 * sizeof(float);
    if (pitch == row_size) {
        // Contiguous memory - single memcpy
        memcpy(texture_data.data(), mapped, height * row_size);
    }
    else {
        // Non-contiguous - batch copy rows
        auto src_ptr = static_cast<uint8_t*>(mapped);
        auto dst_ptr = texture_data.data();
        for (int i = 0; i < height; ++i) {
            memcpy(dst_ptr, src_ptr, row_size);
            src_ptr += pitch;
            dst_ptr += row_size;
        }
    }

    RHI::get_device()->unmapStagingTexture(staging_texture);
    spdlog::info("Direct texture copy completed successfully");
    return true;
}

bool ReadTextureCPU(
    UsdImagingGLEngine* renderer,
    HgiUniquePtr& hgi,
    int width,
    int height,
    std::vector<uint8_t>& texture_data)
{
    spdlog::info("Using CPU readback method...");

    auto hgi_texture = renderer->GetAovTexture(HdAovTokens->color);
    if (!hgi_texture) {
        std::cerr << "Error: Failed to get rendered texture" << std::endl;
        return false;
    }

    size_t buffer_size = width * height * 4 * sizeof(float);
    texture_data.resize(buffer_size);

    auto blit_cmds = hgi->CreateBlitCmds();
    HgiTextureGpuToCpuOp copy_op;
    copy_op.gpuSourceTexture = hgi_texture;
    copy_op.cpuDestinationBuffer = texture_data.data();
    copy_op.destinationBufferByteSize = texture_data.size();
    blit_cmds->CopyTextureGpuToCpu(copy_op);

    hgi->SubmitCmds(blit_cmds.get(), HgiSubmitWaitTypeWaitUntilCompleted);
    return true;
}

std::string LoadJSONScript(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error(
            "Could not open JSON script file: " + filename);
    }

    std::string content;
    std::string line;
    while (std::getline(file, line)) {
        content += line + "\n";
    }
    return content;
}

int main(int argc, char* argv[])
{
    // 禁止 abort 弹窗，改为直接退出
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    // 或者设置错误模式，避免 Windows 弹窗
    _set_error_mode(_OUT_TO_STDERR);

    // Parse command line using cmdparser
    cmdline::parser parser;
    parser.add<std::string>("usd", 'u', "USD file to render", true);
    parser.add<std::string>("json", 'j', "JSON rendering script", true);
    parser.add<std::string>(
        "output", 'o', "Output image filename (PNG/HDR/EXR)", true);
    parser.add<int>("width", 'w', "Image width", false, 1920);
    parser.add<int>("height", 'h', "Image height", false, 1080);
    parser.add<int>("spp", 's', "Samples per pixel", false, 16);
    parser.add<std::string>(
        "camera", 'c', "Camera prim path (e.g., /Camera)", false, "");
    parser.add("verbose", 'v', "Enable verbose logging");

    parser.parse_check(argc, argv);

    // Extract settings
    std::string usd_file = parser.get<std::string>("usd");
    std::string json_script = parser.get<std::string>("json");
    std::string output_image = parser.get<std::string>("output");
    int width = parser.get<int>("width");
    int height = parser.get<int>("height");
    int spp = parser.get<int>("spp");
    std::string camera_path = parser.get<std::string>("camera");
    bool verbose = parser.exist("verbose");

    // Validate input files
    if (!std::filesystem::exists(usd_file)) {
        std::cerr << "Error: USD file not found: " << usd_file << std::endl;
        return 1;
    }
    if (!std::filesystem::exists(json_script)) {
        std::cerr << "Error: JSON script not found: " << json_script
                  << std::endl;
        return 1;
    }

    // Initialize logging
    spdlog::set_level(verbose ? spdlog::level::info : spdlog::level::warn);
    spdlog::set_pattern("%^[%T] %n: %v%$");

    spdlog::info("Starting headless render...");
    spdlog::info("USD file: {}", usd_file);
    spdlog::info("JSON script: {}", json_script);
    spdlog::info("Output image: {}", output_image);
    spdlog::info("Resolution: {}x{}", width, height);
    spdlog::info("SPP: {}", spp);

    try {
        // Initialize OpenGL context
        CreateGLContext();
        GarchGLApiLoad();

        // Create USD stage
        auto stage = create_custom_global_stage(usd_file);
        if (!stage) {
            throw std::runtime_error(
                "Failed to load USD stage from " + usd_file);
        }

        // Find camera
        auto camera = GetCamera(stage->get_usd_stage(), camera_path);
        if (!camera) {
            throw std::runtime_error("No camera found in USD file");
        }

        // Setup rendering engine
        auto hgi = Hgi::CreateNamedHgi(HgiTokens->OpenGL);
        HdDriver hd_driver;
        hd_driver.name = HgiTokens->renderDriver;
        hd_driver.driver = VtValue(hgi.get());

        UsdImagingGLEngine::Parameters params;
        params.allowAsynchronousSceneProcessing = false;
        params.driver = hd_driver;

        auto renderer = std::make_unique<UsdImagingGLEngine>(params);
        renderer->SetRendererPlugin(TfToken("Hd_USTC_CG_RendererPlugin"));
        renderer->SetEnablePresentation(false);

        // Configure render settings
        GfVec2i render_size(width, height);
        renderer->SetRenderBufferSize(render_size);
        renderer->SetRenderViewport(GfVec4d(0.0, 0.0, width, height));

        // Setup camera
        auto gf_camera = camera.GetCamera(UsdTimeCode::Default());
        auto frustum = gf_camera.GetFrustum();
        renderer->SetCameraState(
            frustum.ComputeViewMatrix(), frustum.ComputeProjectionMatrix());

        // Configure render parameters
        UsdImagingGLRenderParams render_params;
        render_params.enableLighting = true;
        render_params.enableSceneMaterials = true;
        render_params.showRender = true;
        render_params.frame = UsdTimeCode(0);
        render_params.drawMode = UsdImagingGLDrawMode::DRAW_SHADED_SMOOTH;
        render_params.colorCorrectionMode = HdxColorCorrectionTokens->disabled;
        render_params.clearColor = GfVec4f(0.2f, 0.2f, 0.2f, 1.0f);
        renderer->SetRendererAov(HdAovTokens->color);

        // Load and apply JSON script
        auto node_system = static_cast<const std::shared_ptr<NodeSystem>*>(
            renderer->GetRendererSetting(pxr::TfToken("RenderNodeSystem"))
                .Get<const void*>());

        std::string nodes_json = LoadJSONScript(json_script);
        (*node_system)->get_node_tree()->deserialize(nodes_json);

        // Render the scene with multiple samples
        UsdPrim root = stage->get_usd_stage()->GetPseudoRoot();
        std::cout << "Starting render with " << spp << " samples..."
                  << std::endl;

        // Start timing (will be set after first sample)
        std::chrono::high_resolution_clock::time_point render_start;
        bool timing_started = false;
        long long total_sample_time = 0;
        int timed_samples = 0;

        for (int sample = 0; sample < spp; ++sample) {
            auto sample_start = std::chrono::high_resolution_clock::now();

            renderer->Render(root, render_params);
            renderer->StopRenderer();
            RHI::get_device()->waitForIdle();
            RHI::get_device()->runGarbageCollection();

            auto sample_end = std::chrono::high_resolution_clock::now();
            auto sample_duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    sample_end - sample_start)
                    .count();

            // Start timing after first sample completes
            if (!timing_started && sample == 0) {
                render_start = std::chrono::high_resolution_clock::now();
                timing_started = true;
            }
            else if (sample > 0) {
                total_sample_time += sample_duration;
                timed_samples++;
            }

            if (verbose) {
                std::cout << " (" << sample_duration << "ms)" << std::endl;
            }
        }

        auto render_end = std::chrono::high_resolution_clock::now();
        auto total_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                render_end - render_start)
                .count();

        std::cout << "Render complete. Total time: "
                  << (total_duration / 1000.0) << "s";
        if (spp > 1) {
            std::cout << " (excluding first sample)";
            double avg_sample_time = (double)total_sample_time / timed_samples;
            std::cout << ", Avg per sample: " << (avg_sample_time / 1000.0)
                      << "s";
        }
        std::cout << std::endl;

        // Read back texture data
        std::vector<uint8_t> texture_data;
        bool success =
            ReadTextureDirectly(renderer.get(), width, height, texture_data);

        if (!success) {
            success = ReadTextureCPU(
                renderer.get(), hgi, width, height, texture_data);
        }

        if (!success) {
            throw std::runtime_error("Failed to read back rendered texture");
        }

        // Save the image
        auto save_start = std::chrono::high_resolution_clock::now();
        std::cout << "Saving image to: " << output_image << std::endl;

        if (!SaveImageToFile(output_image, width, height, texture_data)) {
            throw std::runtime_error("Failed to save image to " + output_image);
        }

        auto save_end = std::chrono::high_resolution_clock::now();
        auto save_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                save_end - save_start)
                .count();

        std::cout << "Image saved in " << (save_duration / 1000.0) << "s"
                  << std::endl;
        std::cout << "Headless render completed successfully!" << std::endl;

        // Cleanup
        renderer.reset();
        hgi.reset();
        stage.reset();
        unregister_cpp_type();
#ifdef GPU_GEOM_ALGORITHM
        deinit_gpu_geometry_algorithms();
#endif
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

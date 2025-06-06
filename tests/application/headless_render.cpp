#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include "GCore/GOP.h"
#include "GCore/algorithms/intersection.h"
#include "GCore/geom_payload.hpp"
#include "Logger/Logger.h"
#include "RHI/rhi.hpp"
#include "nodes/system/node_system.hpp"
#include "nvrhi/nvrhi.h"
#include "pxr/usd/usd/primRange.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usdGeom/camera.h"
#include "stage/stage.hpp"
#include "usd_nodejson.hpp"

// Hydra includes
#include "pxr/base/gf/camera.h"
#include "pxr/base/gf/frustum.h"
#include "pxr/imaging/glf/drawTarget.h"
#include "pxr/imaging/hd/driver.h"
#include "pxr/imaging/hd/tokens.h"
#include "pxr/imaging/hdx/tokens.h"
#include "pxr/imaging/hgi/tokens.h"
#include "pxr/usdImaging/usdImagingGL/engine.h"

// For image saving
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "pxr/imaging/hgi/blitCmdsOps.h"
#include "stb_image_write.h"

using namespace USTC_CG;
using namespace pxr;

struct RenderSettings {
    std::string usd_file;
    std::string json_script;
    std::string output_image;
    int width = 1920;
    int height = 1080;
};

bool ParseCommandLine(int argc, char* argv[], RenderSettings& settings)
{
    if (argc < 4) {
        std::cout
            << "Usage: " << argv[0]
            << " <usd_file> <json_script> <output_image> [width] [height]\n";
        std::cout << "  usd_file: Path to USD file to render\n";
        std::cout << "  json_script: Path to JSON rendering script\n";
        std::cout << "  output_image: Output image filename (PNG format)\n";
        std::cout << "  width: Image width (default: 1920)\n";
        std::cout << "  height: Image height (default: 1080)\n";
        return false;
    }

    settings.usd_file = argv[1];
    settings.json_script = argv[2];
    settings.output_image = argv[3];

    if (argc > 4) {
        settings.width = std::atoi(argv[4]);
    }
    if (argc > 5) {
        settings.height = std::atoi(argv[5]);
    }

    // Validate files exist
    if (!std::filesystem::exists(settings.usd_file)) {
        std::cerr << "Error: USD file not found: " << settings.usd_file
                  << std::endl;
        return false;
    }
    if (!std::filesystem::exists(settings.json_script)) {
        std::cerr << "Error: JSON script not found: " << settings.json_script
                  << std::endl;
        return false;
    }

    return true;
}

UsdGeomCamera FindFirstCamera(const UsdStageRefPtr& stage)
{
    // Search for the first camera prim in the stage
    for (const UsdPrim& prim : stage->Traverse()) {
        if (prim.IsA<UsdGeomCamera>()) {
            log::info("Found camera: %s", prim.GetPath().GetString().c_str());
            return UsdGeomCamera(prim);
        }
    }
    return UsdGeomCamera();
}

void CreateGLContext()
{
#ifdef _WIN32
    HDC hdc = GetDC(GetConsoleWindow());
    PIXELFORMATDESCRIPTOR pfd;
    ZeroMemory(&pfd, sizeof(pfd));
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

bool SaveImageToFile(
    const std::string& filename,
    int width,
    int height,
    const std::vector<uint8_t>& data)
{
    // Convert RGBA float to RGBA uint8
    std::vector<uint8_t> rgba_data(width * height * 4);

    const float* float_data = reinterpret_cast<const float*>(data.data());
    for (int i = 0; i < width * height * 4; ++i) {
        rgba_data[i] = static_cast<uint8_t>(
            std::clamp(float_data[i] * 255.0f, 0.0f, 255.0f));
    }

    // Save as PNG
    return stbi_write_png(
               filename.c_str(),
               width,
               height,
               4,
               rgba_data.data(),
               width * 4) != 0;
}

int main(int argc, char* argv[])
{
    RenderSettings settings;
    if (!ParseCommandLine(argc, argv, settings)) {
        return 1;
    }

    log::SetMinSeverity(Severity::Info);
    log::EnableOutputToConsole(true);

    log::info("Starting headless render...");
    log::info("USD file: %s", settings.usd_file.c_str());
    log::info("JSON script: %s", settings.json_script.c_str());
    log::info("Output image: %s", settings.output_image.c_str());
    log::info("Resolution: %dx%d", settings.width, settings.height);

    try {
        // Initialize OpenGL context
        CreateGLContext();
        GarchGLApiLoad();

        // Create USD stage
        auto stage = create_custom_global_stage(settings.usd_file);
        if (!stage) {
            std::cerr << "Error: Failed to load USD stage from "
                      << settings.usd_file << std::endl;
            return 1;
        }

        init(stage.get());

        // Find camera
        auto camera = FindFirstCamera(stage->get_usd_stage());
        if (!camera) {
            std::cerr << "Error: No camera found in USD file" << std::endl;
            return 1;
        }

        // Setup rendering engine
        UsdImagingGLEngine::Parameters params;
        params.allowAsynchronousSceneProcessing = false;

        // Initialize with OpenGL
        auto hgi = Hgi::CreateNamedHgi(HgiTokens->OpenGL);
        HdDriver hdDriver;
        hdDriver.name = HgiTokens->renderDriver;
        hdDriver.driver = VtValue(hgi.get());
        params.driver = hdDriver;

        auto renderer = std::make_unique<UsdImagingGLEngine>(params);
        renderer->SetRendererPlugin(TfToken("Hd_USTC_CG_RendererPlugin"));
        renderer->SetEnablePresentation(false);

        // Set render buffer size
        GfVec2i renderSize(settings.width, settings.height);
        renderer->SetRenderBufferSize(renderSize);
        renderer->SetRenderViewport(
            GfVec4d(0.0, 0.0, double(settings.width), double(settings.height)));

        // Setup camera
        auto gf_camera = camera.GetCamera(UsdTimeCode::Default());
        auto frustum = gf_camera.GetFrustum();

        GfMatrix4d projectionMatrix = frustum.ComputeProjectionMatrix();
        GfMatrix4d viewMatrix = frustum.ComputeViewMatrix();
        renderer->SetCameraState(viewMatrix, projectionMatrix);

        // Setup render parameters
        UsdImagingGLRenderParams renderParams;
        renderParams.enableLighting = true;
        renderParams.enableSceneMaterials = true;
        renderParams.showRender = true;
        renderParams.frame = UsdTimeCode(0);
        renderParams.drawMode = UsdImagingGLDrawMode::DRAW_SHADED_SMOOTH;
        renderParams.colorCorrectionMode = HdxColorCorrectionTokens->disabled;
        renderParams.clearColor = GfVec4f(0.2f, 0.2f, 0.2f, 1.0f);        // Set AOV to color
        renderer->SetRendererAov(HdAovTokens->color);

        // Load and apply JSON script settings if needed
        // This would depend on your specific JSON format for render settings

        // Render the scene
        UsdPrim root = stage->get_usd_stage()->GetPseudoRoot();
        log::info("Starting render...");
        renderer->Render(root, renderParams);
        log::info("Render complete.");

        // Get the rendered image
        auto hgi_texture = renderer->GetAovTexture(HdAovTokens->color);
        if (!hgi_texture) {
            std::cerr << "Error: Failed to get rendered texture" << std::endl;
            return 1;
        }

        // Read back the texture data
        size_t buffer_size =
            settings.width * settings.height * 4 * sizeof(float);
        std::vector<uint8_t> texture_data(buffer_size);

        auto blitCmds = hgi->CreateBlitCmds();
        HgiTextureGpuToCpuOp copyOp;
        copyOp.gpuSourceTexture = hgi_texture;
        copyOp.cpuDestinationBuffer = texture_data.data();
        copyOp.destinationBufferByteSize = texture_data.size();
        blitCmds->CopyTextureGpuToCpu(copyOp);

        hgi->SubmitCmds(blitCmds.get(), HgiSubmitWaitTypeWaitUntilCompleted);

        // Save the image
        log::info("Saving image to: %s", settings.output_image.c_str());
        if (!SaveImageToFile(
                settings.output_image,
                settings.width,
                settings.height,
                texture_data)) {
            std::cerr << "Error: Failed to save image to "
                      << settings.output_image << std::endl;
            return 1;
        }

        log::info("Headless render completed successfully!");

        // Cleanup
        renderer.reset();
        hgi.reset();
        stage.reset();

        unregister_cpp_type();
        deinit_gpu_geometry_algorithms();
    }
    catch (const std::exception& e) {
        std::cerr << "Error during rendering: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

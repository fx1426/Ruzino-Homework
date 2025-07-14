#ifndef IMGUI_DEFINE_MATH_OPERATORS
#define IMGUI_DEFINE_MATH_OPERATORS
#endif
#include "widgets/usdview/usdview_widget.hpp"

#include <pxr/imaging/hd/driver.h>

#include "GCore/geom_payload.hpp"
#include "Logger/Logger.h"
#include "RHI/Hgi/desc_conversion.hpp"
#include "RHI/rhi.hpp"
#include "free_camera.hpp"
#include "imgui.h"
#include "nvrhi/nvrhi.h"
#include "pxr/base/gf/camera.h"
#include "pxr/base/gf/frustum.h"
#include "pxr/imaging/glf/drawTarget.h"
#include "pxr/imaging/hdx/tokens.h"
#include "pxr/imaging/hgi/blitCmds.h"
#include "pxr/imaging/hgi/blitCmdsOps.h"
#include "pxr/imaging/hgi/tokens.h"
#include "pxr/pxr.h"
#include "pxr/usd/usd/primRange.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usdGeom/camera.h"
#include "pxr/usdImaging/usdImagingGL/engine.h"

USTC_CG_NAMESPACE_OPEN_SCOPE
class NodeTree;

struct UsdviewEnginePrivateData {
    nvrhi::TextureHandle nvrhi_texture = nullptr;
    nvrhi::StagingTextureHandle staging = nullptr;
    nvrhi::Format present_format = nvrhi::Format::RGBA32_FLOAT;
};

UsdviewEngine::UsdviewEngine(Stage* stage) : stage_(stage)
{
    data_ = std::make_unique<UsdviewEnginePrivateData>();
    // Initialize OpenGL context using WGL
    CreateGLContext();
    GarchGLApiLoad();
    pxr::UsdImagingGLEngine::Parameters params;
    params.allowAsynchronousSceneProcessing = true;

    hgi = pxr::Hgi::CreateNamedHgi(pxr::HgiTokens->OpenGL);
    pxr::HdDriver hdDriver;
    hdDriver.name = pxr::HgiTokens->renderDriver;
    hdDriver.driver = pxr::VtValue(hgi.get());
    params.driver = hdDriver;

    renderer_ = std::make_unique<pxr::UsdImagingGLEngine>(params);

    renderer_->SetEnablePresentation(false);
    free_camera_ = std::make_unique<FirstPersonCamera>();

    auto prim = pxr::UsdGeomCamera::Get(
        stage_->get_usd_stage(), pxr::SdfPath("/FreeCamera"));
    if (prim) {
        *free_camera_ = prim;
    }
    else {
        static_cast<pxr::UsdGeomCamera&>(*free_camera_) =
            pxr::UsdGeomCamera::Define(
                stage_->get_usd_stage(), pxr::SdfPath("/FreeCamera"));

        free_camera_->CreateFocusDistanceAttr().Set(10.0f);
        free_camera_->CreateClippingRangeAttr(
            pxr::VtValue(pxr::GfVec2f{ 1.f, 2000.f }));

        static_cast<FirstPersonCamera*>(free_camera_.get())
            ->LookAt(
                pxr::GfVec3d{ -10, 0, 0 },
                pxr::GfVec3d{ 0, 0, 0 },
                pxr::GfVec3d{ 0, 0, 1 });
    }
    auto plugins = renderer_->GetRendererPlugins();

    ChooseRenderer(plugins, engine_status.renderer_id);
}

void UsdviewEngine::ChooseRenderer(
    const pxr::TfTokenVector& available_renderers,
    unsigned i)
{
    renderer_->SetRendererPlugin(available_renderers[i]);
    log::info(
        "Switching to renderer %s", available_renderers[i].GetString().c_str());
    if (available_renderers[i].GetString() == "Hd_USTC_CG_RendererPlugin") {
        renderer_ui_control =
            renderer_->GetRendererSetting(pxr::TfToken("RenderNodeSystem"))
                .Get<const void*>();
    }

    if (available_renderers[i].GetString() == "Hd_USTC_CG_GL_RendererPlugin") {
        renderer_ui_control =
            renderer_->GetRendererSetting(pxr::TfToken("RenderNodeSystem"))
                .Get<const void*>();
    }

    renderer_->SetEnablePresentation(false);
    data_->nvrhi_texture = nullptr;

    this->engine_status.renderer_id = i;
}

void UsdviewEngine::DrawMenuBar()
{
    ImGui::BeginMenuBar();
    if (ImGui::BeginMenu("Free Camera")) {
        if (ImGui::BeginMenu("Camera Type")) {
            if (ImGui::MenuItem(
                    "First Personal",
                    0,
                    this->engine_status.cam_type == CamType::First)) {
                if (engine_status.cam_type != CamType::First) {
                    free_camera_ = std::make_unique<FirstPersonCamera>();
                    engine_status.cam_type = CamType::First;
                }
            }
            if (ImGui::MenuItem(
                    "Third Personal",
                    0,
                    this->engine_status.cam_type == CamType::Third)) {
                if (engine_status.cam_type != CamType::Third) {
                    free_camera_ = std::make_unique<ThirdPersonCamera>();
                    engine_status.cam_type = CamType::Third;
                }
            }
            ImGui::EndMenu();
        }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Renderer")) {
        if (ImGui::BeginMenu("Select Renderer")) {
            auto available_renderers = renderer_->GetRendererPlugins();
            for (unsigned i = 0; i < available_renderers.size(); ++i) {
                if (ImGui::MenuItem(
                        available_renderers[i].GetText(),
                        0,
                        this->engine_status.renderer_id == i)) {
                    if (this->engine_status.renderer_id != i) {
                        ChooseRenderer(available_renderers, i);
                        renderer_->SetRenderBufferSize(render_buffer_size_);
                        renderer_->SetRenderViewport(
                            pxr::GfVec4d{ 0.0,
                                          0.0,
                                          double(render_buffer_size_[0]),
                                          double(render_buffer_size_[1]) });
                    }
                }
            }
            ImGui::EndMenu();
        }
        ImGui::EndMenu();
    }

    ImGui::EndMenuBar();
}

void UsdviewEngine::copy_to_presentation()
{
    // Since Hgi and nvrhi vulkan are on different Vulkan instances and we
    // don't
    // want to modify Hgi's external information definition, we need to do a
    // CPU read back to send the information to nvrhi.

    auto hgi_texture = renderer_->GetAovTexture(pxr::HdAovTokens->color);
    if (hgi_texture) {
        nvrhi::TextureDesc tex_desc =
            RHI::ConvertToNvrhiTextureDesc(hgi_texture->GetDescriptor());
        tex_desc.keepInitialState = true;
        tex_desc.initialState = nvrhi::ResourceStates::CopyDest;

        pxr::HgiBlitCmdsUniquePtr blitCmds = hgi->CreateBlitCmds();
        pxr::HgiTextureGpuToCpuOp copyOp;
        copyOp.gpuSourceTexture = hgi_texture;
        copyOp.cpuDestinationBuffer = texture_data_.data();
        copyOp.destinationBufferByteSize = texture_data_.size();
        blitCmds->CopyTextureGpuToCpu(copyOp);

        hgi->SubmitCmds(
            blitCmds.get(), pxr::HgiSubmitWaitTypeWaitUntilCompleted);
        if (!data_->nvrhi_texture) {
            std::tie(data_->nvrhi_texture, data_->staging) =
                RHI::load_texture(tex_desc, texture_data_.data());
        }
        else {
            RHI::write_texture(
                data_->nvrhi_texture.Get(),
                data_->staging.Get(),
                texture_data_.data());
        }
    }
}

void UsdviewEngine::OnFrame(float delta_time)
{
    if (first_draw) {
        first_draw = false;
        return;
    }
    DrawMenuBar();

    auto previous = data_->nvrhi_texture.Get();

    using namespace pxr;
    GfFrustum frustum =
        free_camera_->GetCamera(UsdTimeCode::Default()).GetFrustum();

    double fov, aspect_ratio, near_distance, far_distance;

    frustum.GetPerspective(&fov, &aspect_ratio, &near_distance, &far_distance);

    frustum.SetPerspective(
        fov,
        float(render_buffer_size_[0]) / float(render_buffer_size_[1]),
        near_distance,
        far_distance);

    GfMatrix4d projectionMatrix = frustum.ComputeProjectionMatrix();
    GfMatrix4d viewMatrix = frustum.ComputeViewMatrix();

    renderer_->SetCameraState(viewMatrix, projectionMatrix);

    _renderParams.enableLighting = true;
    _renderParams.enableSceneMaterials = true;
    _renderParams.showRender = true;
    if (timecode == 0)
        _renderParams.frame = UsdTimeCode(0);
    else {
        _renderParams.frame = std::min(
            UsdTimeCode(stage_->get_current_time()),
            UsdTimeCode(timecode - 1.0f / 30.f));
    }
    _renderParams.drawMode = UsdImagingGLDrawMode::DRAW_WIREFRAME_ON_SURFACE;
    _renderParams.colorCorrectionMode = pxr::HdxColorCorrectionTokens->disabled;

    _renderParams.clearColor = GfVec4f(1.f, 1.f, 1.f, 1.f);
    _renderParams.clearColor = GfVec4f(0.2f, 0.2f, 0.2f, 1.f);

    for (int i = 0; i < free_camera_->GetCamera(UsdTimeCode::Default())
                            .GetClippingPlanes()
                            .size();
         ++i) {
        _renderParams.clipPlanes[i] =
            free_camera_->GetCamera(UsdTimeCode::Default())
                .GetClippingPlanes()[i];
    }

    GlfSimpleLightVector lights(1);
    auto cam_pos = frustum.GetPosition();
    lights[0].SetPosition(GfVec4f{
        float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]), 1.0f });
    lights[0].SetAmbient(GfVec4f(0.8, 0.8, 0.8, 1));
    lights[0].SetDiffuse(GfVec4f(1.0f));
    lights[0].SetSpecular(GfVec4f(1.0f));
    GlfSimpleMaterial material;
    float kA = 6.8f;
    float kS = 0.4f;
    float shiness = 0.8f;
    material.SetDiffuse(GfVec4f(kA, kA, kA, 1.0f));
    material.SetSpecular(GfVec4f(kS, kS, kS, 1.0f));
    material.SetShininess(shiness);
    GfVec4f sceneAmbient = { 0.01, 0.01, 0.01, 1.0 };
    renderer_->SetLightingState(lights, material, sceneAmbient);
    renderer_->SetRendererAov(HdAovTokens->color);

    for (auto&& setting : settings) {
        renderer_->SetRendererSetting(setting.first, setting.second);
    }

    UsdPrim root = stage_->get_usd_stage()->GetPseudoRoot();

    // First try is there a hack?
    renderer_->Render(root, _renderParams);

    auto imgui_frame_size =
        ImVec2(render_buffer_size_[0], render_buffer_size_[1]);

    ImGui::BeginChild("ViewPort", imgui_frame_size, 0, ImGuiWindowFlags_NoMove);

    ImGui::GetIO().WantCaptureMouse = false;
    if (data_->nvrhi_texture.Get()) {
        persistent_texture = data_->nvrhi_texture;
        ImGui::Image(
            static_cast<ImTextureID>(persistent_texture.Get()),
            imgui_frame_size,
            ImVec2(0.0f, 1.0f),
            ImVec2(1.0f, 0.0f));
    }
    else {
        log ::warning("No image!");
    }
    is_active = ImGui::IsWindowFocused();
    is_hovered = ImGui::IsItemHovered();

    if (right_mouse_pressed) {
        auto mouse_pos_rel = ImGui::GetMousePos() - ImGui::GetItemRectMin();

        log::info(
            "Mouse Position Relative: (%.2f, %.2f)",
            mouse_pos_rel.x,
            mouse_pos_rel.y);
        // Normalize the mouse position to be in the range [0, 1]
        ImVec2 mousePosNorm = ImVec2(
            mouse_pos_rel.x / render_buffer_size_[0],
            mouse_pos_rel.y / render_buffer_size_[1]);

        // Convert to NDC coordinates
        ImVec2 mousePosNDC =
            ImVec2(mousePosNorm.x * 2.0f - 1.0f, 1.0f - mousePosNorm.y * 2.0f);

        log::info(
            "Mouse Position NDC: (%.2f, %.2f)", mousePosNDC.x, mousePosNDC.y);

        using namespace pxr;
        UsdPrim root = stage_->get_usd_stage()->GetPseudoRoot();

        GfVec3d point;
        GfVec3d normal;
        SdfPath path;
        SdfPath instancer;
        HdInstancerContext outInstancerContext;
        int outHitInstanceIndex;

        auto narrowed = frustum.ComputeNarrowedFrustum(
            { mousePosNDC[0], mousePosNDC[1] },
            { 1.0 / render_buffer_size_[0], 1.0 / render_buffer_size_[1] });
        if (renderer_->TestIntersection(
                narrowed.ComputeViewMatrix(),
                narrowed.ComputeProjectionMatrix(),
                root,
                _renderParams,
                &point,
                &normal,
                &path,
                &instancer,
                &outHitInstanceIndex,
                &outInstancerContext)) {
            // Create and store the pick event
            current_pick_event_ =
                std::make_shared<PickEvent>(point, normal, path, instancer);

            log::info("Picked prim " + path.GetAsString());
            log::info(
                "Picked point: (%.2f, %.2f, %.2f)",
                point[0],
                point[1],
                point[2]);
        }
    }

    ImGui::GetIO().WantCaptureMouse = true;

    ImGui::EndChild();
    time_controller();
}

void UsdviewEngine::time_controller()
{
    timecode = stage_->get_render_time().GetValue();

    if (is_active && ImGui::IsKeyReleased(ImGuiKey_Space)) {
        playing = !playing;
    }
    if (playing) {
        timecode += 1.0f / 30.f;

        if (timecode > time_code_max) {
            timecode = 0;
        }
        stage_->set_render_time(timecode);
    }

    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    if (ImGui::SliderFloat("##timecode", &timecode, 0, time_code_max)) {
        stage_->set_render_time(timecode);
    }

    // TODO:  1.加个显示当前仿真进度的进度条 (current_time)
}

// std::unique_ptr<USTC_CG::PickEvent> UsdviewEngine::get_pick_event()
//{
//     return std::move(pick_event);
// }
//
// bool UsdviewEngine::CameraCallback(float delta_time)
//{
//    ImGuiIO& io = ImGui::GetIO();
//    if (is_active_) {
//        free_camera_->KeyboardUpdate();
//    }
//
//    if (is_hovered_) {
//        for (int i = 0; i < 5; ++i) {
//            if (io.MouseClicked[i]) {
//                free_camera_->MouseButtonUpdate(i);
//            }
//        }
//        float fovAdjustment = io.MouseWheel * 5.0f;
//        if (fovAdjustment != 0) {
//            free_camera_->MouseScrollUpdate(fovAdjustment);
//        }
//    }
//    for (int i = 0; i < 5; ++i) {
//        if (io.MouseReleased[i]) {
//            free_camera_->MouseButtonUpdate(i);
//        }
//    }
//    free_camera_->MousePosUpdate(io.MousePos.x, io.MousePos.y);
//
//    free_camera_->Animate(delta_time);
//
//    return false;
//}

bool UsdviewEngine::JoystickButtonUpdate(int button, bool pressed)
{
    free_camera_->JoystickButtonUpdate(button, pressed);
    return false;
}

bool UsdviewEngine::JoystickAxisUpdate(int axis, float value)
{
    free_camera_->JoystickUpdate(axis, value);
    return false;
}

bool UsdviewEngine::KeyboardUpdate(int key, int scancode, int action, int mods)
{
    if (is_active) {
        free_camera_->KeyboardUpdate(key, scancode, action, mods);
    }
    return false;
}

bool UsdviewEngine::MousePosUpdate(double xpos, double ypos)
{
    free_camera_->MousePosUpdate(xpos, ypos);
    return false;
}

bool UsdviewEngine::MouseScrollUpdate(double xoffset, double yoffset)
{
    if (is_active && is_hovered) {
        free_camera_->MouseScrollUpdate(xoffset, yoffset);
    }
    return false;
}

bool UsdviewEngine::MouseButtonUpdate(int button, int action, int mods)
{
    if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_LEFT) {
        if (is_hovered) {
            free_camera_->MouseButtonUpdate(button, action, mods);
        }
        return false;
    }
    if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (is_hovered) {
            right_mouse_pressed = true;
            return false;
        }
    }
    if (action == GLFW_RELEASE && button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (right_mouse_pressed) {
            right_mouse_pressed = false;
            return false;
        }
    }

    free_camera_->MouseButtonUpdate(button, action, mods);

    return false;
}

void UsdviewEngine::Animate(float elapsed_time_seconds)
{
    free_camera_->Animate(elapsed_time_seconds);
    IWidget::Animate(elapsed_time_seconds);
}

void UsdviewEngine::CreateGLContext()
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

UsdviewEngine::~UsdviewEngine()
{
    data_.reset();
    assert(RHI::get_device());
    renderer_.reset();
    hgi.reset();
}

bool UsdviewEngine::BuildUI()
{
    auto delta_time = ImGui::GetIO().DeltaTime;

    if (size_changed) {
        auto size = ImGui::GetContentRegionAvail();
        if (size.y > 26)
            size.y -= 26;
        RenderBackBufferResized(size.x, size.y);
    }

    if (render_buffer_size_[0] > 0 && render_buffer_size_[1] > 0) {
        OnFrame(delta_time);
    }

    return true;
}

void UsdviewEngine::SetEditMode(bool editing)
{
    is_editing_ = editing;
}

const void* UsdviewEngine::emit_create_renderer_ui_control()
{
    auto temp = renderer_ui_control;
    renderer_ui_control = nullptr;
    return temp;
}

pxr::VtValue UsdviewEngine::get_renderer_setting(const pxr::TfToken& id) const
{
    return renderer_->GetRendererSetting(id);
}

void UsdviewEngine::set_renderer_setting(
    const pxr::TfToken& id,
    const pxr::VtValue& value)
{
    settings[id] = value;
    renderer_->SetRendererSetting(id, value);
}

void UsdviewEngine::finish_render()
{
    renderer_->StopRenderer();
    auto hacked_handle =
        renderer_->GetRendererSetting(pxr::TfToken("VulkanColorAov"));

    if (hacked_handle.IsHolding<const void*>()) {
        auto rendered = *reinterpret_cast<const nvrhi::TextureHandle*>(
            hacked_handle.Get<const void*>());
        if (rendered) {
            RHI::copy_from_texture(data_->nvrhi_texture, rendered);
        }
    }
    else {
        copy_to_presentation();
    }
}

ImGuiWindowFlags UsdviewEngine::GetWindowFlag()
{
    return ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoCollapse |
           ImGuiWindowFlags_NoScrollbar;
}

const char* UsdviewEngine::GetWindowName()
{
    return "UsdView Engine";
}

std::string UsdviewEngine::GetWindowUniqueName()
{
    return "Hydra Renderer";
}

void UsdviewEngine::RenderBackBufferResized(float x, float y)
{
    render_buffer_size_[0] = x;
    render_buffer_size_[1] = y;

    renderer_->SetRenderBufferSize(render_buffer_size_);
    renderer_->SetRenderViewport(
        pxr::GfVec4d{ 0.0,
                      0.0,
                      double(render_buffer_size_[0]),
                      double(render_buffer_size_[1]) });

    data_->nvrhi_texture = nullptr;
    data_->staging = nullptr;
    texture_data_.resize(
        render_buffer_size_[0] * render_buffer_size_[1] *
        RHI::calculate_bytes_per_pixel(data_->present_format));
}

std::shared_ptr<PickEvent> UsdviewEngine::consume_pick_event()
{
    auto event = current_pick_event_;
    current_pick_event_ = nullptr;
    return event;
}

USTC_CG_NAMESPACE_CLOSE_SCOPE

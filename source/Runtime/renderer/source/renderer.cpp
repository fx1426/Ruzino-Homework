#include "renderer.h"

#include "camera.h"
#include "material/material.h"
#include "node_exec_eager_render.hpp"
#include "nodes/system/node_system.hpp"
#include "pxr/imaging/hd/renderBuffer.h"
#include "pxr/imaging/hd/tokens.h"
#include "renderBuffer.h"
#include "renderParam.h"

USTC_CG_NAMESPACE_OPEN_SCOPE
using namespace pxr;

Hd_USTC_CG_Renderer::Hd_USTC_CG_Renderer(Hd_USTC_CG_RenderParam* render_param)
    : _enableSceneColors(false),
      render_param(render_param)
{
}

Hd_USTC_CG_Renderer::~Hd_USTC_CG_Renderer()
{
    auto executor = dynamic_cast<EagerNodeTreeExecutorRender*>(
        render_param->node_system->get_node_tree_executor());
}

static TextureHandle create_empty_texture(
    const pxr::GfVec2i& size,
    nvrhi::Format format = nvrhi::Format::RGBA32_FLOAT)
{
    nvrhi::TextureDesc desc =
        nvrhi::TextureDesc{}
            .setWidth(size[0])
            .setHeight(size[1])
            .setFormat(format)
            .setInitialState(nvrhi::ResourceStates::ShaderResource)
            .setKeepInitialState(true)
            .setIsUAV(true);
    auto d = RHI::get_device();
    auto texture = d->createTexture(desc);

    auto commandList = d->createCommandList();
    commandList->open();
    commandList->clearTextureFloat(
        texture, nvrhi::AllSubresources, nvrhi::Color(0.0f, 0.0f, 0.0f, 0.0f));
    commandList->close();
    d->executeCommandList(commandList);
    d->waitForIdle();

    return texture;
}

void Hd_USTC_CG_Renderer::Render(HdRenderThread* renderThread)
{
    _completedSamples.store(0);

    render_param->default_texture_name.clear();
    render_param->presented_textures.clear();

    for (auto& material_thread : render_param->material_loading_threads) {
        material_thread.join();
    }
    render_param->material_loading_threads.clear();

    for (auto& texture_thread : render_param->texture_loading_threads) {
        texture_thread.join();
    }
    render_param->texture_loading_threads.clear();
    auto node_system = render_param->node_system;

    {
        auto& global_payload = node_system->get_node_tree_executor()
                                   ->get_global_payload<RenderGlobalPayload&>();

        // Clear all dirty flags from previous frame at the beginning of each
        // frame
        global_payload.clear_dirty(
            RenderGlobalPayload::SceneDirtyBits::DirtyMaterials);
        global_payload.clear_dirty(
            RenderGlobalPayload::SceneDirtyBits::DirtyGeometry);
        global_payload.clear_dirty(
            RenderGlobalPayload::SceneDirtyBits::DirtyTLAS);
        global_payload.clear_dirty(
            RenderGlobalPayload::SceneDirtyBits::DirtyTransforms);
        global_payload.clear_dirty(
            RenderGlobalPayload::SceneDirtyBits::DirtyLights);
        global_payload.clear_dirty(
            RenderGlobalPayload::SceneDirtyBits::DirtyCamera);

        global_payload.InstanceCollection =
            render_param->InstanceCollection.get();

        // Track material shader compilation and changes
        bool materials_changed = false;
        static std::unordered_map<pxr::SdfPath, uint32_t, pxr::TfHash>
            material_generations;

        {
            std::vector<std::future<void>> futures;
            std::mutex change_mutex;

            for (auto& material : *render_param->material_map) {
                if (!material.second) {
                    continue;
                }

                // Record generation before compilation
                uint32_t old_gen = material_generations[material.first];
                auto material_path = material.first;

                futures.push_back(
                    std::async(
                        std::launch::async, [&, material_path, old_gen]() {
                            auto mat =
                                (*render_param->material_map)[material_path];
                            if (!mat)
                                return;

                            mat->ensure_shader_ready(
                                global_payload.shader_factory);

                            // Check if shader generation changed
                            uint32_t new_gen = mat->get_shader_generation();
                            if (old_gen != new_gen) {
                                std::lock_guard<std::mutex> lock(change_mutex);
                                material_generations[material_path] = new_gen;
                                materials_changed = true;
                            }
                        }));
            }

            // Wait for all shader compilations to complete
            for (auto& future : futures) {
                future.wait();
            }

            if (materials_changed) {
                global_payload.InstanceCollection->mark_materials_dirty();
            }
        }

        // Mark dirty flags based on changes
        if (materials_changed) {
            global_payload.mark_dirty(
                RenderGlobalPayload::SceneDirtyBits::DirtyMaterials);
        }

        // Check for geometry/buffer changes
        static uint32_t last_geometry_version = 0;
        uint32_t current_geometry_version =
            global_payload.InstanceCollection->get_geometry_version();
        if (last_geometry_version != current_geometry_version) {
            global_payload.mark_dirty(
                RenderGlobalPayload::SceneDirtyBits::DirtyGeometry);
            last_geometry_version = current_geometry_version;
        }

        if (global_payload.InstanceCollection->get_require_rebuild_tlas()) {
            global_payload.mark_dirty(
                RenderGlobalPayload::SceneDirtyBits::DirtyTLAS);
            global_payload.mark_dirty(
                RenderGlobalPayload::SceneDirtyBits::DirtyGeometry);
        }

        global_payload.resource_allocator.gc();

        global_payload.lens_system = render_param->lens_system;

        global_payload.reset_accumulation = false;

        node_system->execute(false);

        // Clear dirty flags after execution
        // Note: nodes should clear specific flags as they handle them
    }

    for (size_t i = 0; i < _aovBindings.size(); ++i) {
        std::string present_name;

        if (_aovBindings[i].aovName == HdAovTokens->depth) {
            present_name = "present_depth";
        }

        if (_aovBindings[i].aovName == HdAovTokens->color) {
            present_name = "present_color";
        }

        // Find ALL present nodes of this type and store each one with data
        for (auto&& node : node_system->get_node_tree()->nodes) {
            if (std::string(node->typeinfo->id_name) != present_name) {
                continue;  // Skip non-matching nodes
            }

            // Try to fetch texture from this node
            assert(node->get_inputs().size() == 1);
            auto output_socket = node->get_inputs()[0];
            entt::meta_any data;
            node_system->get_node_tree_executor()
                ->sync_node_to_external_storage(output_socket, data);

            if (!data) {
                continue;  // Skip nodes with no data
            }

            nvrhi::TextureHandle texture = data.cast<nvrhi::TextureHandle>();
            if (!texture) {
                continue;  // Skip invalid textures
            }

            // Store texture with node's UI name
            std::string texture_name =
                node->ui_name.empty() ? present_name : node->ui_name;
            render_param->presented_textures[texture_name] = texture;

            // Keep backward compatibility: first texture becomes default
            if (render_param->default_texture_name.empty()) {
                render_param->default_texture_name = texture_name;
            }

            // Update render buffer
            auto rb = static_cast<Hd_USTC_CG_RenderBuffer*>(
                _aovBindings[i].renderBuffer);
#ifdef USTC_CG_DIRECT_VK_DISPLAY
            // Already stored above
#else
            rb->Present(texture);
#endif
            rb->SetConverged(true);
        }

        // Create empty texture if nothing was presented
        if (render_param->default_texture_name.empty()) {
            auto empty_tex = create_empty_texture(
                GfVec2i{ 16, 16 }, nvrhi::Format::RGBA32_FLOAT);
            render_param->presented_textures["_empty"] = empty_tex;
            render_param->default_texture_name = "_empty";
        }
    }

    node_system->finalize();

    // executor->finalize(node_tree);
}

void Hd_USTC_CG_Renderer::Clear()
{
    for (size_t i = 0; i < _aovBindings.size(); ++i) {
        if (_aovBindings[i].clearValue.IsEmpty()) {
            continue;
        }

        auto rb =
            static_cast<Hd_USTC_CG_RenderBuffer*>(_aovBindings[i].renderBuffer);
        rb->Clear();
    }
}

void Hd_USTC_CG_Renderer::SetAovBindings(
    const HdRenderPassAovBindingVector& aovBindings)
{
    _aovBindings = aovBindings;
    _aovNames.resize(_aovBindings.size());
    for (size_t i = 0; i < _aovBindings.size(); ++i) {
        _aovNames[i] = HdParsedAovToken(_aovBindings[i].aovName);
    }

    // Re-validate the attachments.
    _aovBindingsNeedValidation = true;
}

void Hd_USTC_CG_Renderer::MarkAovBuffersUnconverged()
{
    for (size_t i = 0; i < _aovBindings.size(); ++i) {
        auto rb =
            static_cast<Hd_USTC_CG_RenderBuffer*>(_aovBindings[i].renderBuffer);
        rb->SetConverged(false);
    }
}

void Hd_USTC_CG_Renderer::renderTimeUpdateCamera(
    const HdRenderPassStateSharedPtr& renderPassState)
{
    camera_ =
        static_cast<const Hd_USTC_CG_Camera*>(renderPassState->GetCamera());
    camera_->update(renderPassState);
}

bool Hd_USTC_CG_Renderer::nodetree_modified()
{
    //    return render_param->node_tree->GetDirty();
    return false;
}

bool Hd_USTC_CG_Renderer::nodetree_modified(bool new_status)
{
    // auto old_status = render_param->node_tree->GetDirty();
    // render_param->node_tree->SetDirty(new_status);
    // return old_status;

    return false;
}

USTC_CG_NAMESPACE_CLOSE_SCOPE

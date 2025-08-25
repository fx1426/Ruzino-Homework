#include <spdlog/spdlog.h>

#include "GCore/GOP.h"
#include "GCore/algorithms/intersection.h"
#include "GCore/geom_payload.hpp"
#include "GUI/window.h"
#include "cmdparser.hpp"
#include "nodes/system/node_system.hpp"
#include "nodes/ui/imgui.hpp"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usdGeom/sphere.h"
#include "stage/stage.hpp"
#include "usd_nodejson.hpp"
#include "widgets/usdtree/usd_fileviewer.h"
#include "widgets/usdview/usdview_widget.hpp"
using namespace USTC_CG;

int main(int argc, char* argv[])
{
#ifdef _DEBUG
    spdlog::set_level(spdlog::level::debug);
#endif
    spdlog::set_pattern("%^[%T] %n: %v%$");
    auto window = std::make_unique<Window>();

    // Check for command line arguments to specify USD file
    std::unique_ptr<Stage> stage;
    if (argc > 1) {
        // Use custom stage path from command line
        std::string stage_path = argv[1];
        stage = create_custom_global_stage(stage_path);
    }
    else {
        // Use default stage
        stage = create_global_stage();
    }

#ifdef REAL_TIME
    window->register_function_before_frame(
        [&stage](Window* window) { stage->tick(window->get_elapsed_time()); });
#else

    window->register_function_before_frame(
        [&stage](Window* window) { stage->tick(1.0f / 30.f); });
#endif
    // Add a sphere

    auto usd_file_viewer = std::make_unique<UsdFileViewer>(stage.get());
    auto render = std::make_unique<UsdviewEngine>(stage.get());

    auto render_bare = render.get();

    render->SetCallBack([](Window* window, IWidget* render_widget) {
        auto node_system = static_cast<const std::shared_ptr<NodeSystem>*>(
            dynamic_cast<UsdviewEngine*>(render_widget)
                ->emit_create_renderer_ui_control());
        if (node_system) {
            FileBasedNodeWidgetSettings desc;
            desc.system = *node_system;
            desc.json_path = "../../Assets/render_nodes_save.json";

            std::unique_ptr<IWidget> node_widget =
                std::move(create_node_imgui_widget(desc));

            window->register_widget(std::move(node_widget));
        }
    });

    window->register_widget(std::move(render));
    window->register_widget(std::move(usd_file_viewer));

    window->register_function_after_frame([&stage,
                                           render_bare](Window* window) {
        pxr::SdfPath json_path;
        if (stage->consume_editor_creation(json_path)) {
            auto system = create_dynamic_loading_system();
            /* Load the node system */
            auto loaded = system->load_configuration("geometry_nodes.json");
            loaded = system->load_configuration("basic_nodes.json");

            // iterate over path Plugin (not recursively), get all the json and
            // load them

            auto plugin_path = std::filesystem::path("./Plugins");

            if (std::filesystem::exists(plugin_path))
                for (auto& p :
                     std::filesystem::directory_iterator(plugin_path)) {
                    if (p.path().extension() == ".json") {
                        system->load_configuration(p.path().string());
                    }
                }

            system->init();
            system->set_node_tree_executor(create_node_tree_executor({}));
            /* Done! */
            UsdBasedNodeWidgetSettings desc;

            desc.json_path = json_path;
            desc.system = system;
            desc.stage = stage.get();

            std::unique_ptr<IWidget> node_widget =
                std::move(create_node_imgui_widget(desc));
            node_widget->SetCallBack(
                [&stage, json_path, system, render_bare](Window*, IWidget*) {
                    GeomPayload geom_global_params;
#ifdef GEOM_USD_EXTENSION
                    geom_global_params.stage = stage->get_usd_stage();
                    geom_global_params.prim_path = json_path;
#endif

                    geom_global_params.has_simulation = false;

                    // Pass pick event from UI to geometry payload
                    geom_global_params.pick = render_bare->consume_pick_event();

                    system->set_global_params(geom_global_params);
                    if (geom_global_params.pick) {
                        system->execute();
                    }
                });

            window->register_widget(std::move(node_widget));
        }
    });

    window->register_function_after_frame(
        [render_bare](Window* window) { render_bare->finish_render(); });

    window->register_function_after_frame(
        [&stage](Window* window) { stage->finish_tick(); });

    window->run();

    unregister_cpp_type();

#ifdef GPU_GEOM_ALGORITHM
    deinit_gpu_geometry_algorithms();
#endif

    window.reset();
    stage.reset();
}

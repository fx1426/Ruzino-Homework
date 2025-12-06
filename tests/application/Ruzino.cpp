#include <MaterialXCore/Document.h>
#include <MaterialXFormat/XmlIo.h>
#include <rzconsole/ConsoleInterpreter.h>
#include <rzconsole/ConsoleObjects.h>
#include <rzconsole/imgui_console.h>
#include <rzconsole/spdlog_console_sink.h>
#include <spdlog/spdlog.h>

#include <any>
#include <filesystem>
#include <rzpython/interpreter.hpp>
#include <rzpython/rzpython.hpp>

#include "GCore/GOP.h"
#include "GCore/algorithms/intersection.h"
#include "GCore/geom_payload.hpp"
#include "GUI/window.h"
#include "MCore/MaterialXDocumentViewer.hpp"
#include "MCore/MaterialXNodeTree.hpp"
#include "MCore/MaterialXNodeTreeWidget.h"
#include "cmdparser.hpp"
#include "nodes/system/node_system.hpp"
#include "nodes/ui/imgui.hpp"
#include "pxr/base/tf/setenv.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usdGeom/sphere.h"
#include "pxr/usd/usdShade/material.h"
#include "pxr/usd/usdShade/shader.h"
#include "stage/stage.hpp"
#include "usd_nodejson.hpp"
#include "widgets/usdtree/usd_fileviewer.h"
#include "widgets/usdview/usdview_widget.hpp"

using namespace USTC_CG;
namespace mx = MaterialX;

class MaterialXNodeSystem : public NodeSystem {
   public:
    MaterialXNodeSystem()
    {
        descriptor = std::make_shared<MaterialXNodeTreeDescriptor>();
    }

    // Factory method to create a MaterialX system with a default material
    // document
    static std::shared_ptr<MaterialXNodeSystem> create_with_default_material(
        const std::string& material_name,
        mx::DocumentPtr ptr = nullptr)
    {
        auto system = std::make_shared<MaterialXNodeSystem>();

        mx::DocumentPtr doc;
        // Create a minimal MaterialX document in memory using MaterialX API
        if (!ptr) {
            doc = mx::createDocument();

            // Create only a surface material node
            mx::NodePtr materialNode =
                doc->addNode("surfacematerial", material_name, "material");
        }
        else {
            doc = ptr;
        }
        // Create MaterialXNodeTree with the in-memory document
        std::unique_ptr<NodeTree> tree = std::make_unique<MaterialXNodeTree>(
            system->node_tree_descriptor(), doc);

        system->init(std::move(tree));
        system->set_node_tree_executor(create_node_tree_executor({}));

        return system;
    }

    void set_node_tree_executor(
        std::unique_ptr<NodeTreeExecutor> executor) override
    {
    }

    bool load_configuration(const std::string& config) override
    {
        return true;
    }

    ~MaterialXNodeSystem() override
    {
    }

    void execute(bool is_ui_execution, Node* required_node) const override
    {
    }

    bool get_dirty()
    {
        return get_node_tree()->GetDirty();
    }

    std::shared_ptr<NodeTreeDescriptor> node_tree_descriptor() override
    {
        return descriptor;
    }

   private:
    std::shared_ptr<MaterialXNodeTreeDescriptor> descriptor;
};

class PythonConsoleWidgetFactory : public IWidgetFactory {
   public:
    std::unique_ptr<IWidget> Create(
        const std::vector<std::unique_ptr<IWidget>>& others) override
    {
        // Create Python interpreter
        auto interpreter = python::CreatePythonInterpreter();

        // Register some test commands that work with Python
        console::CommandDesc test_cmd = {
            "test",
            "A test command that demonstrates Python integration",
            [](console::Command::Args const& args) -> console::Command::Result {
                try {
                    // Use Python to do some computation
                    python::send("test_value", 42);
                    python::call<void>("test_result = test_value * 2");
                    int result = python::call<int>("test_result");

                    return { true,
                             "Test result from Python: " +
                                 std::to_string(result) + "\n" };
                }
                catch (const std::exception& e) {
                    return { false,
                             "Python test failed: " + std::string(e.what()) +
                                 "\n" };
                }
            }
        };
        console::RegisterCommand(test_cmd);

        // Test Python functionality
        console::CommandDesc math_cmd = {
            "math_test",
            "Test Python math operations",
            [](console::Command::Args const& args) -> console::Command::Result {
                try {
                    python::call<void>("import math");
                    python::send("radius", 5.0f);
                    python::call<void>("area = math.pi * radius ** 2");
                    float area = python::call<float>("area");

                    return { true,
                             "Circle area (r=5): " + std::to_string(area) +
                                 "\n" };
                }
                catch (const std::exception& e) {
                    return { false,
                             "Math test failed: " + std::string(e.what()) +
                                 "\n" };
                }
            }
        };
        console::RegisterCommand(math_cmd);

        // Add simple debug command to test if commands work at all
        console::CommandDesc simple_test_cmd = {
            "simple_test",
            "Simple test command",
            [](console::Command::Args const& args) -> console::Command::Result {
                return { true, "Simple test command works!\n" };
            }
        };
        console::RegisterCommand(simple_test_cmd);

        // Add debug command to test Python interpreter directly
        console::CommandDesc debug_cmd = {
            "debug_python",
            "Debug Python interpreter state",
            [](console::Command::Args const& args) -> console::Command::Result {
                try {
                    std::string debug_info = "Python initialized: ";
                    debug_info += python::initialized ? "Yes" : "No";
                    debug_info += "\nTesting basic operation...\n";

                    python::call<void>("test_var = 42");
                    int result = python::call<int>("test_var");
                    debug_info +=
                        "Basic test result: " + std::to_string(result) + "\n";

                    return { true, debug_info };
                }
                catch (const std::exception& e) {
                    return { false,
                             "Debug failed: " + std::string(e.what()) + "\n" };
                }
            }
        };
        console::RegisterCommand(debug_cmd);

        // Create console with capture_log enabled
        ImGui_Console::Options opts;
        opts.show_info = true;
        opts.show_warnings = true;
        opts.show_errors = true;
        opts.capture_log = false;

        auto console = std::make_unique<ImGui_Console>(interpreter, opts);

        // Add some initial messages
        console->Print("=== Python Interactive Console ===");
        console->Print("Python Console initialized successfully!");
        console->Print("FIRST: Test basic console commands:");
        console->Print("  simple_test      # Test if console commands work");
        console->Print("  debug_python     # Test Python state");
        console->Print("  math_test        # Test Python math");
        console->Print("");
        console->Print(
            "THEN: Test Python commands directly (these should work):");
        console->Print("  x = 10           # Python assignment");
        console->Print("  x                # Python variable lookup");
        console->Print("  print('hello')   # Python function call");
        console->Print("  2 + 3            # Python expression");
        console->Print("");
        console->Print(
            "You can now enter Python code directly or use console commands");
        console->Print("Type 'help' for available commands");

        return std::move(console);
    }
};

int main(int argc, char* argv[])
{
#ifdef _DEBUG
    spdlog::set_level(spdlog::level::debug);
#endif
    spdlog::set_pattern("%^[%T] %n: %v%$");
    auto window = std::make_unique<Window>();

    // Set MaterialX standard library path using USD's TfSetenv (preferred
    // method)
    std::string mtlx_stdlib = "libraries";
    if (std::filesystem::exists(mtlx_stdlib)) {
        pxr::TfSetenv("PXR_MTLX_STDLIB_SEARCH_PATHS", mtlx_stdlib.c_str());
        spdlog::info("Set PXR_MTLX_STDLIB_SEARCH_PATHS={}", mtlx_stdlib);
    }
    else {
        spdlog::warn("MaterialX stdlib not found at {}", mtlx_stdlib);
    }

    python::initialize();
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

    // Register Python Console widget in menu
    auto python_console_factory =
        std::make_unique<PythonConsoleWidgetFactory>();
    window->register_widget(python_console_factory->Create({}));
    window->register_openable_widget(
        std::move(python_console_factory), { "Tools", "Python Console" });

    // Add Python reference to window for console access
    python::reference("window", window.get());

    // Subscribe to material editor events
    window->events().subscribe(
        "material_editor_requested",
        [&stage, &window](const std::string& material_path_str) {
            spdlog::info(
                "Material editor requested for: {}", material_path_str);

            pxr::SdfPath material_path(material_path_str);
            auto material_prim =
                stage->get_usd_stage()->GetPrimAtPath(material_path);

            if (!material_prim) {
                spdlog::error("Material prim not found: {}", material_path_str);
                return;
            }

            // Step 1: Create MaterialX file next to the stage file
            std::string stage_path = stage->GetStagePath();
            std::filesystem::path stage_file(stage_path);
            std::filesystem::path stage_dir = stage_file.parent_path();

            // Get material name from the prim
            std::string material_name = material_prim.GetName();
            std::string mtlx_filename = material_name + ".mtlx";
            std::filesystem::path mtlx_path = stage_dir / mtlx_filename;

            // Check if MaterialX file exists AND material prim already has a
            // reference
            bool has_mtlx_file = std::filesystem::exists(mtlx_path);
            bool has_reference = false;

            // Check if material already has references using GetPrimStack
            auto prim_stack = material_prim.GetPrimStack();
            for (const auto& spec : prim_stack) {
                if (spec->HasReferences()) {
                    has_reference = true;
                    spdlog::info("Material already has reference(s)");
                    break;
                }
            }

            std::shared_ptr<MaterialXNodeSystem> mtlx_system;

            // Only load existing if BOTH file exists AND reference exists
            if (has_mtlx_file && has_reference) {
                // Load existing MaterialX document
                spdlog::info(
                    "Loading existing MaterialX file: {}", mtlx_path.string());
                try {
                    mx::DocumentPtr existing_doc = mx::createDocument();
                    mx::readFromXmlFile(
                        existing_doc, mx::FilePath(mtlx_path.string()));

                    // Create system with existing document using factory method
                    mtlx_system =
                        MaterialXNodeSystem::create_with_default_material(
                            material_name, existing_doc);

                    spdlog::info(
                        "Successfully loaded existing MaterialX document");
                }
                catch (const std::exception& e) {
                    spdlog::error(
                        "Failed to load existing MaterialX file: {}", e.what());
                    has_mtlx_file = false;
                    has_reference = false;
                }
            }

            // Create new if file doesn't exist OR reference doesn't exist
            if (!has_mtlx_file || !has_reference) {
                // Create new MaterialX document
                spdlog::info(
                    "Creating new MaterialX file at: {}", mtlx_path.string());

                mtlx_system = MaterialXNodeSystem::create_with_default_material(
                    material_name);

                // Save the new MaterialX document to file
                auto* mtlx_tree_temp = static_cast<MaterialXNodeTree*>(
                    mtlx_system->get_node_tree());
                mtlx_tree_temp->saveDocument(mx::FilePath(mtlx_path.string()));

                // MaterialX materials are at root level in the document
                std::string mtlx_material_path_str = "/" + material_name;

                // Add reference to the MaterialX file (clear existing first if
                // any)
                auto references = material_prim.GetReferences();
                references.ClearReferences();
                references.AddReference(
                    pxr::SdfReference(
                        mtlx_path.string(),
                        pxr::SdfPath(mtlx_material_path_str)));

                spdlog::info(
                    "Added reference: {} -> {}",
                    mtlx_path.string(),
                    mtlx_material_path_str);

                // Save stage to persist the reference
                stage->get_usd_stage()->Save();
            }

            // Launch MaterialX editor widget
            FileBasedNodeWidgetSettings widget_desc;
            widget_desc.system = mtlx_system;
            widget_desc.json_path =
                (stage_dir / (material_name + "_layout.json")).string();

            std::unique_ptr<IWidget> node_widget =
                std::make_unique<MaterialXNodeTreeWidget>(
                    widget_desc, mtlx_path.string(), material_path_str);

            // Setup callback to save MaterialX file and update USD reference
            // when editor closes
            auto mtlx_path_copy = mtlx_path.string();
            auto material_name_copy = material_name;
            auto stage_ptr = stage.get();
            auto* mtlx_tree =
                static_cast<MaterialXNodeTree*>(mtlx_system->get_node_tree());

            window->register_widget(std::move(node_widget));
        });

    // Subscribe to MaterialX graph change events
    window->events().subscribe(
        "materialx_graph_changed",
        [&stage](const std::string& material_path_str) {
            spdlog::info("MaterialX graph changed for: {}", material_path_str);

            pxr::SdfPath material_path(material_path_str);
            auto material_prim =
                stage->get_usd_stage()->GetPrimAtPath(material_path);

            if (!material_prim) {
                spdlog::error("Material prim not found: {}", material_path_str);
                return;
            }

            // Get the MaterialX file path
            std::string stage_path = stage->GetStagePath();
            std::filesystem::path stage_file(stage_path);
            std::filesystem::path stage_dir = stage_file.parent_path();
            std::string material_name = material_prim.GetName();
            std::filesystem::path mtlx_path =
                stage_dir / (material_name + ".mtlx");

            // Load the MaterialX file to find the material and shader nodes
            mx::DocumentPtr mtlx_doc = mx::createDocument();
            try {
                mx::readFromXmlFile(mtlx_doc, mx::FilePath(mtlx_path.string()));
            }
            catch (const std::exception& e) {
                spdlog::error("Failed to read MaterialX file: {}", e.what());
                return;
            }

            std::string surface_shader_name;

            for (auto material_node : mtlx_doc->getMaterialNodes()) {
                // Find the surface shader output
                auto surfaceshader_input =
                    material_node->getInput("surfaceshader");
                if (surfaceshader_input) {
                    auto connected_node =
                        surfaceshader_input->getConnectedNode();
                    if (connected_node) {
                        surface_shader_name = connected_node->getName();
                    }
                }
                break;  // Use the first material found
            }

            // Connect surface output if the shader exists
            if (!surface_shader_name.empty()) {
                // STEP 1: Ensure MaterialX layer is loaded into USD's registry
                auto mtlx_layer = pxr::SdfLayer::FindOrOpen(mtlx_path.string());
                if (!mtlx_layer) {
                    spdlog::error(
                        "Failed to open MaterialX layer: {}",
                        mtlx_path.string());
                    stage->get_usd_stage()->Save();
                    return;
                }

                spdlog::info("MaterialX layer loaded: {}", mtlx_path.string());

                // STEP 2: Create the connection in the root layer
                auto root_layer = stage->get_usd_stage()->GetRootLayer();

                // Construct paths
                pxr::SdfPath shader_path = material_path.AppendChild(
                    pxr::TfToken(surface_shader_name));
                pxr::SdfPath shader_surface_output_path =
                    shader_path.AppendProperty(pxr::TfToken("outputs:surface"));

                // Create the material prim spec if it doesn't exist in root
                // layer
                auto material_prim_spec =
                    root_layer->GetPrimAtPath(material_path);
                if (!material_prim_spec) {
                    material_prim_spec =
                        pxr::SdfCreatePrimInLayer(root_layer, material_path);
                }

                // Create or update outputs:surface attribute
                pxr::TfToken outputs_surface_token("outputs:surface");
                auto existing_attr = material_prim_spec->GetAttributes().get(
                    outputs_surface_token);

                if (existing_attr) {
                    auto conn_list = existing_attr->GetConnectionPathList();
                    conn_list.ClearEdits();
                    conn_list.GetExplicitItems().clear();
                    conn_list.GetExplicitItems().push_back(
                        shader_surface_output_path);

                    spdlog::info(
                        "Updated surface connection: {} -> {}",
                        material_path.GetString() + ".outputs:surface",
                        shader_surface_output_path.GetString());
                }
                else {
                    auto surface_output_attr = pxr::SdfAttributeSpec::New(
                        material_prim_spec,
                        outputs_surface_token,
                        pxr::SdfValueTypeNames->Token);

                    if (surface_output_attr) {
                        auto conn_list =
                            surface_output_attr->GetConnectionPathList();
                        conn_list.GetExplicitItems().push_back(
                            shader_surface_output_path);

                        spdlog::info(
                            "Created surface connection: {} -> {}",
                            material_path.GetString() + ".outputs:surface",
                            shader_surface_output_path.GetString());
                    }
                }

                // STEP 3: Force stage to recompose by temporarily removing and
                // re-adding the reference This is the most reliable way to
                // ensure USD expands the reference
                auto references = material_prim.GetReferences();

                // Simply clear and re-add with the known reference info
                std::string ref_asset_path = mtlx_path.string();
                std::string ref_prim_path_str =
                    "/MaterialX/Materials/" + material_name;

                // Clear and re-add the reference to force recomposition
                references.ClearReferences();
                references.AddReference(
                    pxr::SdfReference(
                        ref_asset_path, pxr::SdfPath(ref_prim_path_str)));

                spdlog::info(
                    "Forced reference recomposition for material: {} -> {}",
                    material_path.GetString(),
                    ref_asset_path + ref_prim_path_str);
                stage->get_usd_stage()->Save();
                stage->get_usd_stage()->Reload();
            }
        });
    // Subscribe to document viewer events
    window->events().subscribe(
        "material_doc_viewer_requested",
        [&stage, &window](const std::string& material_path_str) {
            spdlog::info(
                "Material document viewer requested for: {}",
                material_path_str);

            // TODO: Implement document viewer creation
            // This would create a MaterialXDocumentViewer widget
        });

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
    window->SetMaximized(true);
    window->run();

    unregister_cpp_type();

#ifdef GPU_GEOM_ALGORITHM
    deinit_gpu_geometry_algorithms();
#endif

    window.reset();
    stage.reset();
}

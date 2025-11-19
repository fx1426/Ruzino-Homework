
#include <spdlog/spdlog.h>

#include "GUI/window.h"
#include "MCore/MaterialXDocumentViewer.hpp"
#include "MCore/MaterialXNodeTree.hpp"
#include "MCore/MaterialXNodeTreeWidget.h"
#include "gtest/gtest.h"
#include "imgui.h"
#include "nodes/core/node_tree.hpp"
#include "nodes/system/node_system.hpp"
#include "nodes/ui/imgui.hpp"

using namespace USTC_CG;

class MaterialXNodeSystem : public NodeSystem {
   public:
    MaterialXNodeSystem()
    {
        descriptor = std::make_shared<MaterialXNodeTreeDescriptor>();
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

    std::shared_ptr<NodeTreeDescriptor> node_tree_descriptor() override
    {
        return descriptor;
    }

   private:
    std::shared_ptr<MaterialXNodeTreeDescriptor> descriptor;
};
int main()
{
    std::shared_ptr<NodeSystem> system_;
    spdlog::set_level(spdlog::level::info);
    spdlog::set_pattern("%^[%T] %n: %v%$");

    system_ = std::make_shared<MaterialXNodeSystem>();
    std::unique_ptr<NodeTree> tree = std::make_unique<MaterialXNodeTree>(
        "resources/Materials/Examples/StandardSurface/"
        "standard_surface_marble_solid.mtlx",
        system_->node_tree_descriptor());

    system_->init(std::move(tree));
    
    // Get the MaterialX node tree for the document viewer
    auto* mtlx_tree = static_cast<MaterialXNodeTree*>(system_->get_node_tree());

    Window window;
    
    FileBasedNodeWidgetSettings widget_desc;
    widget_desc.system = system_;
    system_->set_node_tree_executor(create_node_tree_executor({}));
    widget_desc.json_path = "mtlx_test.json";

    std::unique_ptr<IWidget> node_widget =
        std::move(std::make_unique<MaterialXNodeTreeWidget>(widget_desc));

    window.register_widget(std::move(node_widget));
    
    // Add the MaterialX document viewer widget
    auto document_viewer = std::make_unique<MaterialXDocumentViewer>(
        mtlx_tree,
        "MaterialX Document");
    
    auto* viewer_ptr = document_viewer.get();
    window.register_widget(std::move(document_viewer));
    
    // Subscribe to node graph change events
    window.events().subscribe("materialx_graph_changed", [viewer_ptr](const std::string&) {
        viewer_ptr->RefreshDocument();
    });
    
    // Register callback to emit events when node tree changes
    window.register_function_after_frame([](Window* window) {
        // Emit event when nodes are modified
        // This will be called after any node tree modifications
        static bool first_frame = true;
        if (first_frame) {
            first_frame = false;
            return;
        }
        
        // You can emit this event whenever the graph is modified
        // For now, we'll just emit it every frame for demo purposes
        // In production, MaterialXNodeTreeWidget should emit this when graph changes
        window->events().emit("materialx_graph_changed");
    });
    
    window.run();
}

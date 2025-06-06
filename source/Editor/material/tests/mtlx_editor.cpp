
#include "GUI/window.h"
#include "Logger/Logger.h"
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
    log::SetMinSeverity(Severity::Info);
    log::EnableOutputToConsole(true);

    system_ = std::make_shared<MaterialXNodeSystem>();
    std::unique_ptr<NodeTree> tree = std::make_unique<MaterialXNodeTree>(
        "resources/Materials/Examples/StandardSurface/"
        "standard_surface_marble_solid.mtlx",
        system_->node_tree_descriptor());

    system_->init(std::move(tree));

    Window window;

    FileBasedNodeWidgetSettings widget_desc;
    widget_desc.system = system_;
    system_->set_node_tree_executor(create_node_tree_executor({}));
    widget_desc.json_path = "mtlx_test.json";

    std::unique_ptr<IWidget> node_widget =
        std::move(std::make_unique<MaterialXNodeTreeWidget>(widget_desc));

    window.register_widget(std::move(node_widget));
    window.run();
}

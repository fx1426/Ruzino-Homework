#pragma once
#define IMGUI_DEFINE_MATH_OPERATORS

#include <string>

#include "RHI/rhi.hpp"
#include "blueprints/builders.h"
#include "blueprints/images.inl"
#include "blueprints/imgui_node_editor.h"
#include "blueprints/widgets.h"
#include "imgui.h"
#include "nodes/core/node_link.hpp"
#include "nodes/core/node_tree.hpp"
#include "nodes/core/socket.hpp"
#include "nodes/system/node_system.hpp"
#include "nodes/ui/imgui.hpp"
#include "nodes/ui/node_editor_widget_base.hpp"

USTC_CG_NAMESPACE_OPEN_SCOPE
namespace ed = ax::NodeEditor;
namespace util = ax::NodeEditor::Utilities;
using namespace ax;
using ax::Widgets::IconType;

struct NodeIdLess {
    bool operator()(const NodeId& lhs, const NodeId& rhs) const
    {
        return lhs.AsPointer() < rhs.AsPointer();
    }
};

class NodeWidget : public NodeEditorWidgetBase {
   public:
    explicit NodeWidget(const NodeWidgetSettings& desc);

    ~NodeWidget() override;
    std::vector<Node*> create_node_menu(bool cursor);

   protected:
    void initialize() override;

    std::string GetWindowUniqueName() override;

    const char* GetWindowName() override;

    void SetNodeSystemDirty(bool dirty) override;

    void execute_tree(Node* node) override
    {
        if (tree_->GetDirty()) {
            system_->execute(true, node);
            tree_->SetDirty(false);
        }
    }

    void ShowLeftPane(float paneWidth);

    float GetTouchProgress(NodeId id);
    const float m_TouchTime = 1.0f;
    std::map<NodeId, float, NodeIdLess> m_NodeTouchTime;

    ImVec2 newNodePostion;
    bool location_remembered = false;
    std::shared_ptr<NodeSystem> system_;

    std::string widget_name;

    float leftPaneWidth = 400.0f;
    float rightPaneWidth = 800.0f;

    bool draw_socket_controllers(NodeSocket* input) override;

    void create_new_node(ImVec2 openPopupPosition) override;

    ImGuiWindowFlags GetWindowFlag() override;

    ImVector<nvrhi::TextureHandle> m_Textures;

    void ShowInputOrOutput(
        const NodeSocket& socket,
        const entt::meta_any& value);

    std::vector<Node*> add_node(const std::string& id_name);
};

USTC_CG_NAMESPACE_CLOSE_SCOPE
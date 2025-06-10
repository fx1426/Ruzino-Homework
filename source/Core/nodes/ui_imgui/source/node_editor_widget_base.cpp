#include "nodes/ui/node_editor_widget_base.hpp"

#include "entt/core/type_info.hpp"
#include "entt/meta/meta.hpp"
#include "nodes/core/node_exec_eager.hpp"
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui_internal.h>

#include <fstream>
#include <string>

#include "RHI/rhi.hpp"
#include "blueprints/builders.h"
#include "blueprints/images.inl"
#include "blueprints/imgui_node_editor.h"
#include "blueprints/widgets.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "nodes/core/node_link.hpp"
#include "nodes/core/node_tree.hpp"
#include "nodes/core/socket.hpp"
#include "nodes/system/node_system.hpp"
#include "nodes/ui/imgui.hpp"
#include "stb_image.h"
#include "ui_imgui.hpp"

USTC_CG_NAMESPACE_OPEN_SCOPE

NodeEditorWidgetBase::NodeEditorWidgetBase(const NodeWidgetSettings& desc)
    : storage_(desc.create_storage()),
      tree_(desc.system->get_node_tree())
{
    m_HeaderBackground =
        LoadTexture(BlueprintBackground, sizeof(BlueprintBackground));

    NodeEditorWidgetBase::initialize();
}

bool NodeEditorWidgetBase::BuildUI()
{
    if (first_draw) {
        first_draw = false;
        return true;
    }
    auto& io = ImGui::GetIO();

    io.ConfigFlags =
        ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_DockingEnable;
    ed::SetCurrentEditor(m_Editor);

    // if (ed::GetSelectedObjectCount() > 0) {
    //     Splitter(true, 4.0f, &leftPaneWidth,
    //     &rightPaneWidth, 50.0f, 50.0f); ShowLeftPane(leftPaneWidth
    //     - 4.0f); ImGui::SameLine(0.0f, 12.0f);
    // }

    execute_tree(nullptr);

    ed::Begin(GetWindowUniqueName().c_str(), ImGui::GetContentRegionAvail());
    {
        auto cursorTopLeft = ImGui::GetCursorScreenPos();

        util::BlueprintNodeBuilder builder(
            m_HeaderBackground.Get(),
            m_HeaderBackground->getDesc().width,
            m_HeaderBackground->getDesc().height);

        for (auto&& node : tree_->nodes) {
            if (node->typeinfo->INVISIBLE) {
                continue;
            }

            constexpr auto isSimple = false;

            builder.Begin(node->ID);
            if constexpr (!isSimple) {
                ImColor color;
                memcpy(&color, node->Color, sizeof(ImColor));
                if (node->MISSING_INPUT) {
                    color = ImColor(255, 206, 69, 255);
                }
                if (!node->REQUIRED) {
                    color = ImColor(18, 15, 16, 255);
                }

                if (!node->execution_failed.empty()) {
                    color = ImColor(255, 0, 0, 255);
                }
                builder.Header(color);
                ImGui::Spring(0);
                ImGui::TextUnformatted(node->ui_name.c_str());
                if (!node->execution_failed.empty()) {
                    ImGui::TextUnformatted(
                        (": " + node->execution_failed).c_str());
                }
                ImGui::Spring(1);
                ImGui::Dummy(ImVec2(0, 28));
                ImGui::Spring(0);
                builder.EndHeader();
            }

            for (auto& input : node->get_inputs()) {
                auto alpha = ImGui::GetStyle().Alpha;
                if (newLinkPin && !tree_->can_create_link(newLinkPin, input) &&
                    input != newLinkPin)
                    alpha = alpha * (48.0f / 255.0f);

                builder.Input(input->ID);
                ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);

                DrawPinIcon(
                    *input,
                    tree_->is_pin_linked(input->ID),
                    (int)(alpha * 255));
                ImGui::Spring(0);

                if (tree_->is_pin_linked(input->ID)) {
                    ImGui::TextUnformatted(input->ui_name);
                    ImGui::Spring(0);
                }
                else {
                    ImGui::PushItemWidth(120.0f);
                    if (draw_socket_controllers(input))
                        tree_->SetDirty();
                    ImGui::PopItemWidth();
                    ImGui::Spring(0);
                }

                ImGui::PopStyleVar();
                builder.EndInput();
            }

            for (auto& output : node->get_outputs()) {
                auto alpha = ImGui::GetStyle().Alpha;
                if (newLinkPin && !tree_->can_create_link(newLinkPin, output) &&
                    output != newLinkPin)
                    alpha = alpha * (48.0f / 255.0f);

                ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
                builder.Output(output->ID);

                ImGui::Spring(0);
                ImGui::TextUnformatted(output->ui_name);
                ImGui::Spring(0);
                DrawPinIcon(
                    *output,
                    tree_->is_pin_linked(output->ID),
                    (int)(alpha * 255));
                ImGui::PopStyleVar();

                builder.EndOutput();
            }

            builder.End();
        }

        connectLinks();

        if (!createNewNode) {
            if (ed::BeginCreate(ImColor(255, 255, 255), 2.0f)) {
                auto showLabel = [](const char* label, ImColor color) {
                    ImGui::SetCursorPosY(
                        ImGui::GetCursorPosY() - ImGui::GetTextLineHeight());
                    auto size = ImGui::CalcTextSize(label);

                    auto padding = ImGui::GetStyle().FramePadding;
                    auto spacing = ImGui::GetStyle().ItemSpacing;

                    ImGui::SetCursorPos(
                        ImGui::GetCursorPos() + ImVec2(spacing.x, -spacing.y));

                    auto rectMin = ImGui::GetCursorScreenPos() - padding;
                    auto rectMax = ImGui::GetCursorScreenPos() + size + padding;

                    auto drawList = ImGui::GetWindowDrawList();
                    drawList->AddRectFilled(
                        rectMin, rectMax, color, size.y * 0.15f);
                    ImGui::TextUnformatted(label);
                };

                SocketID startPinId = 0, endPinId = 0;
                if (ed::QueryNewLink(&startPinId, &endPinId)) {
                    auto startPin = tree_->find_pin(startPinId);
                    auto endPin = tree_->find_pin(endPinId);

                    newLinkPin = startPin ? startPin : endPin;

                    if (startPin && endPin) {
                        if (tree_->can_create_link(startPin, endPin)) {
                            showLabel(
                                "+ Create Link", ImColor(32, 45, 32, 180));
                            if (ed::AcceptNewItem(
                                    ImColor(128, 255, 128), 4.0f)) {
                                tree_->add_link(startPinId, endPinId);
                            }
                        }
                    }
                }

                SocketID pinId = 0;
                if (ed::QueryNewNode(&pinId)) {
                    newLinkPin = tree_->find_pin(pinId);
                    if (newLinkPin)
                        showLabel("+ Create Node", ImColor(32, 45, 32, 180));

                    if (ed::AcceptNewItem()) {
                        createNewNode = true;
                        newNodeLinkPin = tree_->find_pin(pinId);
                        newLinkPin = nullptr;
                        ed::Suspend();
                        create_new_node_search_cursor = true;
                        ImGui::OpenPopup("Create New Node");
                        ed::Resume();
                    }
                }
            }
            else
                newLinkPin = nullptr;

            ed::EndCreate();

            if (ed::BeginDelete()) {
                NodeId nodeId = 0;
                while (ed::QueryDeletedNode(&nodeId)) {
                    if (ed::AcceptDeletedItem()) {
                        auto id = std::find_if(
                            tree_->nodes.begin(),
                            tree_->nodes.end(),
                            [nodeId](auto& node) {
                                return node->ID == nodeId;
                            });
                        if (id != tree_->nodes.end())
                            tree_->delete_node(nodeId);
                    }
                }

                LinkId linkId = 0;
                while (ed::QueryDeletedLink(&linkId)) {
                    if (ed::AcceptDeletedItem()) {
                        tree_->delete_link(linkId);
                    }
                }

                // tree_->trigger_refresh_topology();
            }
            ed::EndDelete();
        }

        ImGui::SetCursorScreenPos(cursorTopLeft);
    }

    auto openPopupPosition = ImGui::GetMousePos();

    std::vector<NodeId> selectedNodes;
    selectedNodes.resize(ed::GetSelectedObjectCount());

    int nodeCount = ed::GetSelectedNodes(
        selectedNodes.data(), static_cast<int>(selectedNodes.size()));
    selectedNodes.resize(nodeCount);

    ed::Suspend();
    if (ed::ShowNodeContextMenu(&contextNodeId))
        ImGui::OpenPopup("Node Context Menu");
    else if (ed::ShowPinContextMenu(&contextPinId))
        ImGui::OpenPopup("Pin Context Menu");
    else if (ed::ShowLinkContextMenu(&contextLinkId))
        ImGui::OpenPopup("Link Context Menu");
    else if (ed::ShowBackgroundContextMenu()) {
        ImGui::OpenPopup("Create New Node");
        create_new_node_search_cursor = true;
        newNodeLinkPin = nullptr;
    }

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 8));
    if (ImGui::BeginPopup("Node Context Menu")) {
        auto node = tree_->find_node(contextNodeId);

        ImGui::TextUnformatted("Node Context Menu");
        ImGui::Separator();
        if (node) {
            ImGui::Text("ID: %p", node->ID.AsPointer());
            ImGui::Text("Inputs: %d", (int)node->get_inputs().size());
            ImGui::Text("Outputs: %d", (int)node->get_outputs().size());
        }
        else
            ImGui::Text("Unknown node: %p", contextNodeId.AsPointer());
        ImGui::Separator();
        if (ImGui::MenuItem("Run")) {
            execute_tree(node);
        }
        if (node->is_node_group())
            if (ImGui::MenuItem("UnGroup")) {
                if (node) {
                    ed::DeleteNode(node->ID);
                    tree_->ungroup(node);
                }
            }

        if (selectedNodes.size() > 1)
            if (ImGui::MenuItem("Group")) {
                auto group_node = tree_->group_up(selectedNodes);
                if (group_node) {
                    tree_->SetDirty();
                    ed::SetNodePosition(group_node->ID, openPopupPosition);
                }
            }

        if (ImGui::MenuItem("Delete"))
            ed::DeleteNode(contextNodeId);
        ImGui::EndPopup();
    }

    if (ImGui::BeginPopup("Pin Context Menu")) {
        auto pin = tree_->find_pin(contextPinId);

        ImGui::TextUnformatted("Pin Context Menu");
        ImGui::Separator();
        if (pin) {
            ImGui::Text("ID: %p", pin->ID.AsPointer());
            if (pin->node)
                ImGui::Text("Node: %p", pin->node->ID.AsPointer());
            else
                ImGui::Text("Node: %s", "<none>");
        }
        else
            ImGui::Text("Unknown pin: %p", contextPinId.AsPointer());

        ImGui::EndPopup();
    }

    if (ImGui::BeginPopup("Link Context Menu")) {
        auto link = tree_->find_link(contextLinkId);

        ImGui::TextUnformatted("Link Context Menu");
        ImGui::Separator();
        if (link) {
            ImGui::Text("ID: %p", link->ID.AsPointer());
            ImGui::Text("From: %p", link->StartPinID.AsPointer());
            ImGui::Text("To: %p", link->EndPinID.AsPointer());
        }
        else
            ImGui::Text("Unknown link: %p", contextLinkId.AsPointer());
        ImGui::Separator();
        if (ImGui::MenuItem("Delete"))
            ed::DeleteLink(contextLinkId);
        ImGui::EndPopup();
    }

    create_new_node(openPopupPosition);

    ImGui::PopStyleVar();
    ed::Resume();

    ed::End();

    // None
    // io.ConfigFlags = ImGuiConfigFlags_None;

    return true;
}

void NodeEditorWidgetBase::connectLinks()
{
    for (std::unique_ptr<NodeLink>& link : tree_->links) {
        auto type = link->from_sock->type_info;
        if (!type)
            type = link->to_sock->type_info;

        ImColor color = GetIconColor(type);

        auto linkId = link->ID;
        auto startPin = link->StartPinID;
        auto endPin = link->EndPinID;

        // If there is an invisible node after the link, use the first as
        // the id for the ui link
        if (link->nextLink) {
            endPin = link->nextLink->to_sock->ID;
        }

        ed::Link(linkId, startPin, endPin, color, 2.0f);
    }
}

ImColor NodeEditorWidgetBase::GetIconColor(SocketType type)
{
    // Compute a hash for the hue based on type name.
    auto hashHSVComponent = [](const std::string& prefix,
                               const std::string& typeName) {
        return static_cast<unsigned int>(
            entt::hashed_string{ (prefix + typeName).c_str() }.value());
    };

    const std::string typeName = get_type_name(type);
    unsigned int hashHue = hashHSVComponent("h", typeName);
    // Map the hash to a hue in [0, 360)
    float hue = static_cast<float>(hashHue % 360);
    // Set saturation and value to ensure colors are bright and not too dark.
    float saturation = 0.8f;
    float value = 0.9f;

    // Lambda to convert HSV (h in degrees, s and v in [0, 1]) to RGB in [0,
    // 255]
    auto hsv2rgb = [](float h, float s, float v) -> std::tuple<int, int, int> {
        float C = v * s;
        float X = C * (1 - fabs(fmod(h / 60.0f, 2) - 1));
        float m = v - C;
        float r1 = 0, g1 = 0, b1 = 0;

        if (h < 60) {
            r1 = C;
            g1 = X;
            b1 = 0;
        }
        else if (h < 120) {
            r1 = X;
            g1 = C;
            b1 = 0;
        }
        else if (h < 180) {
            r1 = 0;
            g1 = C;
            b1 = X;
        }
        else if (h < 240) {
            r1 = 0;
            g1 = X;
            b1 = C;
        }
        else if (h < 300) {
            r1 = X;
            g1 = 0;
            b1 = C;
        }
        else {
            r1 = C;
            g1 = 0;
            b1 = X;
        }

        int r = static_cast<int>((r1 + m) * 255);
        int g = static_cast<int>((g1 + m) * 255);
        int b = static_cast<int>((b1 + m) * 255);
        return { r, g, b };
    };

    auto [r, g, b] = hsv2rgb(hue, saturation, value);
    return ImColor(r, g, b);
}

void NodeEditorWidgetBase::DrawPinIcon(
    const NodeSocket& pin,
    bool connected,
    int alpha)
{
    IconType iconType;

    ImColor color = GetIconColor(pin.type_info);

    if (!pin.type_info) {
        if (pin.directly_linked_sockets.size() > 0) {
            color = GetIconColor(pin.directly_linked_sockets[0]->type_info);
        }
    }

    color.Value.w = alpha / 255.0f;
    iconType = IconType::Circle;

    Widgets::Icon(
        ImVec2(
            static_cast<float>(m_PinIconSize),
            static_cast<float>(m_PinIconSize)),
        iconType,
        connected,
        color,
        ImColor(32, 32, 32, alpha));
}

nvrhi::TextureHandle NodeEditorWidgetBase::LoadTexture(
    const unsigned char* data,
    size_t buffer_size)
{
    int width = 0, height = 0, component = 0;
    if (auto loaded_data = stbi_load_from_memory(
            data, buffer_size, &width, &height, &component, 4)) {
        nvrhi::TextureDesc desc;
        desc.width = width;
        desc.height = height;
        desc.format = nvrhi::Format::RGBA8_UNORM;
        desc.isRenderTarget = false;
        desc.isUAV = false;
        desc.initialState = nvrhi::ResourceStates::ShaderResource;
        desc.keepInitialState = true;

        auto [texture, _] = RHI::load_texture(desc, loaded_data);
        stbi_image_free(loaded_data);
        return texture;
    }
    else
        return nullptr;
}

USTC_CG_NAMESPACE_CLOSE_SCOPE

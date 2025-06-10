

#include "entt/core/type_info.hpp"
#include "entt/meta/meta.hpp"
#include "nodes/core/node_exec_eager.hpp"
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui_internal.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>

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
namespace ed = ax::NodeEditor;
namespace util = ax::NodeEditor::Utilities;
using namespace ax;
using ax::Widgets::IconType;

static ImRect ImGui_GetItemRect()
{
    return ImRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
}

static ImRect ImRect_Expanded(const ImRect& rect, float x, float y)
{
    auto result = rect;
    result.Min.x -= x;
    result.Min.y -= y;
    result.Max.x += x;
    result.Max.y += y;
    return result;
}

static bool Splitter(
    bool split_vertically,
    float thickness,
    float* size1,
    float* size2,
    float min_size1,
    float min_size2,
    float splitter_long_axis_size = -1.0f)
{
    using namespace ImGui;
    ImGuiContext& g = *GImGui;
    ImGuiWindow* window = g.CurrentWindow;
    ImGuiID id = window->GetID("##Splitter");
    ImRect bb;
    bb.Min = window->DC.CursorPos +
             (split_vertically ? ImVec2(*size1, 0.0f) : ImVec2(0.0f, *size1));
    bb.Max = bb.Min + CalcItemSize(
                          split_vertically
                              ? ImVec2(thickness, splitter_long_axis_size)
                              : ImVec2(splitter_long_axis_size, thickness),
                          0.0f,
                          0.0f);
    return SplitterBehavior(
        bb,
        id,
        split_vertically ? ImGuiAxis_X : ImGuiAxis_Y,
        size1,
        size2,
        min_size1,
        min_size2,
        0.0f);
}

NodeWidget::NodeWidget(const NodeWidgetSettings& desc)
    : NodeEditorWidgetBase(desc),
      system_(desc.system),
      widget_name(desc.WidgetName())
{
    NodeWidget::initialize();
}

void NodeWidget::initialize()
{
    ed::Config config;

    config.UserPointer = this;

    config.SaveSettings = [](const char* data,
                             size_t size,
                             ax::NodeEditor::SaveReasonFlags reason,
                             void* userPointer) -> bool {
        if (static_cast<bool>(
                reason & (NodeEditor::SaveReasonFlags::Navigation))) {
            return true;
        }
        auto ptr = static_cast<NodeWidget*>(userPointer);
        auto storage = ptr->storage_.get();

        auto ui_json = std::string(data + 1, size - 2);

        ptr->tree_->set_ui_settings(ui_json);

        std::string node_serialize = ptr->tree_->serialize();

        storage->save(node_serialize);
        return true;
    };

    config.LoadSettings = [](void* userPointer) {
        auto ptr = static_cast<NodeWidget*>(userPointer);
        auto storage = ptr->storage_.get();

        std::string data = storage->load();
        if (!data.empty()) {
            ptr->tree_->deserialize(data);
        }

        return data;
    };

    m_Editor = ed::CreateEditor(&config);
}

NodeWidget::~NodeWidget()
{
    ed::SetCurrentEditor(m_Editor);
    ed::DestroyEditor(m_Editor);
}

std::vector<Node*> NodeWidget::create_node_menu(bool cursor)
{
    bool open_AddPopup = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootWindow) &&
                         ImGui::IsKeyReleased(ImGuiKey_Tab);

    auto& node_registry = tree_->get_descriptor()->node_registry;

    std::vector<Node*> nodes = {};

    static char input[32]{ "" };

    ImGui::Text("Add Node");
    ImGui::Separator();
    if (cursor)
        ImGui::SetKeyboardFocusHere();
    ImGui::InputText("##input", input, sizeof(input));
    std::string subs(input);

    for (auto&& value : node_registry) {
        auto name = value.second.ui_name;

        auto id_name = value.second.id_name;
        std::ranges::replace(subs, ' ', '_');

        if (subs.size() > 0) {
            ImGui::SetNextWindowSizeConstraints(
                ImVec2(250.0f, 300.0f), ImVec2(-1.0f, 500.0f));

            if (name.find(subs) != std::string::npos) {
                if (ImGui::MenuItem(name.c_str()) ||
                    (ImGui::IsItemFocused() &&
                     ImGui::IsKeyPressed(ImGuiKey_Enter))) {
                    nodes = add_node(id_name);
                    memset(input, '\0', sizeof(input));
                    ImGui::CloseCurrentPopup();
                    break;
                }
            }
        }
        else {
            ImGui::SetNextWindowSizeConstraints(
                ImVec2(100, 10), ImVec2(-1, 300));

            if (ImGui::MenuItem(name.c_str())) {
                nodes = add_node(id_name);
                break;
            }
        }
    }

    return nodes;
}

std::string NodeWidget::GetWindowUniqueName()
{
    if (!widget_name.empty()) {
        return widget_name;
    }
    return "NodeEditor##" +
           std::to_string(reinterpret_cast<uint64_t>(system_.get()));
}

const char* NodeWidget::GetWindowName()
{
    return "Node editor";
}

void NodeWidget::SetNodeSystemDirty(bool dirty)
{
    tree_->SetDirty(dirty);
}

void NodeWidget::ShowInputOrOutput(
    const NodeSocket& socket,
    const entt::meta_any& value)
{
    if (value) {
        // 若输入为int float string类型，直接显示
        // 否则检查是否可以转换为int float string
        switch (value.type().info().hash()) {
            case entt::type_hash<int>().value():
                ImGui::Text("%s: %d", socket.ui_name, value.cast<int>());
                break;
            case entt::type_hash<long long>().value():
                ImGui::Text(
                    "%s: %lld", socket.ui_name, value.cast<long long>());
                break;
            case entt::type_hash<unsigned>().value():
                ImGui::Text("%s: %u", socket.ui_name, value.cast<unsigned>());
                break;
            case entt::type_hash<unsigned long long>().value():
                ImGui::Text(
                    "%s: %llu",
                    socket.ui_name,
                    value.cast<unsigned long long>());
                break;
            case entt::type_hash<float>().value():
                ImGui::Text("%s: %f", socket.ui_name, value.cast<float>());
                break;
            case entt::type_hash<double>().value():
                ImGui::Text("%s: %f", socket.ui_name, value.cast<double>());
                break;
            case entt::type_hash<std::string>().value():
                ImGui::Text(
                    "%s: %s",
                    socket.ui_name,
                    value.cast<std::string>().c_str());
                break;
            case entt::type_hash<bool>().value():
                ImGui::Text(
                    "%s: %s",
                    socket.ui_name,
                    value.cast<bool>() ? "true" : "false");
                break;
            case entt::type_hash<char>().value():
                ImGui::Text("%s: %c", socket.ui_name, value.cast<char>());
                break;
            case entt::type_hash<unsigned char>().value():
                ImGui::Text(
                    "%s: %u", socket.ui_name, value.cast<unsigned char>());
                break;
            case entt::type_hash<short>().value():
                ImGui::Text("%s: %d", socket.ui_name, value.cast<short>());
                break;
            case entt::type_hash<unsigned short>().value():
                ImGui::Text(
                    "%s: %u", socket.ui_name, value.cast<unsigned short>());
                break;
            default: {
                ImGui::Text(
                    "%s: %s (%s)",
                    socket.ui_name,
                    "Unknown Type",
                    value.type().info().name().data());
            }
        }
    }
    else {
        ImGui::Text("%s: %s", socket.ui_name, "Not Executed");
    }
}

std::vector<Node*> NodeWidget::add_node(const std::string& id_name)
{
    std::vector<Node*> nodes;

    auto from_node = tree_->add_node(id_name.c_str());
    nodes.push_back(from_node);

    auto synchronization =
        system_->node_tree_descriptor()->require_syncronization(id_name);

    if (!synchronization.empty()) {
        assert(synchronization.size() > 1);
        std::map<std::string, Node*> related_node_created;
        related_node_created[id_name] = from_node;
        for (auto& group : synchronization) {
            auto node_name = std::get<0>(group);

            if (related_node_created.find(node_name) ==
                related_node_created.end()) {
                auto node = tree_->add_node(node_name.c_str());
                nodes.push_back(node);
                related_node_created[node_name] = node;
            }
        }

        for (auto it1 = synchronization.begin(); it1 != synchronization.end();
             ++it1) {
            for (auto it2 = std::next(it1); it2 != synchronization.end();
                 ++it2) {
                auto& [nodeName1, groupName1, groupKind1] = *it1;
                auto& [nodeName2, groupName2, groupKind2] = *it2;
                auto* from_node = related_node_created[nodeName1];
                auto* to_node = related_node_created[nodeName2];
                auto* from_group =
                    from_node->find_socket_group(groupName1, groupKind1);
                auto* to_group =
                    to_node->find_socket_group(groupName2, groupKind2);
                from_group->add_sync_group(to_group);
            }
        }
    }

    if (nodes.size() == 2) {
        nodes[0]->paired_node = nodes[1];
        nodes[1]->paired_node = nodes[0];
    }

    return nodes;
}

void NodeWidget::ShowLeftPane(float paneWidth)
{
    auto& io = ImGui::GetIO();

    std::vector<NodeId> selectedNodes;
    std::vector<LinkId> selectedLinks;
    selectedNodes.resize(ed::GetSelectedObjectCount());
    selectedLinks.resize(ed::GetSelectedObjectCount());

    int nodeCount = ed::GetSelectedNodes(
        selectedNodes.data(), static_cast<int>(selectedNodes.size()));
    int linkCount = ed::GetSelectedLinks(
        selectedLinks.data(), static_cast<int>(selectedLinks.size()));

    selectedNodes.resize(nodeCount);
    selectedLinks.resize(linkCount);
    ImGui::BeginChild("Selection", ImVec2(paneWidth, 0));

    ImGui::Text(
        "FPS: %.2f (%.2gms)",
        io.Framerate,
        io.Framerate ? 1000.0f / io.Framerate : 0.0f);

    paneWidth = ImGui::GetContentRegionAvail().x;

    ImGui::BeginHorizontal("Style Editor", ImVec2(paneWidth, 0));
    ImGui::Spring(0.0f, 0.0f);
    if (ImGui::Button("Zoom to Content"))
        ed::NavigateToContent();
    ImGui::Spring(0.0f);
    if (ImGui::Button("Show Flow")) {
        for (auto& link : tree_->links)
            ed::Flow(link->ID);
    }
    ImGui::Spring();
    ImGui::EndHorizontal();
    ImGui::GetWindowDrawList()->AddRectFilled(
        ImGui::GetCursorScreenPos(),
        ImGui::GetCursorScreenPos() +
            ImVec2(paneWidth, ImGui::GetTextLineHeight()),
        ImColor(ImGui::GetStyle().Colors[ImGuiCol_HeaderActive]),
        ImGui::GetTextLineHeight() * 0.25f);
    ImGui::Spacing();
    ImGui::SameLine();
    ImGui::TextUnformatted("Nodes");
    ImGui::Indent();
    for (auto& node : tree_->nodes) {
        ImGui::PushID(node->ID.AsPointer());
        auto start = ImGui::GetCursorScreenPos();

        if (const auto progress = GetTouchProgress(node->ID)) {
            ImGui::GetWindowDrawList()->AddLine(
                start + ImVec2(-8, 0),
                start + ImVec2(-8, ImGui::GetTextLineHeight()),
                IM_COL32(255, 0, 0, 255 - (int)(255 * progress)),
                4.0f);
        }

        bool isSelected =
            std::find(selectedNodes.begin(), selectedNodes.end(), node->ID) !=
            selectedNodes.end();
        ImGui::SetNextItemAllowOverlap();
        if (ImGui::Selectable(
                (node->ui_name + "##" +
                 std::to_string(
                     reinterpret_cast<uintptr_t>(node->ID.AsPointer())))
                    .c_str(),
                &isSelected)) {
            if (io.KeyCtrl) {
                if (isSelected)
                    ed::SelectNode(node->ID, true);
                else
                    ed::DeselectNode(node->ID);
            }
            else
                ed::SelectNode(node->ID, false);

            ed::NavigateToSelection();
        }
        ImGui::PopID();
    }
    ImGui::Unindent();

    static int changeCount = 0;

    ImGui::GetWindowDrawList()->AddRectFilled(
        ImGui::GetCursorScreenPos(),
        ImGui::GetCursorScreenPos() +
            ImVec2(paneWidth, ImGui::GetTextLineHeight()),
        ImColor(ImGui::GetStyle().Colors[ImGuiCol_HeaderActive]),
        ImGui::GetTextLineHeight() * 0.25f);
    ImGui::Spacing();
    ImGui::SameLine();
    ImGui::TextUnformatted("Selection");

    ImGui::Indent();
    EagerNodeTreeExecutor* executor =
        dynamic_cast<EagerNodeTreeExecutor*>(system_->get_node_tree_executor());
    for (int i = 0; i < nodeCount; ++i) {
        ImGui::Text("Node (%p)", selectedNodes[i].AsPointer());
        auto node = tree_->find_node(selectedNodes[i]);
        auto input = node->get_inputs();
        auto output = node->get_outputs();
        ImGui::Text("Inputs:");
        ImGui::Indent();
        for (auto& in : input) {
            auto input_value = *executor->FindPtr(in);
            ShowInputOrOutput(*in, input_value);
        }
        ImGui::Unindent();
        ImGui::Text("Outputs:");
        ImGui::Indent();
        for (auto& out : output) {
            auto output_value = *executor->FindPtr(out);
            ShowInputOrOutput(*out, output_value);
        }
        ImGui::Unindent();
        if (node->override_left_pane_info)
            node->override_left_pane_info();
    }

    for (int i = 0; i < linkCount; ++i)
        ImGui::Text("Link (%p)", selectedLinks[i].AsPointer());
    ImGui::Unindent();

    if (ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_Z)))
        for (auto& link : tree_->links)
            ed::Flow(link->ID);

    if (ed::HasSelectionChanged())
        ++changeCount;

    ImGui::GetWindowDrawList()->AddRectFilled(
        ImGui::GetCursorScreenPos(),
        ImGui::GetCursorScreenPos() +
            ImVec2(paneWidth, ImGui::GetTextLineHeight()),
        ImColor(ImGui::GetStyle().Colors[ImGuiCol_HeaderActive]),
        ImGui::GetTextLineHeight() * 0.25f);
    ImGui::Spacing();
    ImGui::SameLine();
    ImGui::TextUnformatted("Node Tree Info");

    ImGui::EndChild();
}

float NodeWidget::GetTouchProgress(NodeId id)
{
    auto it = m_NodeTouchTime.find(id);
    if (it != m_NodeTouchTime.end() && it->second > 0.0f)
        return (m_TouchTime - it->second) / m_TouchTime;
    else
        return 0.0f;
}

bool NodeWidget::draw_socket_controllers(NodeSocket* input)
{
    if (input->socket_group) {
        return false;
    }
    bool changed = false;
    switch (input->type_info.id()) {
        default:
            ImGui::TextUnformatted(input->ui_name);
            ImGui::Spring(0);
            break;
        case entt::type_hash<int>().value():
            changed |= ImGui::SliderInt(
                (input->ui_name + ("##" + std::to_string(input->ID.Get())))
                    .c_str(),
                &input->dataField.value.cast<int&>(),
                input->dataField.min.cast<int>(),
                input->dataField.max.cast<int>());
            break;

        case entt::type_hash<float>().value():
            changed |= ImGui::SliderFloat(
                (input->ui_name + ("##" + std::to_string(input->ID.Get())))
                    .c_str(),
                &input->dataField.value.cast<float&>(),
                input->dataField.min.cast<float>(),
                input->dataField.max.cast<float>());
            break;

        case entt::type_hash<std::string>().value():
            input->dataField.value.cast<std::string&>().resize(255);
            changed |= ImGui::InputText(
                (input->ui_name + ("##" + std::to_string(input->ID.Get())))
                    .c_str(),
                input->dataField.value.cast<std::string&>().data(),
                255);
            break;
        case entt::type_hash<bool>().value():
            changed |= ImGui::Checkbox(
                (input->ui_name + ("##" + std::to_string(input->ID.Get())))
                    .c_str(),
                &input->dataField.value.cast<bool&>());
            break;
        case entt::type_hash<pxr::GfVec2f>().value(): {
            auto& vec = input->dataField.value.cast<pxr::GfVec2f&>();
            auto min_vec = input->dataField.min.cast<pxr::GfVec2f>();
            auto max_vec = input->dataField.max.cast<pxr::GfVec2f>();
            changed |= ImGui::SliderFloat(
                ("##" + std::to_string(input->ID.Get()) + "_x").c_str(),
                &vec[0],
                min_vec[0],
                max_vec[0]);
            changed |= ImGui::SliderFloat(
                ("##" + std::to_string(input->ID.Get()) + "_y").c_str(),
                &vec[1],
                min_vec[1],
                max_vec[1]);
            ImGui::Text("%s", input->ui_name);
            break;
        }
        case entt::type_hash<pxr::GfVec3f>().value(): {
            auto& vec = input->dataField.value.cast<pxr::GfVec3f&>();
            auto min_vec = input->dataField.min.cast<pxr::GfVec3f>();
            auto max_vec = input->dataField.max.cast<pxr::GfVec3f>();
            changed |= ImGui::SliderFloat(
                ("##" + std::to_string(input->ID.Get()) + "_x").c_str(),
                &vec[0],
                min_vec[0],
                max_vec[0]);
            changed |= ImGui::SliderFloat(
                ("##" + std::to_string(input->ID.Get()) + "_y").c_str(),
                &vec[1],
                min_vec[1],
                max_vec[1]);
            changed |= ImGui::SliderFloat(
                ("##" + std::to_string(input->ID.Get()) + "_z").c_str(),
                &vec[2],
                min_vec[2],
                max_vec[2]);

            ImGui::Text("%s", input->ui_name);
            break;
        }
    }
    return changed;
}

void NodeWidget::create_new_node(ImVec2 openPopupPosition)
{
    if (ImGui::BeginPopup("Create New Node")) {
        if (!location_remembered) {
            newNodePostion = openPopupPosition;
            location_remembered = true;
        }

        std::vector<Node*> nodes =
            create_node_menu(create_new_node_search_cursor);
        create_new_node_search_cursor = false;
        // ImGui::Separator();
        // if (ImGui::MenuItem("Comment"))
        //     node = SpawnComment();
        for (auto node : nodes)
            if (node) {
                location_remembered = false;
                createNewNode = false;
                tree_->SetDirty();

                ed::SetNodePosition(node->ID, newNodePostion);

                newNodePostion += ImVec2(200, 0);

                if (auto startPin = newNodeLinkPin) {
                    auto& pins = startPin->in_out == PinKind::Input
                                     ? node->get_outputs()
                                     : node->get_inputs();

                    for (auto& pin : pins) {
                        if (tree_->can_create_link(startPin, pin)) {
                            auto endPin = pin;

                            tree_->add_link(startPin->ID, endPin->ID);

                            break;
                        }
                    }
                }

                newNodeLinkPin = nullptr;
            }

        ImGui::EndPopup();
    }
    else
        createNewNode = false;
}

ImGuiWindowFlags NodeWidget::GetWindowFlag()
{
    return ImGuiWindowFlags_NoScrollbar;
}

NodeWidgetSettings::NodeWidgetSettings()
{
}

struct NodeSystemFileStorage : public NodeSystemStorage {
    explicit NodeSystemFileStorage(const std::filesystem::path& json_path)
        : json_path_(json_path)
    {
    }

    void save(const std::string& data) override
    {
        std::ofstream file(json_path_);
        file << data;
    }

    std::string load() override
    {
        std::ifstream file(json_path_);
        if (!file) {
            return std::string();
        }

        std::string data;
        file.seekg(0, std::ios_base::end);
        auto size = static_cast<size_t>(file.tellg());
        file.seekg(0, std::ios_base::beg);

        data.reserve(size);
        data.assign(
            std::istreambuf_iterator<char>(file),
            std::istreambuf_iterator<char>());

        return data;
    }

    std::filesystem::path json_path_;
};

std::unique_ptr<NodeSystemStorage> FileBasedNodeWidgetSettings::create_storage()
    const
{
    return std::make_unique<NodeSystemFileStorage>(json_path);
}

std::string FileBasedNodeWidgetSettings::WidgetName() const
{
    return json_path.string();
}

std::unique_ptr<IWidget> create_node_imgui_widget(
    const NodeWidgetSettings& desc)
{
    return std::make_unique<NodeWidget>(desc);
}

USTC_CG_NAMESPACE_CLOSE_SCOPE

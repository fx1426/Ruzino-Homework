#pragma once

#include <GUI/widget.h>
#include <MCore/Graph.h>
#include <MaterialXFormat/Util.h>
#include <imgui_node_editor_internal.h>
#include <imgui_stdlib.h>

#include <iostream>
#include <nodes/core/node_link.hpp>

#include "MCore/api.h"
namespace mx = MaterialX;

USTC_CG_NAMESPACE_OPEN_SCOPE
class MaterialXNodeTreeDescriptor : public NodeTreeDescriptor { };

class MaterialXNodeTree : public NodeTree {
   public:
    explicit MaterialXNodeTree(
        const std::string& materialFilename,
        const std::shared_ptr<NodeTreeDescriptor>& descriptor)
        : _materialFilename(materialFilename),
          NodeTree(descriptor)
    {
        loadStandardLibraries();
        _graphDoc = loadDocument(materialFilename);

        if (_graphDoc) {
            buildUiBaseGraph(_graphDoc);
            //_currGraphElem = _graphDoc;
            //_prevUiNode = nullptr;
        }
    }

    explicit MaterialXNodeTree(const NodeTree& other) : NodeTree(other)
    {
    }

    void loadStandardLibraries();

    mx::DocumentPtr loadDocument(const mx::FilePath& filename);

    // Build UiNode nodegraph upon loading a document
    void buildUiBaseGraph(mx::DocumentPtr doc);

    // Build UiNode node graph upon diving into a nodegraph node
    void buildUiNodeGraph(const mx::NodeGraphPtr& nodeGraphs);

    void setUiNodeInfo(
        UiNodePtr node,
        const std::string& type,
        const std::string& category);

    int findNode(int nodeId);

    int findNode(const std::string& name, const std::string& type);

    void addNode(
        const std::string& category,
        const std::string& name,
        const std::string& type);

    void addLink(SocketID startPinId, SocketID endPinId);

    void removeEdge(int downNode, int upNode, UiPinPtr pin);

    void deleteLink(LinkId deletedLinkId);

    void deleteNode(UiNodePtr node);

    bool edgeExists(UiEdge newEdge);

    void addNodeGraphPins();

    mx::ElementPredicate getElementPredicate() const;

    ~MaterialXNodeTree() override;
    Node* find_node(NodeId id) const override;
    Node* find_node(const char* identifier) const override;
    Node* add_node(const char* str) override;
    void delete_node(Node* nodeId, bool allow_repeat_delete) override;
    void delete_node(NodeId nodeId, bool allow_repeat_delete) override;
    void delete_link(
        LinkId linkId,
        bool refresh_topology,
        bool remove_from_group) override;
    void delete_link(
        NodeLink* link,
        bool refresh_topology,
        bool remove_from_group) override;
    mx::DocumentPtr get_mtlx_stdlib();

    // document and initializing information
    mx::FilePath _materialFilename;
    mx::DocumentPtr _graphDoc;
    mx::StringSet _xincludeFiles;

    mx::FileSearchPath _searchPath;
    mx::FilePathVec _libraryFolders;
    mx::DocumentPtr _stdLib;
};

std::shared_ptr<MaterialXNodeTree> createMaterialXNodeTree(
    const std::string& materialFilename)
{
    std::shared_ptr<NodeTreeDescriptor> descriptor =
        std::make_shared<MaterialXNodeTreeDescriptor>();
    return std::make_shared<MaterialXNodeTree>(materialFilename, descriptor);
}

USTC_CG_NAMESPACE_CLOSE_SCOPE
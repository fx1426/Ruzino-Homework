#pragma once

#include <MCore/MaterialXNodeTreeWidget.h>
#include <MaterialXFormat/Util.h>
#include <spdlog/fwd.h>

namespace mx = MaterialX;

RUZINO_NAMESPACE_OPEN_SCOPE
class MCORE_API MaterialXNodeTreeDescriptor : public NodeTreeDescriptor {
   public:
    NodeTypeInfo* get_node_type(const std::string& name) override
    {
        auto it = _nodeTypes.find(name);
        if (it != _nodeTypes.end()) {
            return it->second.get();
        }
        else {
            _nodeTypes[name] = std::make_unique<NodeTypeInfo>(name.c_str());
            return _nodeTypes[name].get();
        }
    }

   private:
    mutable std::map<std::string, std::unique_ptr<NodeTypeInfo>> _nodeTypes;
};

class MCORE_API MaterialXNodeTree : public NodeTree {
   public:
    explicit MaterialXNodeTree(
        const std::string& materialFilename,
        const std::shared_ptr<NodeTreeDescriptor>& descriptor)
        : _materialFilename(materialFilename),
          NodeTree(descriptor)
    {
        _searchPath = mx::getDefaultDataSearchPath();
        _libraryFolders = { "libraries" };

        loadStandardLibraries();
        _graphDoc = loadDocument(materialFilename);

        if (_graphDoc) {
            buildUiBaseGraph(_graphDoc);
            _currGraphElem = _graphDoc;
        }
    }

    // Constructor for creating a new empty MaterialX document
    MaterialXNodeTree(
        const std::shared_ptr<NodeTreeDescriptor>& descriptor,
        mx::DocumentPtr doc);

    // Delete copy constructor and assignment operator to prevent issues with
    // MaterialX pointers
    MaterialXNodeTree(const MaterialXNodeTree&) = delete;
    MaterialXNodeTree& operator=(const MaterialXNodeTree&) = delete;

    void loadStandardLibraries();

    mx::DocumentPtr loadDocument(const mx::FilePath& filename);

    // Build UiNode nodegraph upon loading a document
    void buildUiBaseGraph(mx::DocumentPtr doc);

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

    SocketID getOutputPin(UiNodePtr node, UiNodePtr upNode, UiPinPtr input);

    NodeLink* add_link(
        SocketID startPinId,
        SocketID endPinId,
        bool refresh_topology = true) override;

    void delete_link(
        LinkId linkid,
        bool refresh_topology = true,
        bool remove_from_group = true) override;

    mx::ElementPredicate getElementPredicate() const;

    ~MaterialXNodeTree() override;
    Node* find_node(NodeId id) const override;
    Node* find_node(const char* identifier) const override;
    Node* add_node(const char* str) override;
    void delete_node(NodeId nodeId, bool allow_repeat_delete) override;

    mx::DocumentPtr get_mtlx_stdlib();

    // MaterialX Edition

    // Add input pointer to node based on input pin
    void addNodeInput(UiNodePtr node, mx::InputPtr& input);
    void setDefaults(mx::InputPtr input);

    void deleteLinkInfo(NodeSocket* from_sock, NodeSocket* to_sock);
    // document and initializing information
    mx::FilePath _materialFilename;
    mx::DocumentPtr _graphDoc;
    mx::StringSet _xincludeFiles;

    mx::FileSearchPath _searchPath;
    mx::FilePathVec _libraryFolders;
    mx::DocumentPtr _stdLib;

    SocketType get_unique_socket_type(const char* name);
    unsigned int _uniqueSocketType = 0;

    mx::GraphElementPtr _currGraphElem;

    void saveDocument(mx::FilePath filePath);
};

MCORE_API std::shared_ptr<MaterialXNodeTree> createMaterialXNodeTree(
    const std::string& materialFilename);

// Create a more user-friendly node definition name
MCORE_API std::string getUserNodeDefName(const std::string& val);

MCORE_API mx::NodePtr getMaterialXNode(Node* node);
MCORE_API mx::NodeGraphPtr getMaterialXNodeGraph(Node* node);
MCORE_API mx::InputPtr getMaterialXInput(Node* node);
MCORE_API mx::OutputPtr getMaterialXOutput(Node* node);
MCORE_API mx::InputPtr getMaterialXPinInput(NodeSocket* socket);
MCORE_API mx::OutputPtr getMaterialXPinOutput(NodeSocket* socket);

RUZINO_NAMESPACE_CLOSE_SCOPE
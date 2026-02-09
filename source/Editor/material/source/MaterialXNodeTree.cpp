//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//
#define IMGUI_DEFINE_MATH_OPERATORS

#include "MCore/MaterialXNodeTree.hpp"

#include <spdlog/spdlog.h>

#include <iostream>

#include "foo_socket_types.inl"

RUZINO_NAMESPACE_OPEN_SCOPE

void MaterialXNodeTree::saveDocument(mx::FilePath filePath)
{
    spdlog::info(
        "[MaterialXNodeTree::saveDocument] Called with filePath = {}",
        filePath.asString());
    spdlog::info(
        "[MaterialXNodeTree::saveDocument] _graphDoc = {}",
        (void*)_graphDoc.get());

    if (filePath.getExtension() != mx::MTLX_EXTENSION) {
        filePath.addExtension(mx::MTLX_EXTENSION);
    }

    mx::DocumentPtr writeDoc = _graphDoc;
    spdlog::info(
        "[MaterialXNodeTree::saveDocument] writeDoc = {}",
        (void*)writeDoc.get());

    mx::XmlWriteOptions writeOptions;
    writeOptions.elementPredicate = getElementPredicate();
    mx::writeToXmlFile(writeDoc, filePath, &writeOptions);
}

std::shared_ptr<MaterialXNodeTree> createMaterialXNodeTree(
    const std::string& materialFilename)
{
    std::shared_ptr<NodeTreeDescriptor> descriptor =
        std::make_shared<MaterialXNodeTreeDescriptor>();
    return std::make_shared<MaterialXNodeTree>(materialFilename, descriptor);
}

class MaterialXSocketDeclaration : public SocketDeclaration {
   public:
    NodeSocket* build(NodeTree* ntree, Node* node) const override;
};

mx::NodePtr getMaterialXNode(Node* node)
{
    auto cast = node->storage.try_cast<mx::NodePtr>();
    if (cast) {
        return *cast;
    }
    return nullptr;
}

mx::NodeGraphPtr getMaterialXNodeGraph(Node* node)
{
    auto cast = node->storage.try_cast<mx::NodeGraphPtr>();
    if (cast) {
        return *cast;
    }
    return nullptr;
}

mx::InputPtr getMaterialXInput(Node* node)
{
    auto cast = node->storage.try_cast<mx::InputPtr>();
    if (cast) {
        return *cast;
    }
    return nullptr;
}

mx::OutputPtr getMaterialXOutput(Node* node)
{
    auto cast = node->storage.try_cast<mx::OutputPtr>();
    if (cast) {
        return *cast;
    }
    return nullptr;
}

mx::InputPtr getMaterialXPinInput(NodeSocket* socket)
{
    auto cast = socket->storage.try_cast<mx::InputPtr>();
    if (cast) {
        return *cast;
    }
    return nullptr;
}

mx::OutputPtr getMaterialXPinOutput(NodeSocket* socket)
{
    auto cast = socket->storage.try_cast<mx::OutputPtr>();
    if (cast) {
        return *cast;
    }
    return nullptr;
}

NodeSocket* MaterialXSocketDeclaration::build(NodeTree* ntree, Node* node) const
{
    NodeSocket* socket = node->add_socket(
        type.info().name().data(),
        this->identifier.c_str(),
        this->name.c_str(),
        this->in_out);
    socket->type_info = type;
    update_default_value(socket);

    return socket;
}

MaterialXNodeTree::MaterialXNodeTree(
    const std::shared_ptr<NodeTreeDescriptor>& descriptor,
    mx::DocumentPtr doc)
    : NodeTree(descriptor)
{
    _searchPath = mx::getDefaultDataSearchPath();
    _libraryFolders = { "libraries" };

    loadStandardLibraries();

    spdlog::info(
        "[MaterialXNodeTree] Constructor called with doc = {}",
        (void*)doc.get());

    if (doc) {
        _graphDoc = doc;
        spdlog::info(
            "[MaterialXNodeTree] Assigned _graphDoc = {}",
            (void*)_graphDoc.get());
        _graphDoc->importLibrary(_stdLib);
        spdlog::info(
            "[MaterialXNodeTree] After importLibrary, _graphDoc = {}",
            (void*)_graphDoc.get());
    }
    else {
        _graphDoc = mx::createDocument();
        _graphDoc->importLibrary(_stdLib);
    }

    _materialFilename = "";
    buildUiBaseGraph(_graphDoc);
    spdlog::info(
        "[MaterialXNodeTree] After buildUiBaseGraph, _graphDoc = {}",
        (void*)_graphDoc.get());
    _currGraphElem = _graphDoc;
    spdlog::info(
        "[MaterialXNodeTree] Constructor finished, _graphDoc = {}",
        (void*)_graphDoc.get());
}

void MaterialXNodeTree::loadStandardLibraries()
{
    // Initialize the standard library.
    try {
        _stdLib = mx::createDocument();
        _xincludeFiles =
            mx::loadLibraries(_libraryFolders, _searchPath, _stdLib);
        if (_xincludeFiles.empty()) {
            std::cerr << "Could not find standard data libraries on the given "
                         "search path: "
                      << _searchPath.asString() << std::endl;
        }
    }
    catch (std::exception& e) {
        std::cerr << "Failed to load standard data libraries: " << e.what()
                  << std::endl;
        return;
    }
}

mx::DocumentPtr MaterialXNodeTree::loadDocument(const mx::FilePath& filename)
{
    mx::FilePathVec libraryFolders = { "libraries" };
    _libraryFolders = libraryFolders;
    mx::XmlReadOptions readOptions;
    readOptions.readXIncludeFunction = [](mx::DocumentPtr doc,
                                          const mx::FilePath& filename,
                                          const mx::FileSearchPath& searchPath,
                                          const mx::XmlReadOptions* options) {
        mx::FilePath resolvedFilename = searchPath.find(filename);
        if (resolvedFilename.exists()) {
            try {
                readFromXmlFile(doc, resolvedFilename, searchPath, options);
            }
            catch (mx::Exception& e) {
                std::cerr << "Failed to read include file: "
                          << filename.asString() << ". "
                          << std::string(e.what()) << std::endl;
            }
        }
        else {
            std::cerr << "Include file not found: " << filename.asString()
                      << std::endl;
        }
    };

    mx::DocumentPtr doc = mx::createDocument();
    try {
        if (!filename.isEmpty()) {
            mx::readFromXmlFile(doc, filename, _searchPath, &readOptions);
            doc->importLibrary(_stdLib);
            std::string message;
            if (!doc->validate(&message)) {
                std::cerr << "*** Validation warnings for "
                          << filename.asString() << " ***" << std::endl;
                std::cerr << message << std::endl;
            }

            // Cache the currently loaded file
            _materialFilename = filename;
        }
    }
    catch (mx::Exception& e) {
        std::cerr << "Failed to read file: " << filename.asString() << ": \""
                  << std::string(e.what()) << "\"" << std::endl;
    }
    //_graphStack = std::stack<std::vector<UiNodePtr>>();
    //_pinStack = std::stack<std::vector<UiPinPtr>>();
    return doc;
}

void MaterialXNodeTree::buildUiBaseGraph(mx::DocumentPtr doc)
{
    std::vector<mx::NodeGraphPtr> nodeGraphs = doc->getNodeGraphs();
    std::vector<mx::InputPtr> inputNodes = doc->getActiveInputs();
    std::vector<mx::OutputPtr> outputNodes = doc->getOutputs();
    std::vector<mx::NodePtr> docNodes = doc->getNodes();

    mx::ElementPredicate includeElement = getElementPredicate();

    this->clear();
    // nodes.clear();
    // links.clear();
    //_currEdge.clear();
    // sockets.clear();
    //_graphTotalSize = 1;

    // Create UiNodes for nodes that belong to the document so they are not in a
    // nodegraph
    for (mx::NodePtr node : docNodes) {
        if (!includeElement(node))
            continue;
        std::string getType = node->getType();

        auto currNode = add_node(getType.c_str());
        currNode->ui_name = node->getName();

        // auto currNode = std::make_shared<UiNode>(name, _graphTotalSize);
        currNode->storage = node;
        setUiNodeInfo(currNode, node->getType(), node->getCategory());
    }

    // Create UiNodes for the nodegraph
    for (mx::NodeGraphPtr nodeGraph : nodeGraphs) {
        if (!includeElement(nodeGraph))
            continue;
        auto currNode = add_node("groupnode");
        currNode->ui_name = nodeGraph->getName();
        currNode->storage = nodeGraph;
        setUiNodeInfo(currNode, nodeGraph->getType(), nodeGraph->getCategory());
    }
    for (mx::InputPtr input : inputNodes) {
        if (!includeElement(input))
            continue;
        auto currNode = add_node(input->getType().c_str());
        currNode->ui_name = input->getName();
        currNode->storage = input;
        setUiNodeInfo(currNode, input->getType(), input->getCategory());
    }
    for (mx::OutputPtr output : outputNodes) {
        if (!includeElement(output))
            continue;
        auto currNode = add_node(output->getType().c_str());
        currNode->ui_name = output->getName();
        currNode->storage = output;
        setUiNodeInfo(currNode, output->getType(), output->getCategory());
    }

    // Create edges for nodegraphs
    for (mx::NodeGraphPtr graph : nodeGraphs) {
        for (mx::InputPtr input : graph->getActiveInputs()) {
            int downNum = -1;
            int upNum = -1;
            mx::string nodeGraphName = input->getNodeGraphString();
            mx::NodePtr connectedNode = input->getConnectedNode();
            if (!nodeGraphName.empty()) {
                downNum = findNode(graph->getName(), "nodegraph");
                upNum = findNode(nodeGraphName, "nodegraph");
            }
            else if (connectedNode) {
                downNum = findNode(graph->getName(), "nodegraph");
                upNum = findNode(connectedNode->getName(), "node");
            }

            if (downNum == -1 || upNum == -1)
                continue;

            auto& up_node = nodes[upNum];
            auto& down_node = nodes[downNum];

            auto input_socket =
                down_node->get_input_socket(input->getName().c_str());

            auto output =
                getOutputPin(down_node.get(), up_node.get(), input_socket);

            add_link(output, input_socket->ID);
        }
    }

    // Create edges for surface and material nodes
    for (mx::NodePtr node : docNodes) {
        mx::NodeDefPtr nD = node->getNodeDef(node->getName());
        for (mx::InputPtr input : node->getActiveInputs()) {
            mx::string nodeGraphName = input->getNodeGraphString();
            mx::NodePtr connectedNode = input->getConnectedNode();
            mx::OutputPtr connectedOutput = input->getConnectedOutput();
            int upNum = -1;
            int downNum = -1;
            if (!nodeGraphName.empty()) {
                upNum = findNode(nodeGraphName, "nodegraph");
                downNum = findNode(node->getName(), "node");
            }
            else if (connectedNode) {
                upNum = findNode(connectedNode->getName(), "node");
                downNum = findNode(node->getName(), "node");
            }
            else if (connectedOutput) {
                upNum = findNode(connectedOutput->getName(), "output");
                downNum = findNode(node->getName(), "node");
            }
            else if (!input->getInterfaceName().empty()) {
                upNum = findNode(input->getInterfaceName(), "input");
                downNum = findNode(node->getName(), "node");
            }

            if (downNum == -1 || upNum == -1)
                continue;

            auto& up_node = nodes[upNum];
            auto& down_node = nodes[downNum];

            auto input_socket =
                down_node->get_input_socket(input->getName().c_str());

            auto output =
                getOutputPin(down_node.get(), up_node.get(), input_socket);

            NodeTree::add_link(output, input_socket->ID);
        }
    }
}

void MaterialXNodeTree::setUiNodeInfo(
    UiNodePtr node,
    const std::string& type,
    const std::string& category)
{
    // Always set up sockets for MaterialX nodes since each instance may have
    // different inputs/outputs Clear existing declarations to ensure fresh
    // setup for this specific node instance
    node->typeinfo->static_declaration.items.clear();
    node->typeinfo->static_declaration.inputs.clear();
    node->typeinfo->static_declaration.outputs.clear();

    // Lambda to create and add a socket declaration.
    auto createSocketDeclaration = [&](const std::string& socketName,
                                       const std::string& socketType,
                                       PinKind kind,
                                       auto& targetVector) {
        auto socketDecl = std::make_shared<MaterialXSocketDeclaration>();

        socketDecl->name = socketName;
        socketDecl->identifier = socketName;
        // Setup type using socket category.

        socketDecl->type = get_unique_socket_type(socketType.c_str());
        socketDecl->in_out = kind;

        node->typeinfo->static_declaration.items.push_back(socketDecl);
        targetVector.push_back(socketDecl.get());
    };

    // If the node stores a NodeGraph.
    if (node->storage.allow_cast<mx::NodeGraphPtr>()) {
        auto nodeGraph = node->storage.cast<mx::NodeGraphPtr>();

        for (mx::OutputPtr out : nodeGraph->getOutputs())
            createSocketDeclaration(
                out->getName(),
                out->getType(),
                PinKind::Output,
                node->typeinfo->static_declaration.outputs);
        for (mx::InputPtr in : nodeGraph->getInputs())
            createSocketDeclaration(
                in->getName(),
                in->getType(),
                PinKind::Input,
                node->typeinfo->static_declaration.inputs);
    }
    // If the node stores a regular Node.
    else if (node->storage.allow_cast<mx::NodePtr>()) {
        auto mnode = node->storage.cast<mx::NodePtr>();
        mx::NodeDefPtr nodeDef = mnode->getNodeDef(mnode->getName());
        if (nodeDef) {
            for (mx::InputPtr in : nodeDef->getActiveInputs())
                createSocketDeclaration(
                    in->getName(),
                    in->getType(),
                    PinKind::Input,
                    node->typeinfo->static_declaration.inputs);
            for (mx::OutputPtr out : nodeDef->getActiveOutputs())
                createSocketDeclaration(
                    out->getName(),
                    out->getType(),
                    PinKind::Output,
                    node->typeinfo->static_declaration.outputs);
        }
    }
    // If the node stores an Input (provides output socket).
    else if (node->storage.allow_cast<mx::InputPtr>()) {
        auto input = node->storage.cast<mx::InputPtr>();
        createSocketDeclaration(
            input->getName(),
            input->getType(),
            PinKind::Output,
            node->typeinfo->static_declaration.outputs);
    }
    // If the node stores an Output (accepts input socket).
    else if (node->storage.allow_cast<mx::OutputPtr>()) {
        auto output = node->storage.cast<mx::OutputPtr>();
        createSocketDeclaration(
            output->getName(),
            output->getType(),
            PinKind::Input,
            node->typeinfo->static_declaration.inputs);
    }

    node->refresh_node();

    // If the node stores a NodeGraph.
    if (node->storage.allow_cast<mx::NodeGraphPtr>()) {
        auto nodeGraph = node->storage.cast<mx::NodeGraphPtr>();

        for (int i = 0; i < nodeGraph->getOutputs().size(); i++) {
            auto out = nodeGraph->getOutputs()[i];
            auto socket = node->get_outputs()[i];
            socket->storage = out;
        }

        for (int i = 0; i < nodeGraph->getInputs().size(); i++) {
            auto in = nodeGraph->getInputs()[i];
            auto socket = node->get_input_socket(in->getName().c_str());
            if (socket)
                socket->storage = in;
        }
    }
    // If the node stores a regular Node.
    else if (node->storage.allow_cast<mx::NodePtr>()) {
        auto mnode = node->storage.cast<mx::NodePtr>();
        mx::NodeDefPtr nodeDef = mnode->getNodeDef(mnode->getName());
        if (nodeDef) {
            // Store inputs
            for (auto& in : nodeDef->getActiveInputs()) {
                auto socket = node->get_input_socket(in->getName().c_str());
                if (socket) {
                    // For node inputs, store the actual input from the node,
                    // not the nodedef
                    auto nodeInput = mnode->getInput(in->getName());
                    if (nodeInput)
                        socket->storage = nodeInput;
                    else
                        socket->storage = in;
                }
            }

            // Store outputs
            for (int i = 0; i < nodeDef->getActiveOutputs().size(); i++) {
                auto out = nodeDef->getActiveOutputs()[i];
                auto nodeOutput = mnode->getOutput(out->getName());
                auto socket = node->get_outputs()[i];
                if (socket) {
                    if (nodeOutput)
                        socket->storage = nodeOutput;
                    else
                        socket->storage = out;
                }
            }
        }
    }
    // If the node stores an Input (has output socket).
    else if (node->storage.allow_cast<mx::InputPtr>()) {
        auto input = node->storage.cast<mx::InputPtr>();
        if (!node->get_outputs().empty()) {
            node->get_outputs()[0]->storage = input;
        }
    }
    // If the node stores an Output (has input socket).
    else if (node->storage.allow_cast<mx::OutputPtr>()) {
        auto output = node->storage.cast<mx::OutputPtr>();
        if (!node->get_inputs().empty()) {
            node->get_inputs()[0]->storage = output;
        }
    }
}

int MaterialXNodeTree::findNode(int nodeId)
{
    int count = 0;
    for (size_t i = 0; i < nodes.size(); i++) {
        if (nodes[i]->ID == NodeId(nodeId)) {
            return count;
        }
        count++;
    }
    return -1;
}

int MaterialXNodeTree::findNode(
    const std::string& name,
    const std::string& type)
{
    int count = 0;
    for (size_t i = 0; i < nodes.size(); i++) {
        if (nodes[i]->getName() == name) {
            if (type == "node" && getMaterialXNode(nodes[i].get()) != nullptr) {
                return count;
            }
            else if (
                type == "input" &&
                getMaterialXInput(nodes[i].get()) != nullptr) {
                return count;
            }
            else if (
                type == "output" &&
                getMaterialXOutput(nodes[i].get()) != nullptr) {
                return count;
            }
            else if (
                type == "nodegraph" &&
                getMaterialXNodeGraph(nodes[i].get()) != nullptr) {
                return count;
            }
        }
        count++;
    }
    return -1;
}

// Create a more user-friendly node definition name
std::string getUserNodeDefName(const std::string& val)
{
    const std::string ND_PREFIX = "ND_";
    std::string result = val;
    if (mx::stringStartsWith(val, ND_PREFIX)) {
        result = val.substr(3, val.length());
    }
    return result;
}

void MaterialXNodeTree::addNode(
    const std::string& category,
    const std::string& name,
    const std::string& type)
{
    mx::NodePtr node = nullptr;
    std::vector<mx::NodeDefPtr> matchingNodeDefs;

    // Create document or node graph is there is not already one
    if (category == "output") {
        std::string outName = "";
        mx::OutputPtr newOut;
        // add output as child of correct parent and create valid name
        outName = _currGraphElem->createValidChildName(name);
        newOut = _currGraphElem->addOutput(outName, type);

        auto outputNode = add_node(type.c_str());
        outputNode->ui_name = outName;
        outputNode->storage = newOut;

        setUiNodeInfo(outputNode, type, category);
        return;
    }
    else if (category == "input") {
        std::string inName = "";
        mx::InputPtr newIn = nullptr;

        // Add input as child of correct parent and create valid name
        inName = _currGraphElem->createValidChildName(name);
        newIn = _currGraphElem->addInput(inName, type);
        auto inputNode = add_node(type.c_str());
        inputNode->ui_name = inName;
        inputNode->storage = newIn;

        setDefaults(newIn);
        setUiNodeInfo(inputNode, type, category);
        return;
    }
    else if (category == "group") {
        throw std::runtime_error("Not implemented");

        auto groupNode = add_node("groupnode");
        groupNode->ui_name = name;
        // Create new mx::NodeGraph and set as current node graph
        _graphDoc->addNodeGraph();
        std::string nodeGraphName =
            _graphDoc->getNodeGraphs().back()->getName();

        // Set message of group UiNode in order to identify it as such
        // groupNode->setMessage("Comment");
        setUiNodeInfo(groupNode, type, "group");

        // Create ui portions of group node
        // buildGroupNode(nodes.back());
        return;
    }
    else if (category == "nodegraph") {
        // Create new mx::NodeGraph and set as current node graph
        _graphDoc->addNodeGraph();
        std::string nodeGraphName =
            _graphDoc->getNodeGraphs().back()->getName();
        auto nodeGraphNode = add_node("groupnode");
        // Set mx::Nodegraph as node graph for uiNode
        nodeGraphNode->storage = _graphDoc->getNodeGraphs().back();
        nodeGraphNode->ui_name = nodeGraphName;

        setUiNodeInfo(nodeGraphNode, type, "nodegraph");
        nodeGraphNode->refresh_node();
        return;
    }
    else {
        matchingNodeDefs = _graphDoc->getMatchingNodeDefs(category);
        for (mx::NodeDefPtr nodedef : matchingNodeDefs) {
            std::string userNodeDefName =
                getUserNodeDefName(nodedef->getName());
            if (userNodeDefName == name) {
                node = _currGraphElem->addNodeInstance(
                    nodedef, _currGraphElem->createValidChildName(name));
            }
        }
    }

    if (node) {
        int num = 0;
        int countDef = 0;
        for (size_t i = 0; i < matchingNodeDefs.size(); i++) {
            std::string userNodeDefName =
                getUserNodeDefName(matchingNodeDefs[i]->getName());
            if (userNodeDefName == name) {
                num = countDef;
            }
            countDef++;
        }
        std::vector<mx::InputPtr> defInputs =
            matchingNodeDefs[num]->getActiveInputs();

        // Add inputs to UiNode as pins so that we can later add them to the
        // node if necessary
        auto newNode = add_node(type.c_str());
        // std::make_shared<UiNode>(node->getName(), int(++_graphTotalSize));
        newNode->storage = node;
        newNode->ui_name = node->getName();

        node->setType(type);

        setUiNodeInfo(newNode, type, category);

        // Note: refresh_node() is already called inside setUiNodeInfo(), no
        // need to call it again
    }
}

SocketID MaterialXNodeTree::getOutputPin(
    UiNodePtr node,
    UiNodePtr upNode,
    UiPinPtr input)
{
    if (getMaterialXNodeGraph(upNode) != nullptr) {
        // For nodegraph need to get the correct output pin according to the
        // names of the output nodes
        mx::OutputPtr output;
        if (getMaterialXNode(input->node)) {
            output = getMaterialXNode(input->node)
                         ->getConnectedOutput(input->identifier);
        }
        else if (getMaterialXNodeGraph(input->node)) {
            output = getMaterialXNodeGraph(input->node)
                         ->getConnectedOutput(input->identifier);
        }

        if (output) {
            std::string outName = output->getName();
            for (UiPinPtr outputs : upNode->get_outputs()) {
                if (outputs->identifier == outName) {
                    return outputs->ID;
                }
            }
        }
        return SocketID();
    }
    else {
        // For node need to get the correct output pin based on the output
        // attribute

        // throw std::runtime_error("Not implemented");
        if (!upNode->get_outputs().empty()) {
            std::string outputName = mx::EMPTY_STRING;
            if (getMaterialXPinInput(input)) {
                outputName = getMaterialXPinInput(input)->getOutputString();
            }
            else if (getMaterialXPinOutput(input)) {
                outputName = getMaterialXPinOutput(input)->getOutputString();
            }

            size_t pinIndex = 0;
            if (!outputName.empty()) {
                for (size_t i = 0; i < upNode->get_outputs().size(); i++) {
                    if (upNode->get_outputs()[i]->identifier == outputName) {
                        pinIndex = i;
                        break;
                    }
                }
            }
            return (upNode->get_outputs()[pinIndex]->ID);
        }
        return SocketID();
    }
}

NodeLink* MaterialXNodeTree::add_link(
    SocketID startPinId,
    SocketID endPinId,
    bool refresh_topology)
{
    // First, find the sockets to check if we need to remove an existing link
    auto startPin = find_pin(startPinId);
    auto endPin = find_pin(endPinId);

    if (!startPin || !endPin) {
        return nullptr;
    }

    // Determine which is input and which is output
    auto from_sock = (startPin->in_out == PinKind::Output) ? startPin : endPin;
    auto to_sock = (startPin->in_out == PinKind::Input) ? startPin : endPin;

    // If the accepting node already has a link, remove it BEFORE adding new
    // link
    if (to_sock->directly_linked_links.size() > 0) {
        assert(to_sock->directly_linked_links.size() == 1);
        delete_link(
            to_sock->directly_linked_links[0]->ID,
            false);  // Don't refresh topology yet
    }

    // Now add the new link
    auto link = NodeTree::add_link(
        startPinId, endPinId, false);  // Don't refresh topology yet

    // Check if link was successfully created
    if (!link) {
        return nullptr;
    }

    auto inputPinId = to_sock->ID;
    auto outputPinId = from_sock->ID;

    auto upNode = from_sock->node;
    auto downNode = to_sock->node;

    auto uiUpNode = upNode;
    auto uiDownNode = downNode;

    {
        if (getMaterialXNode(uiDownNode) || getMaterialXNodeGraph(uiDownNode)) {
            mx::InputPtr connectingInput = nullptr;
            for (UiPinPtr pin : uiDownNode->get_inputs()) {
                if (pin->ID == inputPinId) {
                    addNodeInput(
                        uiDownNode, pin->storage.cast<mx::InputPtr&>());

                    // Update value to be empty
                    if (getMaterialXNode(uiDownNode) &&
                        getMaterialXNode(uiDownNode)->getType() ==
                            mx::SURFACE_SHADER_TYPE_STRING) {
                        if (getMaterialXOutput(uiUpNode) != nullptr) {
                            getMaterialXPinInput(pin)->setConnectedOutput(
                                getMaterialXOutput(uiUpNode));
                        }
                        else if (getMaterialXInput(uiUpNode) != nullptr) {
                            getMaterialXPinInput(pin)->setInterfaceName(
                                uiUpNode->getName());
                        }
                        else {
                            if (getMaterialXNodeGraph(uiUpNode) != nullptr) {
                                for (UiPinPtr outPin :
                                     uiUpNode->get_outputs()) {
                                    // Set pin connection to correct output
                                    if (outPin->ID == outputPinId) {
                                        mx::OutputPtr outputs =
                                            getMaterialXNodeGraph(uiUpNode)
                                                ->getOutput(outPin->identifier);
                                        getMaterialXPinInput(pin)
                                            ->setConnectedOutput(outputs);
                                    }
                                }
                            }
                            else {
                                getMaterialXPinInput(pin)->setConnectedNode(
                                    getMaterialXNode(uiUpNode));
                            }
                        }
                    }
                    else {
                        if (getMaterialXInput(uiUpNode)) {
                            getMaterialXPinInput(pin)->setInterfaceName(
                                uiUpNode->getName());
                        }
                        else {
                            if (getMaterialXNode(uiUpNode)) {
                                mx::NodePtr upstreamNode =
                                    getMaterialXNode(upNode);
                                mx::NodeDefPtr upstreamNodeDef =
                                    upstreamNode->getNodeDef();
                                bool isMultiOutput =
                                    upstreamNodeDef
                                        ? upstreamNodeDef->getOutputs().size() >
                                              1
                                        : false;
                                if (!isMultiOutput) {
                                    getMaterialXPinInput(pin)->setConnectedNode(
                                        getMaterialXNode(uiUpNode));
                                }
                                else {
                                    for (UiPinPtr outPin :
                                         upNode->get_outputs()) {
                                        // Set pin connection to correct
                                        // output
                                        if (outPin->ID == outputPinId) {
                                            mx::OutputPtr outputs =
                                                getMaterialXNode(uiUpNode)
                                                    ->getOutput(
                                                        outPin->identifier);
                                            if (!outputs) {
                                                outputs =
                                                    getMaterialXNode(uiUpNode)
                                                        ->addOutput(
                                                            outPin->identifier,
                                                            getMaterialXPinInput(
                                                                pin)
                                                                ->getType());
                                            }
                                            getMaterialXPinInput(pin)
                                                ->setConnectedOutput(outputs);
                                        }
                                    }
                                }
                            }
                            else if (getMaterialXNodeGraph(uiUpNode)) {
                                for (UiPinPtr outPin :
                                     uiUpNode->get_outputs()) {
                                    // Set pin connection to correct output
                                    if (outPin->ID == outputPinId) {
                                        mx::OutputPtr outputs =
                                            getMaterialXNodeGraph(uiUpNode)
                                                ->getOutput(outPin->identifier);
                                        getMaterialXPinInput(pin)
                                            ->setConnectedOutput(outputs);
                                    }
                                }
                            }
                        }
                    }

                    connectingInput = getMaterialXPinInput(pin);
                    break;
                }
            }
        }
        else if (getMaterialXOutput(downNode) != nullptr) {
            mx::InputPtr connectingInput = nullptr;
            getMaterialXOutput(downNode)->setConnectedNode(
                getMaterialXNode(upNode));
        }
    }

    SetDirty();

    // Refresh topology if requested
    if (refresh_topology) {
        ensure_topology_cache();
    }

    return link;
}

mx::ElementPredicate MaterialXNodeTree::getElementPredicate() const
{
    return [this](mx::ConstElementPtr elem) {
        if (elem->hasSourceUri()) {
            return (_xincludeFiles.count(elem->getSourceUri()) == 0);
        }
        return true;
    };
}

MaterialXNodeTree::~MaterialXNodeTree()
{
}

Node* MaterialXNodeTree::find_node(NodeId id) const
{
    return NodeTree::find_node(id);
}

Node* MaterialXNodeTree::find_node(const char* identifier) const
{
    return NodeTree::find_node(identifier);
}

Node* MaterialXNodeTree::add_node(const char* str)
{
    return NodeTree::add_node(str);
}

void MaterialXNodeTree::delete_node(NodeId nodeId, bool allow_repeat_delete)
{
    // Delete link

    auto node = find_node(nodeId);

    for (UiPinPtr outputPin : node->get_outputs()) {
        // Update downNode info
        for (UiPinPtr pin : outputPin->directly_linked_sockets) {
            mx::ValuePtr val;
            if (getMaterialXNode(pin->node)) {
                mx::NodeDefPtr nodeDef =
                    getMaterialXNode(pin->node)->getNodeDef(
                        getMaterialXNode(pin->node)->getName());
                val = nodeDef
                          ->getActiveInput(getMaterialXPinInput(pin)->getName())
                          ->getValue();
                if (getMaterialXNode(pin->node)->getType() ==
                    mx::SURFACE_SHADER_TYPE_STRING) {
                    getMaterialXPinInput(pin)->setConnectedOutput(nullptr);
                }
                else {
                    getMaterialXPinInput(pin)->setConnectedNode(nullptr);
                }
                if (getMaterialXInput(node)) {
                    // Remove interface in order to set the default of the input
                    getMaterialXPinInput(pin)->setInterfaceName(
                        mx::EMPTY_STRING);
                    setDefaults(getMaterialXPinInput(pin));
                    setDefaults(getMaterialXInput(node));
                }
            }
            else if (getMaterialXNodeGraph(pin->node)) {
                if (getMaterialXInput(node)) {
                    getMaterialXNodeGraph(pin->node)
                        ->getInput(pin->identifier)
                        ->setInterfaceName(mx::EMPTY_STRING);
                    setDefaults(getMaterialXInput(node));
                }
                getMaterialXPinInput(pin)->setConnectedNode(nullptr);
                setDefaults(getMaterialXPinInput(pin));
            }

            if (val) {
                getMaterialXPinInput(pin)->setValueString(
                    val->getValueString());
            }
        }
    }

    // Remove from NodeGraph
    // All link information is handled in delete link which is called    before
    // this
    _currGraphElem->removeChild(node->getName());

    NodeTree::delete_node(nodeId, allow_repeat_delete);
}

void MaterialXNodeTree::delete_link(
    LinkId linkid,
    bool refresh_topology,
    bool remove_from_group)
{
    auto link = find_link(linkid);

    if (link)
        deleteLinkInfo(link->from_sock, link->to_sock);
    NodeTree::delete_link(linkid, refresh_topology, remove_from_group);
}

mx::DocumentPtr MaterialXNodeTree::get_mtlx_stdlib()
{
    return _stdLib;
}

void MaterialXNodeTree::addNodeInput(UiNodePtr node, mx::InputPtr& input)
{
    if (getMaterialXNode(node)) {
        if (!getMaterialXNode(node)->getInput(input->getName())) {
            input = getMaterialXNode(node)->addInput(
                input->getName(), input->getType());
            input->setConnectedNode(nullptr);
        }
    }
}

void MaterialXNodeTree::setDefaults(mx::InputPtr input)
{
    if (input->getType() == "float") {
        input->setValue(0.f, "float");
    }
    else if (input->getType() == "integer") {
        input->setValue(0, "integer");
    }
    else if (input->getType() == "color3") {
        input->setValue(mx::Color3(0.f, 0.f, 0.f), "color3");
    }
    else if (input->getType() == "color4") {
        input->setValue(mx::Color4(0.f, 0.f, 0.f, 1.f), "color4");
    }
    else if (input->getType() == "vector2") {
        input->setValue(mx::Vector2(0.f, 0.f), "vector2");
    }
    else if (input->getType() == "vector3") {
        input->setValue(mx::Vector3(0.f, 0.f, 0.f), "vector3");
    }
    else if (input->getType() == "vector4") {
        input->setValue(mx::Vector4(0.f, 0.f, 0.f, 0.f), "vector4");
    }
    else if (input->getType() == "string") {
        input->setValue("", "string");
    }
    else if (input->getType() == "filename") {
        input->setValue("", "filename");
    }
    else if (input->getType() == "boolean") {
        input->setValue(false, "boolean");
    }
}

void MaterialXNodeTree::deleteLinkInfo(
    NodeSocket* from_sock,
    NodeSocket* to_sock)
{
    auto upNode = from_sock->node;
    auto downNode = to_sock->node;

    // Change input to default value
    if (getMaterialXNode(downNode)) {
        mx::NodeDefPtr nodeDef = getMaterialXNode(downNode)->getNodeDef(
            getMaterialXNode(downNode)->getName());

        auto pin = to_sock;
        mx::ValuePtr val =
            nodeDef->getActiveInput(getMaterialXPinInput(pin)->getName())
                ->getValue();
        if (getMaterialXNode(downNode)->getType() ==
                mx::SURFACE_SHADER_TYPE_STRING &&
            getMaterialXNodeGraph(upNode)) {
            getMaterialXPinInput(pin)->setConnectedOutput(nullptr);
        }
        else {
            getMaterialXPinInput(pin)->setConnectedNode(nullptr);
        }
        if (getMaterialXInput(upNode)) {
            // Remove interface value in order to set the default of the
            // input
            getMaterialXPinInput(pin)->setInterfaceName(mx::EMPTY_STRING);
            setDefaults(getMaterialXPinInput(pin));
            setDefaults(getMaterialXInput(upNode));
        }

        // Remove any output reference
        getMaterialXPinInput(pin)->removeAttribute(
            mx::PortElement::OUTPUT_ATTRIBUTE);

        // If a value exists update the input with it
        if (val) {
            getMaterialXPinInput(pin)->setValueString(val->getValueString());
        }
    }
    else if (getMaterialXNodeGraph(downNode)) {
        // Set default values for nodegraph node pins ie nodegraph inputs
        mx::NodeDefPtr nodeDef = getMaterialXNodeGraph(downNode)->getNodeDef();

        auto pin = to_sock;
        if (getMaterialXInput(upNode)) {
            getMaterialXNodeGraph(downNode)
                ->getInput(pin->identifier)
                ->setInterfaceName(mx::EMPTY_STRING);
            setDefaults(getMaterialXInput(upNode));
        }

        getMaterialXPinInput(pin)->setConnectedNode(nullptr);
        setDefaults(getMaterialXPinInput(pin));
    }
    else if (getMaterialXOutput(downNode)) {
        getMaterialXOutput(downNode)->removeAttribute("nodename");
    }
}

RUZINO_NAMESPACE_CLOSE_SCOPE
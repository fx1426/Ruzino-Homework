#pragma once

#include "GUI/text_editor_widget.hpp"
#include "MCore/MaterialXNodeTree.hpp"
#include "MCore/api.h"

USTC_CG_NAMESPACE_OPEN_SCOPE

class MCORE_API MaterialXDocumentViewer : public TextEditorWidget {
   public:
    explicit MaterialXDocumentViewer(
        MaterialXNodeTree* node_tree,
        const std::string& title = "MaterialX Document");

    // Update the displayed document from the node tree
    void RefreshDocument();

   private:
    MaterialXNodeTree* node_tree_;
};

USTC_CG_NAMESPACE_CLOSE_SCOPE

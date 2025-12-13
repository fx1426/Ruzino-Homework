//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/Nodes/CompoundNode.h>

#include "api.h"

MATERIALX_NAMESPACE_BEGIN

/// Extending the CompoundNode with requirements for closures.
class HD_USTC_CG_API ClosureCompoundNodeSlang : public CompoundNode {
   public:
    static ShaderNodeImplPtr create();

    void addClassification(ShaderNode& node) const override;

    void emitFunctionDefinition(
        const ShaderNode& node,
        GenContext& context,
        ShaderStage& stage) const override;

    void emitFunctionCall(
        const ShaderNode& node,
        GenContext& context,
        ShaderStage& stage) const override;

   protected:
    void emitOpacityFetchFunctionDefinition(
        const ShaderNode& node,
        GenContext& context,
        ShaderStage& stage,
        bool isUsdPreviewSurface) const;
};

MATERIALX_NAMESPACE_END

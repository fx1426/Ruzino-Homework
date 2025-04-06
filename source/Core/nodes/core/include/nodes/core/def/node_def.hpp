#pragma once
#include <nodes/core/node_tree.hpp>
#include <nodes/core/node_exec.hpp>
#include <string>

#ifdef _WIN32
#define USTC_CG_EXPORT __declspec(dllexport)
#else
#define USTC_CG_EXPORT
#endif

// Add a new macro that supports auto - adding student names as a prefix.

#ifndef CGHW_STUDENT_NAME
#define NODE_DEF_OPEN_SCOPE \
    extern "C" {            \
    namespace USTC_CG {

#define NODE_DEF_CLOSE_SCOPE \
    }                        \
    }

#define NODE_DECLARATION_FUNCTION(name) \
    USTC_CG_EXPORT void node_declare_##name(USTC_CG::NodeDeclarationBuilder& b)

#define NODE_EXECUTION_FUNCTION(name) \
    USTC_CG_EXPORT bool node_execution_##name(ExeParams params)

#define NODE_DECLARATION_UI(name) \
    USTC_CG_EXPORT const char* node_ui_name_##name()

#define CONVERSION_DECLARATION_FUNCTION(from, to)      \
    USTC_CG_EXPORT void node_declare_##from##_to_##to( \
        USTC_CG::NodeDeclarationBuilder& b)

#define CONVERSION_EXECUTION_FUNCTION(from, to) \
    USTC_CG_EXPORT bool node_execution_##from##_to_##to(ExeParams params)

#define CONVERSION_FUNC_NAME(from, to)                               \
    USTC_CG_EXPORT std::string node_id_name_##from##_to_##to()       \
    {                                                                \
        return "conv_" + std::string(type_name<from>().data()) + "_to_" + \
               std::string(type_name<to>().data());                       \
    }

#define NODE_DECLARATION_REQUIRED(name)        \
    USTC_CG_EXPORT bool node_required_##name() \
    {                                          \
        return true;                           \
    }
#else // CGHW_STUDENT_NAME is defined!

#define PASTE_HELPER(a, b) a##b 
#define PASTE(a, b) PASTE_HELPER(a, b) 

#define NODE_DEF_OPEN_SCOPE \
    extern "C" {            \
    namespace USTC_CG {

#define NODE_DEF_CLOSE_SCOPE \
    }                        \
    }

#define NODE_DECLARATION_FUNCTION(name) \
    USTC_CG_EXPORT void PASTE(node_declare_##name##_,CGHW_STUDENT_NAME) (USTC_CG::NodeDeclarationBuilder& b)

#define NODE_EXECUTION_FUNCTION(name) \
    USTC_CG_EXPORT bool PASTE(node_execution_##name##_, CGHW_STUDENT_NAME)(ExeParams params)

#define NODE_DECLARATION_UI(name) \
    USTC_CG_EXPORT const char* PASTE(node_ui_name_##name##_, CGHW_STUDENT_NAME)()

#define CONVERSION_DECLARATION_FUNCTION(from, to)      \
    USTC_CG_EXPORT void node_declare_##from##_to_##to( \
        USTC_CG::NodeDeclarationBuilder& b)

#define CONVERSION_EXECUTION_FUNCTION(from, to) \
    USTC_CG_EXPORT bool node_execution_##from##_to_##to(ExeParams params)

#define CONVERSION_FUNC_NAME(from, to)                               \
    USTC_CG_EXPORT std::string node_id_name_##from##_to_##to()       \
    {                                                                \
        return "conv_" + std::string(type_name<from>().data()) + "_to_" + \
               std::string(type_name<to>().data());                       \
    }

#define NODE_DECLARATION_REQUIRED(name)        \
    USTC_CG_EXPORT bool PASTE(node_required_##name##_, CGHW_STUDENT_NAME)() \
    {                                          \
        return true;                           \
    }

#endif
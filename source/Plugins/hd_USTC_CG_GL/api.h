
#pragma once

#define USTC_CG_NAMESPACE_OPEN_SCOPE  namespace USTC_CG {
#define USTC_CG_NAMESPACE_CLOSE_SCOPE }

#if defined(_MSC_VER)
#define HD_USTC_CG_GL_EXPORT   __declspec(dllexport)
#define HD_USTC_CG_GL_IMPORT   __declspec(dllimport)
#define HD_USTC_CG_GL_NOINLINE __declspec(noinline)
#define HD_USTC_CG_GL_INLINE   __forceinline
#else
#define HD_USTC_CG_GL_EXPORT __attribute__((visibility("default")))
#define HD_USTC_CG_GL_IMPORT
#define HD_USTC_CG_GL_NOINLINE __attribute__((noinline))
#define HD_USTC_CG_GL_INLINE   __attribute__((always_inline)) inline
#endif

#if BUILD_HD_USTC_CG_GL_MODULE
#define HD_USTC_CG_GL_API    HD_USTC_CG_GL_EXPORT
#define HD_USTC_CG_GL_EXTERN extern
#else
#define HD_USTC_CG_GL_API HD_USTC_CG_GL_IMPORT
#if defined(_MSC_VER)
#define HD_USTC_CG_GL_EXTERN
#else
#define HD_USTC_CG_GL_EXTERN extern
#endif
#endif

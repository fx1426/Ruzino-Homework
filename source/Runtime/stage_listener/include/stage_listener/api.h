#pragma once

#define USTC_CG_NAMESPACE_OPEN_SCOPE  namespace USTC_CG {
#define USTC_CG_NAMESPACE_CLOSE_SCOPE }

#if defined(_MSC_VER)
#define STAGE_LISTENER_EXPORT   __declspec(dllexport)
#define STAGE_LISTENER_IMPORT   __declspec(dllimport)
#define STAGE_LISTENER_NOINLINE __declspec(noinline)
#define STAGE_LISTENER_INLINE   __forceinline
#else
#define STAGE_LISTENER_EXPORT __attribute__((visibility("default")))
#define STAGE_LISTENER_IMPORT
#define STAGE_LISTENER_NOINLINE __attribute__((noinline))
#define STAGE_LISTENER_INLINE   __attribute__((always_inline)) inline
#endif

#if BUILD_STAGE_LISTENER_MODULE
#define STAGE_LISTENER_API    STAGE_LISTENER_EXPORT
#define STAGE_LISTENER_EXTERN extern
#else
#define STAGE_LISTENER_API STAGE_LISTENER_IMPORT
#if defined(_MSC_VER)
#define STAGE_LISTENER_EXTERN
#else
#define STAGE_LISTENER_EXTERN extern
#endif
#endif

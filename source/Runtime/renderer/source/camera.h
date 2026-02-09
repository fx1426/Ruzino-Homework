#pragma once
#include "api.h"
#include "pxr/base/gf/matrix4f.h"
#include "pxr/base/gf/rect2i.h"
#include "pxr/imaging/hd/camera.h"
#include "pxr/imaging/hd/renderPassState.h"
#include "pxr/pxr.h"
#include "pxr/usd/sdf/path.h"

RUZINO_NAMESPACE_OPEN_SCOPE
using namespace pxr;

class Hd_RUZINO_Camera : public HdCamera {
   public:
    explicit Hd_RUZINO_Camera(SdfPath const& id) : HdCamera(id)
    {
    }

    void Sync(
        HdSceneDelegate* sceneDelegate,
        HdRenderParam* renderParam,
        HdDirtyBits* dirtyBits) override;

    void update(const HdRenderPassStateSharedPtr& renderPassState) const;

    mutable GfMatrix4f projMatrix;
    mutable GfMatrix4f inverseProjMatrix;
    mutable GfMatrix4f viewMatrix;
    mutable GfMatrix4f inverseViewMatrix;
    mutable GfRect2i dataWindow;
};
RUZINO_NAMESPACE_CLOSE_SCOPE
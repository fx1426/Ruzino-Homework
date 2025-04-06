#pragma once

#include <pxr/base/tf/weakPtr.h>
#include <pxr/usd/usd/common.h>
#include <pxr/usd/usd/notice.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>

#include <mutex>

#include "api.h"
#include "stage/stage.hpp"
#include "stage_listener/api.h"

USTC_CG_NAMESPACE_OPEN_SCOPE

class STAGE_LISTENER_API StageListener : public pxr::TfWeakBase {
   public:
    using DirtyPathSet = std::unordered_set<pxr::SdfPath, pxr::SdfPath::Hash>;
    explicit StageListener(Stage* stage);

    void CapturePrimSnapshot();

    std::mutex& GetMutex();

    void GetDirtyPaths(DirtyPathSet& outDirtyPaths);

   private:
    // 处理Prim结构变化（添加/删除）
    void OnStageContentsChanged(
        const pxr::UsdNotice::StageContentsChanged& notice);

    // 处理属性变化（如变换、几何数据）
    void OnObjectsChanged(const pxr::UsdNotice::ObjectsChanged& notice);

    Stage* stage_;
    std::mutex mutex_;
    DirtyPathSet dirtyPaths_;  // 脏Prim路径集合
    pxr::TfNotice::Key stageContentsChangedKey_, objectsChangedKey_;
    DirtyPathSet previousPrimPaths_;  // 上一帧的Prim路径集合
};

USTC_CG_NAMESPACE_CLOSE_SCOPE
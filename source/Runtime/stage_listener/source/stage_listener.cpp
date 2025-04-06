#include "stage_listener/stage_listener.h"

#include "stage/stage.hpp"

USTC_CG_NAMESPACE_OPEN_SCOPE

StageListener::StageListener(Stage* stage) : stage_(stage)
{
    // 检查stage是否有效
    if (!stage_ || !stage_->get_usd_stage()) {
        throw std::runtime_error("Invalid stage pointer");
    }

    pxr::UsdStageWeakPtr stage_weak_ptr(stage_->get_usd_stage());

    // 注册监听场景内容变化（Prim添加/删除）
    stageContentsChangedKey_ = pxr::TfNotice::Register(
        pxr::TfCreateWeakPtr(this),
        &StageListener::OnStageContentsChanged,
        stage_weak_ptr);

    // 注册监听属性变化
    objectsChangedKey_ = pxr::TfNotice::Register(
        pxr::TfCreateWeakPtr(this),
        &StageListener::OnObjectsChanged,
        stage_weak_ptr);
}

void StageListener::CapturePrimSnapshot()
{
    std::lock_guard<std::mutex> lock(mutex_);
    previousPrimPaths_.clear();
    auto stage_ptr = stage_->get_usd_stage();
    for (const pxr::UsdPrim& prim : stage_ptr->Traverse()) {
        previousPrimPaths_.insert(prim.GetPath());
    }
}

std::mutex& StageListener::GetMutex()
{
    return mutex_;
}

void StageListener::GetDirtyPaths(DirtyPathSet& outDirtyPaths)
{
    std::lock_guard<std::mutex> lock(mutex_);
    outDirtyPaths = std::move(dirtyPaths_);
    dirtyPaths_.clear();  // 确保原容器为空
}

void StageListener::OnStageContentsChanged(
    const pxr::UsdNotice::StageContentsChanged& notice)
{
    if (!stage_ || !stage_->get_usd_stage())
        return;

    std::lock_guard<std::mutex> lock(mutex_);

    // 获取当前所有 Prim 路径
    DirtyPathSet currentPrimPaths;
    auto stage_ptr = stage_->get_usd_stage();
    for (const pxr::UsdPrim& prim : stage_ptr->Traverse()) {
        currentPrimPaths.insert(prim.GetPath());
    }

    // 对比差异：新增的 Prim
    for (const auto& path : currentPrimPaths) {
        if (!previousPrimPaths_.count(path)) {
            dirtyPaths_.insert(path);  // 新增的 Prim
        }
    }

    // 对比差异：删除的 Prim
    for (const auto& path : previousPrimPaths_) {
        if (!currentPrimPaths.count(path)) {
            dirtyPaths_.insert(path);  // 删除的 Prim
        }
    }

    // 更新快照为当前状态
    previousPrimPaths_.swap(currentPrimPaths);
}

void StageListener::OnObjectsChanged(
    const pxr::UsdNotice::ObjectsChanged& notice)
{
    if (!stage_) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    for (const pxr::SdfPath& path : notice.GetChangedInfoOnlyPaths()) {
        // 标记属性变化的Prim路径
        dirtyPaths_.insert(path.GetPrimPath());
    }
}

USTC_CG_NAMESPACE_CLOSE_SCOPE
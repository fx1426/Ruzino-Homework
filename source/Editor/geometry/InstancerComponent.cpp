#include <GCore/Components/InstancerComponent.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/vt/array.h>

USTC_CG_NAMESPACE_OPEN_SCOPE
InstancerComponent::InstancerComponent(Geometry* attached_operand)
    : GeometryComponent(attached_operand)
{
}

InstancerComponent::~InstancerComponent()
{
}

GeometryComponentHandle InstancerComponent::copy(Geometry* operand) const
{
    auto instancer = std::make_shared<InstancerComponent>(operand);
    instancer->instances_ = instances_;
    return instancer;
}

std::string InstancerComponent::to_string() const
{
    return "InstancerComponent";
}

void InstancerComponent::apply_transform(const pxr::GfMatrix4d& transform)
{
    for (auto& instance : instances_) {
        instance = transform * pxr::GfMatrix4d(instance);
    }
}

void InstancerComponent::add_instance(const pxr::GfMatrix4d& instance)
{
    instances_.push_back(instance);
}

const pxr::VtArray<pxr::GfMatrix4d>& InstancerComponent::get_instances() const
{
    return instances_;
}

pxr::VtArray<int> InstancerComponent::get_proto_indices()
{
    return pxr::VtArray<int>(instances_.size(), 0);
}

USTC_CG_NAMESPACE_CLOSE_SCOPE

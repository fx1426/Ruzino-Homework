#pragma once
#include "GCore/Components.h"
#include "GCore/api.h"
#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/vt/array.h"

USTC_CG_NAMESPACE_OPEN_SCOPE

class GEOMETRY_API InstancerComponent final : public GeometryComponent {
   public:
    explicit InstancerComponent(Geometry* attached_operand);

    ~InstancerComponent() override;
    GeometryComponentHandle copy(Geometry* operand) const override;
    std::string to_string() const override;
    void apply_transform(const pxr::GfMatrix4d& transform) override;

    void add_instance(const pxr::GfMatrix4d& instance);
    const pxr::VtArray<pxr::GfMatrix4d>& get_instances() const;
    pxr::VtArray<int> get_proto_indices();

   private:
    pxr::VtArray<pxr::GfMatrix4d> instances_;
};

USTC_CG_NAMESPACE_CLOSE_SCOPE

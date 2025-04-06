#include <random>
#include <string>

#include "GCore/Components/MeshOperand.h"
#include "GCore/GOP.h"
#include "nodes/core/def/node_def.hpp"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/vt/array.h"

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(mesh_add_vertex_scalar_quantity)
{
    b.add_input<Geometry>("Geometry");
    b.add_input<pxr::VtArray<float>>("Vertex scalar");
    b.add_input<std::string>("Quantity name");

    b.add_output<Geometry>("Geometry");
}

NODE_EXECUTION_FUNCTION(mesh_add_vertex_scalar_quantity)
{
    auto mesh = params.get_input<Geometry>("Geometry");
    auto vertexScalar = params.get_input<pxr::VtArray<float>>("Vertex scalar");
    auto quantityName = params.get_input<std::string>("Quantity name");
    quantityName = "vs_" + quantityName;

    auto meshComponent = mesh.get_component<MeshComponent>();

    if (!meshComponent) {
        return false;
    }

    if (meshComponent->get_vertices().size() != vertexScalar.size()) {
        return false;
    }

    meshComponent->add_vertex_scalar_quantity(quantityName, vertexScalar);
    params.set_output("Geometry", std::move(mesh));
    // params.set_output("Mesh", mesh);
    return true;
}

NODE_DECLARATION_FUNCTION(mesh_add_face_scalar_quantity)
{
    b.add_input<Geometry>("Geometry");
    b.add_input<pxr::VtArray<float>>("Face scalar");
    b.add_input<std::string>("Quantity name");

    b.add_output<Geometry>("Geometry");
}

NODE_EXECUTION_FUNCTION(mesh_add_face_scalar_quantity)
{
    auto mesh = params.get_input<Geometry>("Geometry");
    auto faceScalar = params.get_input<pxr::VtArray<float>>("Face scalar");
    auto quantityName = params.get_input<std::string>("Quantity name");
    quantityName = "fs_" + quantityName;

    auto meshComponent = mesh.get_component<MeshComponent>();

    if (!meshComponent) {
        return false;
    }

    if (meshComponent->get_face_vertex_counts().size() != faceScalar.size()) {
        return false;
    }

    meshComponent->add_face_scalar_quantity(quantityName, faceScalar);
    params.set_output("Geometry", std::move(mesh));
    return true;
}

NODE_DECLARATION_FUNCTION(mesh_add_vertex_color_quantity)
{
    b.add_input<Geometry>("Geometry");
    b.add_input<pxr::VtArray<pxr::GfVec3f>>("Vertex color");
    b.add_input<std::string>("Quantity name");

    b.add_output<Geometry>("Geometry");
}

NODE_EXECUTION_FUNCTION(mesh_add_vertex_color_quantity)
{
    auto mesh = params.get_input<Geometry>("Geometry");
    auto vertexColor =
        params.get_input<pxr::VtArray<pxr::GfVec3f>>("Vertex color");
    auto quantityName = params.get_input<std::string>("Quantity name");
    quantityName = "vc_" + quantityName;

    auto meshComponent = mesh.get_component<MeshComponent>();

    if (!meshComponent) {
        return false;
    }

    if (meshComponent->get_vertices().size() != vertexColor.size()) {
        return false;
    }

    meshComponent->add_vertex_color_quantity(quantityName, vertexColor);
    params.set_output("Geometry", std::move(mesh));
    return true;
}

NODE_DECLARATION_FUNCTION(mesh_add_face_color_quantity)
{
    b.add_input<Geometry>("Geometry");
    b.add_input<pxr::VtArray<pxr::GfVec3f>>("Face color");
    b.add_input<std::string>("Quantity name");

    b.add_output<Geometry>("Geometry");
}

NODE_EXECUTION_FUNCTION(mesh_add_face_color_quantity)
{
    auto mesh = params.get_input<Geometry>("Geometry");
    auto faceColor = params.get_input<pxr::VtArray<pxr::GfVec3f>>("Face color");
    auto quantityName = params.get_input<std::string>("Quantity name");
    quantityName = "fc_" + quantityName;

    auto meshComponent = mesh.get_component<MeshComponent>();

    if (!meshComponent) {
        return false;
    }

    if (meshComponent->get_face_vertex_counts().size() != faceColor.size()) {
        return false;
    }

    meshComponent->add_face_color_quantity(quantityName, faceColor);
    params.set_output("Geometry", std::move(mesh));
    return true;
}

NODE_DECLARATION_FUNCTION(mesh_add_vertex_vector_quantity)
{
    b.add_input<Geometry>("Geometry");
    b.add_input<pxr::VtArray<pxr::GfVec3f>>("Vertex vector");
    b.add_input<std::string>("Quantity name");

    b.add_output<Geometry>("Geometry");
}

NODE_EXECUTION_FUNCTION(mesh_add_vertex_vector_quantity)
{
    auto mesh = params.get_input<Geometry>("Geometry");
    auto vertexVector =
        params.get_input<pxr::VtArray<pxr::GfVec3f>>("Vertex vector");
    auto quantityName = params.get_input<std::string>("Quantity name");
    quantityName = "vv_" + quantityName;

    auto meshComponent = mesh.get_component<MeshComponent>();

    if (!meshComponent) {
        return false;
    }

    if (meshComponent->get_vertices().size() != vertexVector.size()) {
        return false;
    }

    meshComponent->add_vertex_vector_quantity(quantityName, vertexVector);
    params.set_output("Geometry", std::move(mesh));
    return true;
}

NODE_DECLARATION_FUNCTION(mesh_add_face_vector_quantity)
{
    b.add_input<Geometry>("Geometry");
    b.add_input<pxr::VtArray<pxr::GfVec3f>>("Face vector");
    b.add_input<std::string>("Quantity name");

    b.add_output<Geometry>("Geometry");
}

NODE_EXECUTION_FUNCTION(mesh_add_face_vector_quantity)
{
    auto mesh = params.get_input<Geometry>("Geometry");
    auto faceVector =
        params.get_input<pxr::VtArray<pxr::GfVec3f>>("Face vector");
    auto quantityName = params.get_input<std::string>("Quantity name");
    quantityName = "fv_" + quantityName;

    auto meshComponent = mesh.get_component<MeshComponent>();

    if (!meshComponent) {
        return false;
    }

    if (meshComponent->get_face_vertex_counts().size() != faceVector.size()) {
        return false;
    }

    meshComponent->add_face_vector_quantity(quantityName, faceVector);
    params.set_output("Geometry", std::move(mesh));
    return true;
}

NODE_DECLARATION_FUNCTION(mesh_add_vertex_parameterization_quantity)
{
    b.add_input<Geometry>("Geometry");
    b.add_input<pxr::VtArray<pxr::GfVec2f>>("Vertex parameterization");
    b.add_input<std::string>("Quantity name");

    b.add_output<Geometry>("Geometry");
}

NODE_EXECUTION_FUNCTION(mesh_add_vertex_parameterization_quantity)
{
    auto mesh = params.get_input<Geometry>("Geometry");
    auto vertexParameterization =
        params.get_input<pxr::VtArray<pxr::GfVec2f>>("Vertex parameterization");
    auto quantityName = params.get_input<std::string>("Quantity name");
    quantityName = "vp_" + quantityName;

    auto meshComponent = mesh.get_component<MeshComponent>();

    if (!meshComponent) {
        return false;
    }

    if (meshComponent->get_vertices().size() != vertexParameterization.size()) {
        return false;
    }

    meshComponent->add_vertex_parameterization_quantity(
        quantityName, vertexParameterization);
    params.set_output("Geometry", std::move(mesh));
    return true;
}

NODE_DECLARATION_FUNCTION(mesh_add_face_corner_parameterization_quantity)
{
    b.add_input<Geometry>("Geometry");
    b.add_input<pxr::VtArray<pxr::GfVec2f>>("Face corner parameterization");
    b.add_input<std::string>("Quantity name");

    b.add_output<Geometry>("Geometry");
}

NODE_EXECUTION_FUNCTION(mesh_add_face_corner_parameterization_quantity)
{
    auto mesh = params.get_input<Geometry>("Geometry");
    auto faceCornerParameterization =
        params.get_input<pxr::VtArray<pxr::GfVec2f>>(
            "Face corner parameterization");
    auto quantityName = params.get_input<std::string>("Quantity name");
    quantityName = "fcp_" + quantityName;

    auto meshComponent = mesh.get_component<MeshComponent>();

    if (!meshComponent) {
        return false;
    }

    int expectedSize = 0;
    for (int count : meshComponent->get_face_vertex_counts()) {
        expectedSize += count;
    }

    if (expectedSize != faceCornerParameterization.size()) {
        return false;
    }

    meshComponent->add_face_corner_parameterization_quantity(
        quantityName, faceCornerParameterization);
    params.set_output("Geometry", std::move(mesh));
    return true;
}

NODE_DECLARATION_UI(mesh_add_quantity);
NODE_DEF_CLOSE_SCOPE

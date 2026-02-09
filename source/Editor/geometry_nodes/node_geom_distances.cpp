#include <pxr/usd/usdGeom/mesh.h>

#include <Eigen/Core>
#include <Eigen/Eigen>
#include <cfloat>
#include <cstdlib>
#include <unordered_set>
#include <vector>

#include "GCore/Components.h"
#include "GCore/Components/MeshComponent.h"
#include "GCore/GOP.h"
#include "GCore/util_openmesh_bind.h"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"

using Vec = Eigen::Vector3d;
NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(vertedge_dist)
{
    b.add_input<Vec>("vertex");
    b.add_input<Vec>("Edge_v1");
    b.add_input<Vec>("Edge_v2");
    b.add_output<double>("distance");
}
NODE_EXECUTION_FUNCTION(vertedge_dist)
{
    auto vertex = params.get_input<Vec>("vertex");
    auto v1 = params.get_input<Vec>("Edge_v1");
    auto v2 = params.get_input<Vec>("Edge_v2");
    double distance = 0;
    Vec d1 = v2 - v1;      // direction vector one
    Vec d2 = vertex - v1;  // direction vector two
    distance = (d1.cross(d2)).norm() / d1.norm();

    params.set_output<double>("distance", std::move(distance));
    return true;
}

NODE_DECLARATION_FUNCTION(vertface_dist)
{
    b.add_input<Vec>("vertex");
    b.add_input<Vec>("Face_v0");
    b.add_input<Vec>("Face_v1");
    b.add_input<Vec>("Face_v2");
    b.add_output<double>("distance");
}
NODE_EXECUTION_FUNCTION(vertface_dist)
{
    auto vertex = params.get_input<Vec>("vertex");
    auto v0 = params.get_input<Vec>("Face_v0");
    auto v1 = params.get_input<Vec>("Face_v1");
    auto v2 = params.get_input<Vec>("Face_v2");  // 3 points of the triangle
                                                 // face
    double distance = 0;
    Vec vec1 = v1 - v0;
    Vec vec2 = v2 - v0;
    Vec normal = vec1.cross(vec2);
    distance = std::abs(normal.dot(v0 - vertex) / normal.norm());

    params.set_output<double>("distance", std::move(distance));
    return true;
}

NODE_DECLARATION_FUNCTION(vertvert_dist)
{
    b.add_input<Vec>("vertex1");
    b.add_input<Vec>("vertex2");
    b.add_output<double>("distance");
}

NODE_EXECUTION_FUNCTION(vertvert_dist)
{
    auto v1 = params.get_input<Vec>("vertex1");
    auto v2 = params.get_input<Vec>("vertex2");
    double distance = (v1 - v2).norm();
    params.set_output<double>("distance", std::move(distance));
    return true;
}

NODE_DECLARATION_FUNCTION(edgeedge_dist)
{
    b.add_input<Vec>("Edge1_p1");
    b.add_input<Vec>("Edge1_p2");
    b.add_input<Vec>("Edge2_p3");
    b.add_input<Vec>("Edge2_p4");
    b.add_output<double>("distance");
}

NODE_EXECUTION_FUNCTION(edgeedge_dist)
{
    auto p1 = params.get_input<Vec>("Edge1_p1");
    auto p2 = params.get_input<Vec>("Edge1_p2");
    auto p3 = params.get_input<Vec>("Edge2_p3");
    auto p4 = params.get_input<Vec>("Edge2_p4");
    Vec d1 = p2 - p1;
    Vec d2 = p3 - p4;  // direction vector
    Vec p1p3 = p3 - p1;
    double distance = 0;
    if ((d1.cross(d2)).norm() == 0) {  // Two edges are parallel
        distance = (p1p3.cross(d1)).norm() / d1.norm();
    }
    else {
        distance = std::abs(p1p3.dot(d1.cross(d2))) / (d1.cross(d2)).norm();
    }

    params.set_output<double>("distance", std::move(distance));
    return true;
}

NODE_DECLARATION_FUNCTION(vertmesh_dist)
{
    b.add_input<Vec>("vertex");
    b.add_input<Geometry>("Geometry");
    b.add_input<double>("radius");
    b.add_input<double>("voxelSize");
    b.add_output<double>("distance");
}

NODE_EXECUTION_FUNCTION(vertmesh_dist)
{
    auto vertex = params.get_input<Vec>("vertex");
    auto geometry = params.get_input<Geometry>("Geometry");
    double radius = params.get_input<double>("radius");
    double voxelSize = params.get_input<double>("voxelSize");
    auto mesh = operand_to_openmesh(&geometry);
    auto meshcomponent = geometry.get_component<MeshComponent>();
    auto faceVertexIndices = meshcomponent->get_face_vertex_indices();
    auto faceVertexCounts = meshcomponent->get_face_vertex_counts();
    auto vertices = meshcomponent->get_vertices();
    double distance = DBL_MAX;
    Eigen::MatrixXd verticesMatrix(mesh->n_vertices(), 3);
    for (const auto& vh : mesh->vertices()) {
        auto point = mesh->point(vh);
        int i = vh.idx();
        verticesMatrix(i, 0) = point[0];
        verticesMatrix(i, 1) = point[1];
        verticesMatrix(i, 2) = point[2];
    }
    Eigen::RowVector3d leftBottomCorner, rightTopCorner;
    leftBottomCorner = verticesMatrix.colwise().minCoeff();
    rightTopCorner = verticesMatrix.colwise().maxCoeff();
    double one_div_voxelSize;
    one_div_voxelSize = 1.0 / voxelSize;
    Eigen::Array<double, 1, 3> range = rightTopCorner - leftBottomCorner;
    Eigen::Array<int, 1, 3> voxelCount =
        (range * one_div_voxelSize).ceil().template cast<int>();
    if (voxelCount.minCoeff() <= 0) {
        one_div_voxelSize = 1.0 / (range.maxCoeff() * 1.01);
        voxelCount.setOnes();
    }
    int voxelCount0x1 = voxelCount[0] * voxelCount[1];
    Eigen::Array<int, 1, 3> mins, maxs;
    Eigen::Matrix<double, 1, 3> vertex_minus;
    Eigen::Matrix<double, 1, 3> vertex_plus;
    for (int i = 0; i <= 2; i++) {
        vertex_minus[i] =
            (vertex[i] - radius - leftBottomCorner[i]) * one_div_voxelSize;
        vertex_plus[i] =
            (vertex[i] + radius - leftBottomCorner[i]) * one_div_voxelSize;
    }
    for (int i = 0; i <= 2; i++) {
        mins(i) = std::max(static_cast<int>(floor(vertex_minus[i])), 0);
        maxs(i) = std::min(
            static_cast<int>(floor(vertex_plus[i])), voxelCount[i] - 1);
    }
    std::unordered_set<int> faceID;
    faceID.clear();
    std::map<int, std::vector<int>> voxel;
    for (int svI = 0; svI < mesh->n_vertices(); ++svI) {
        int idx = 0;
        Eigen::Matrix<double, 1, 3> pos;
        pos[0] =
            (verticesMatrix(svI, 0) - leftBottomCorner[0]) * one_div_voxelSize;
        pos[1] =
            (verticesMatrix(svI, 1) - leftBottomCorner[1]) * one_div_voxelSize;
        pos[2] =
            (verticesMatrix(svI, 2) - leftBottomCorner[2]) * one_div_voxelSize;
        int ix = static_cast<int>(floor(pos[0]));
        int iy = static_cast<int>(floor(pos[1]));
        int iz = static_cast<int>(floor(pos[2]));
        assert(
            ix >= 0 && iy >= 0 && iz >= 0 && ix < voxelCount[0] &&
            iy < voxelCount[1] && iz < voxelCount[2]);
        idx = ix + iy * voxelCount[0] + iz * voxelCount0x1;
        voxel[idx].push_back(svI);
    }
    for (int iz = mins[2]; iz <= maxs[2]; ++iz) {
        int zOffset = iz * voxelCount0x1;
        for (int iy = mins[1]; iy <= maxs[1]; ++iy) {
            int yzOffset = iy * voxelCount[0] + zOffset;
            for (int ix = mins[0]; ix <= maxs[0]; ++ix) {
                const auto voxelI = voxel.find(ix + yzOffset);
                if (voxelI != voxel.end()) {
                    for (const auto& indI : voxelI->second) {
                        if (indI >= mesh->n_faces()) {
                            faceID.insert(indI - mesh->n_faces());
                        }
                    }
                }
            }
        }
    }
    for (const auto& idx : faceID) {
        int start = 0;
        for (int i = 0; i < idx; i++) {
            start += faceVertexCounts[i];
        }
        int faceVertexCount = faceVertexCounts[idx];
        std::vector<int> faceVertexIndicesForFace;
        for (int i = 0; i < faceVertexCount; i++) {
            faceVertexIndicesForFace.push_back(faceVertexIndices[start + i]);
        }
        std::vector<glm::vec3> faceVertices;
        for (int index : faceVertexIndicesForFace) {
            faceVertices.push_back(vertices[index]);
        }
        auto vertices0 = faceVertices[0];
        auto vertices1 = faceVertices[1];
        auto vertices2 = faceVertices[2];
        Vec v0(vertices0[0], vertices0[1], vertices0[2]);
        Vec v1(vertices1[0], vertices1[1], vertices1[2]);
        Vec v2(vertices2[0], vertices2[1], vertices2[2]);
        Vec vec1 = v1 - v0;
        Vec vec2 = v2 - v0;
        Vec normal = vec1.cross(vec2);
        double temp = std::abs(normal.dot(v0 - vertex) / normal.norm());
        distance = std::min(distance, temp);
    }

    params.set_output<double>("distance", std::move(distance));
    return true;
}

NODE_DECLARATION_FUNCTION(meshmesh_dist)
{
    b.add_input<Geometry>("Geometry1");
    b.add_input<Geometry>("Geometry2");
    b.add_input<double>("radius");
    b.add_input<double>("voxelSize");
    b.add_output<double>("distance");
}

NODE_EXECUTION_FUNCTION(meshmesh_dist)
{
    auto geometry1 = params.get_input<Geometry>("Geometry1");
    auto geometry2 = params.get_input<Geometry>("Geometry2");
    double radius = params.get_input<double>("radius");
    double voxelSize = params.get_input<double>("voxelSize");
    auto mesh1 = operand_to_openmesh(&geometry1);
    auto mesh2 = operand_to_openmesh(&geometry2);
    auto meshcomponent2 = geometry2.get_component<MeshComponent>();
    auto faceVertexIndices = meshcomponent2->get_face_vertex_indices();
    auto faceVertexCounts = meshcomponent2->get_face_vertex_counts();
    auto vertices = meshcomponent2->get_vertices();
    double distance = DBL_MAX;
    Eigen::MatrixXd verticesMatrix(mesh2->n_vertices(), 3);
    for (const auto& vh : mesh2->vertices()) {
        auto point = mesh2->point(vh);
        int i = vh.idx();
        verticesMatrix(i, 0) = point[0];
        verticesMatrix(i, 1) = point[1];
        verticesMatrix(i, 2) = point[2];
    }
    Eigen::RowVector3d leftBottomCorner, rightTopCorner;
    leftBottomCorner = verticesMatrix.colwise().minCoeff();
    rightTopCorner = verticesMatrix.colwise().maxCoeff();
    double one_div_voxelSize;
    one_div_voxelSize = 1.0 / voxelSize;
    Eigen::Array<double, 1, 3> range = rightTopCorner - leftBottomCorner;
    Eigen::Array<int, 1, 3> voxelCount =
        (range * one_div_voxelSize).ceil().template cast<int>();
    if (voxelCount.minCoeff() <= 0) {
        one_div_voxelSize = 1.0 / (range.maxCoeff() * 1.01);
        voxelCount.setOnes();
    }
    int voxelCount0x1 = voxelCount[0] * voxelCount[1];
    std::map<int, std::vector<int>> voxel;
    for (int svI = 0; svI < mesh2->n_vertices(); ++svI) {
        int idx = 0;
        Eigen::Matrix<double, 1, 3> pos;
        pos[0] =
            (verticesMatrix(svI, 0) - leftBottomCorner[0]) * one_div_voxelSize;
        pos[1] =
            (verticesMatrix(svI, 1) - leftBottomCorner[1]) * one_div_voxelSize;
        pos[2] =
            (verticesMatrix(svI, 2) - leftBottomCorner[2]) * one_div_voxelSize;
        int ix = static_cast<int>(floor(pos[0]));
        int iy = static_cast<int>(floor(pos[1]));
        int iz = static_cast<int>(floor(pos[2]));
        assert(
            ix >= 0 && iy >= 0 && iz >= 0 && ix < voxelCount[0] &&
            iy < voxelCount[1] && iz < voxelCount[2]);
        idx = ix + iy * voxelCount[0] + iz * voxelCount0x1;
        voxel[idx].push_back(svI);
    }
    Eigen::Array<int, 1, 3> mins, maxs;
    std::unordered_set<int> faceID;
    for (const auto& vertex_handle : mesh1->vertices()) {
        Eigen::Matrix<double, 1, 3> vertex_minus;
        Eigen::Matrix<double, 1, 3> vertex_plus;
        Eigen::Matrix<double, 1, 3> vertex;
        const auto& position = mesh1->point(vertex_handle);
        double temp_distance = DBL_MAX;
        for (int i = 0; i <= 2; i++) {
            vertex[i] = position[i];
        }
        for (int i = 0; i <= 2; i++) {
            vertex_minus[i] =
                (vertex[i] - radius - leftBottomCorner[i]) * one_div_voxelSize;
            vertex_plus[i] =
                (vertex[i] + radius - leftBottomCorner[i]) * one_div_voxelSize;
        }
        for (int i = 0; i <= 2; i++) {
            mins(i) = std::max(static_cast<int>(floor(vertex_minus[i])), 0);
            maxs(i) = std::min(
                static_cast<int>(floor(vertex_plus[i])), voxelCount[i] - 1);
        }
        faceID.clear();
        for (int iz = mins[2]; iz <= maxs[2]; ++iz) {
            int zOffset = iz * voxelCount0x1;
            for (int iy = mins[1]; iy <= maxs[1]; ++iy) {
                int yzOffset = iy * voxelCount[0] + zOffset;
                for (int ix = mins[0]; ix <= maxs[0]; ++ix) {
                    const auto voxelI = voxel.find(ix + yzOffset);
                    if (voxelI != voxel.end()) {
                        for (const auto& indI : voxelI->second) {
                            if (indI >= mesh2->n_faces()) {
                                faceID.insert(indI - mesh2->n_faces());
                            }
                        }
                    }
                }
            }
        }
        for (const auto& idx : faceID) {
            int start = 0;
            for (int i = 0; i < idx; i++) {
                start += faceVertexCounts[i];
            }
            int faceVertexCount = faceVertexCounts[idx];
            std::vector<int> faceVertexIndicesForFace;
            for (int i = 0; i < faceVertexCount; i++) {
                faceVertexIndicesForFace.push_back(
                    faceVertexIndices[start + i]);
            }
            std::vector<glm::vec3> faceVertices;
            for (int index : faceVertexIndicesForFace) {
                faceVertices.push_back(vertices[index]);
            }
            auto vertices0 = faceVertices[0];
            auto vertices1 = faceVertices[1];
            auto vertices2 = faceVertices[2];
            Vec v0(vertices0[0], vertices0[1], vertices0[2]);
            Vec v1(vertices1[0], vertices1[1], vertices1[2]);
            Vec v2(vertices2[0], vertices2[1], vertices2[2]);
            Vec vec1 = v1 - v0;
            Vec vec2 = v2 - v0;
            Vec normal = vec1.cross(vec2);
            double temp =
                std::abs(normal.dot(v0 - vertex.transpose()) / normal.norm());
            temp_distance = std::min(temp_distance, temp);
        }
        distance = std::min(distance, temp_distance);
    }
    params.set_output<double>("distance", std::move(distance));
    return true;
}

NODE_DECLARATION_UI(vertedge_dist);
NODE_DECLARATION_UI(vertface_dist);
NODE_DECLARATION_UI(vertvert_dist);
NODE_DECLARATION_UI(edgeedge_dist);
NODE_DECLARATION_UI(vertmesh_dist);
NODE_DECLARATION_UI(meshmesh_dist);
NODE_DEF_CLOSE_SCOPE
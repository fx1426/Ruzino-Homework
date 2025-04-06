#include <exception>

#include "nodes/core/def/node_def.hpp"
#include "polyscope/curve_network.h"
#include "polyscope/pick.h"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include "polyscope/structure.h"
#include "polyscope/surface_mesh.h"
#include "polyscope_widget/polyscope_renderer.h"

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(get_picked_vertices)
{
    b.add_output<std::string>("Picked Structure Name [0]");
    b.add_output<std::string>("Picked Structure Name [1]");
    b.add_output<size_t>("Picked Vertex [0] Index (Left Ctrl + Left Click)");
    b.add_output<size_t>("Picked Vertex [1] Index             (Left Click)");
    b.add_output<float>("Picked Vertex [0] Position X");
    b.add_output<float>("Picked Vertex [0] Position Y");
    b.add_output<float>("Picked Vertex [0] Position Z");
    b.add_output<float>("Picked Vertex [1] Position X");
    b.add_output<float>("Picked Vertex [1] Position Y");
    b.add_output<float>("Picked Vertex [1] Position Z");
}

NODE_EXECUTION_FUNCTION(get_picked_vertices)
{
    auto pick = PolyscopeRenderer::GetPickResult();
    polyscope::Structure* structure_0 = pick[0].first;
    size_t index_0 = pick[0].second;
    polyscope::Structure* structure_1 = pick[1].first;
    size_t index_1 = pick[1].second;

    if (!structure_0 || !structure_1) {
        std::cerr << "Ensure you have picked two items." << std::endl;
        return false;
    }

    if (structure_0->typeName() == "Surface Mesh") {
        auto mesh = dynamic_cast<polyscope::SurfaceMesh*>(structure_0);
        if (index_0 < mesh->nVertices()) {
            params.set_output(
                "Picked Structure Name [0]", structure_0->getName());
            params.set_output(
                "Picked Vertex [0] Index (Left Ctrl + Left Click)", index_0);

            auto pos = mesh->vertexPositions.getValue(index_0);
            params.set_output("Picked Vertex [0] Position X", pos.x);
            params.set_output("Picked Vertex [0] Position Y", pos.y);
            params.set_output("Picked Vertex [0] Position Z", pos.z);
        }
        else {
            std::cerr << "The first picked item is not a vertex/point/node."
                      << std::endl;
            return false;
        }
    }
    else if (structure_0->typeName() == "Point Cloud") {
        auto point_cloud = dynamic_cast<polyscope::PointCloud*>(structure_0);
        if (index_0 < point_cloud->nPoints()) {
            params.set_output(
                "Picked Structure Name [0]", structure_0->getName());
            params.set_output(
                "Picked Vertex [0] Index (Left Ctrl + Left Click)", index_0);

            auto pos = point_cloud->getPointPosition(index_0);
            params.set_output("Picked Vertex [0] Position X", pos.x);
            params.set_output("Picked Vertex [0] Position Y", pos.y);
            params.set_output("Picked Vertex [0] Position Z", pos.z);
        }
        else {
            std::cerr << "The first picked item is not a vertex/point/node."
                      << std::endl;
            return false;
        }
    }
    else if (structure_0->typeName() == "Curve Network") {
        auto curve_network =
            dynamic_cast<polyscope::CurveNetwork*>(structure_0);
        if (index_0 < curve_network->nNodes()) {
            params.set_output(
                "Picked Structure Name [0]", structure_0->getName());
            params.set_output(
                "Picked Vertex [0] Index (Left Ctrl + Left Click)", index_0);

            auto pos = curve_network->nodePositions.getValue(index_0);
            params.set_output("Picked Vertex [0] Position X", pos.x);
            params.set_output("Picked Vertex [0] Position Y", pos.y);
            params.set_output("Picked Vertex [0] Position Z", pos.z);
        }
        else {
            std::cerr << "The first picked item is not a vertex/point/node."
                      << std::endl;
            return false;
        }
    }
    else {
        std::cerr << "The first picked structure is not a surface mesh, point "
                     "cloud, or curve network."
                  << std::endl;
        return false;
    }

    if (structure_1->typeName() == "Surface Mesh") {
        auto mesh = dynamic_cast<polyscope::SurfaceMesh*>(structure_1);
        if (index_1 < mesh->nVertices()) {
            params.set_output(
                "Picked Structure Name [1]", structure_1->getName());
            params.set_output(
                "Picked Vertex [1] Index             (Left Click)", index_1);

            auto pos = mesh->vertexPositions.getValue(index_1);
            params.set_output("Picked Vertex [1] Position X", pos.x);
            params.set_output("Picked Vertex [1] Position Y", pos.y);
            params.set_output("Picked Vertex [1] Position Z", pos.z);
        }
        else {
            std::cerr << "The second picked item is not a vertex/point/node."
                      << std::endl;
            return false;
        }
    }
    else if (structure_1->typeName() == "Point Cloud") {
        auto point_cloud = dynamic_cast<polyscope::PointCloud*>(structure_1);
        if (index_1 < point_cloud->nPoints()) {
            params.set_output(
                "Picked Structure Name [1]", structure_1->getName());
            params.set_output(
                "Picked Vertex [1] Index             (Left Click)", index_1);

            auto pos = point_cloud->getPointPosition(index_1);
            params.set_output("Picked Vertex [1] Position X", pos.x);
            params.set_output("Picked Vertex [1] Position Y", pos.y);
            params.set_output("Picked Vertex [1] Position Z", pos.z);
        }
        else {
            std::cerr << "The second picked item is not a vertex/point/node."
                      << std::endl;
            return false;
        }
    }
    else if (structure_1->typeName() == "Curve Network") {
        auto curve_network =
            dynamic_cast<polyscope::CurveNetwork*>(structure_1);
        if (index_1 < curve_network->nNodes()) {
            params.set_output(
                "Picked Structure Name [1]", structure_1->getName());
            params.set_output(
                "Picked Vertex [1] Index             (Left Click)", index_1);

            auto pos = curve_network->nodePositions.getValue(index_1);
            params.set_output("Picked Vertex [1] Position X", pos.x);
            params.set_output("Picked Vertex [1] Position Y", pos.y);
            params.set_output("Picked Vertex [1] Position Z", pos.z);
        }
        else {
            std::cerr << "The second picked item is not a vertex/point/node."
                      << std::endl;
            return false;
        }
    }
    else {
        std::cerr << "The second picked structure is not a surface mesh, point "
                     "cloud, or curve network."
                  << std::endl;
        return false;
    }

    return true;
}

NODE_DECLARATION_UI(get_picked_vertices);
NODE_DEF_CLOSE_SCOPE
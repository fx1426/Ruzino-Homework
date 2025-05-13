// #define __GNUC__
#include "GCore/Components/CurveComponent.h"
#include "GCore/Components/MeshOperand.h"
#include "GCore/Components/PointsComponent.h"
#include "geom_node_base.h"
#include "nodes/core/socket.hpp"

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(create_grid)
{
    b.add_input<int>("resolution").min(1).max(20).default_val(2);
    b.add_input<float>("size").min(1).max(20);
    b.add_output<Geometry>("Geometry");
}

NODE_EXECUTION_FUNCTION(create_grid)
{
    int resolution = params.get_input<int>("resolution") + 1;
    float size = params.get_input<float>("size");
    Geometry geometry;
    std::shared_ptr<MeshComponent> mesh =
        std::make_shared<MeshComponent>(&geometry);
    geometry.attach_component(mesh);

    pxr::VtArray<pxr::GfVec3f> points;
    pxr::VtArray<pxr::GfVec2f> texcoord;
    pxr::VtArray<pxr::GfVec3f> normals;
    pxr::VtArray<int> faceVertexIndices;
    pxr::VtArray<int> faceVertexCounts;

    for (int i = 0; i < resolution; ++i) {
        for (int j = 0; j < resolution; ++j) {
            float y = size * static_cast<float>(i) / (resolution - 1);
            float z = size * static_cast<float>(j) / (resolution - 1);

            float u = static_cast<float>(i) / (resolution - 1);
            float v = static_cast<float>(j) / (resolution - 1);
            points.push_back(pxr::GfVec3f(0, y, z));
            texcoord.push_back(pxr::GfVec2f(u, v));
            normals.push_back(-pxr::GfVec3f(1.0f, 0.0f, 0.0f));
        }
    }

    for (int i = 0; i < resolution - 1; ++i) {
        for (int j = 0; j < resolution - 1; ++j) {
            faceVertexCounts.push_back(4);
            faceVertexIndices.push_back(i * resolution + j);
            faceVertexIndices.push_back(i * resolution + j + 1);
            faceVertexIndices.push_back((i + 1) * resolution + j + 1);
            faceVertexIndices.push_back((i + 1) * resolution + j);
        }
    }

    mesh->set_vertices(points);
    mesh->set_face_vertex_indices(faceVertexIndices);
    mesh->set_face_vertex_counts(faceVertexCounts);
    mesh->set_texcoords_array(texcoord);
    mesh->set_normals(normals);

    params.set_output("Geometry", std::move(geometry));
    return true;
}

NODE_DECLARATION_FUNCTION(create_circle)
{
    b.add_input<int>("resolution").min(1).max(100).default_val(10);
    b.add_input<float>("radius").min(1).max(20);
    b.add_output<Geometry>("Circle");
}

NODE_EXECUTION_FUNCTION(create_circle)
{
    int resolution = params.get_input<int>("resolution");
    float radius = params.get_input<float>("radius");
    Geometry geometry;
    std::shared_ptr<CurveComponent> curve =
        std::make_shared<CurveComponent>(&geometry);
    geometry.attach_component(curve);

    pxr::VtArray<pxr::GfVec3f> points;

    pxr::GfVec3f center(0.0f, 0.0f, 0.0f);

    float angleStep = 2.0f * M_PI / resolution;

    for (int i = 0; i < resolution; ++i) {
        float angle = i * angleStep;
        pxr::GfVec3f point(
            radius * std::cos(angle) + center[0],
            radius * std::sin(angle) + center[1],
            center[2]);
        points.push_back(point);
    }

    pxr::VtArray<pxr::GfVec3f> normals;

    for (int i = 0; i < resolution; ++i) {
        normals.push_back(pxr::GfVec3f(0.0f, 0.0f, 1.0f));
    }

    curve->set_vertices(points);
    curve->set_curve_normals(normals);
    curve->set_vert_count({ resolution });

    curve->set_periodic(true);

    params.set_output("Circle", std::move(geometry));
    return true;
}


NODE_DECLARATION_FUNCTION(create_cylinder_section)
{
    b.add_input<float>("height").min(0.1).max(20).default_val(1.0);
    b.add_input<float>("radius").min(0.1).max(20).default_val(1.0);
    b.add_input<int>("resolution").min(2).max(100).default_val(16);
    b.add_output<Geometry>("Geometry");
}

NODE_EXECUTION_FUNCTION(create_cylinder_section)
{
    float height = params.get_input<float>("height");
    float radius = params.get_input<float>("radius");
    int resolution = params.get_input<int>("resolution");

    Geometry geometry;
    std::shared_ptr<MeshComponent> mesh =
        std::make_shared<MeshComponent>(&geometry);
    geometry.attach_component(mesh);

    // Since height = arc length = radius * angle
    // angle = height / radius (in radians)
    float angle = height / radius;
    
    pxr::VtArray<pxr::GfVec3f> points;
    pxr::VtArray<pxr::GfVec3f> normals;
    pxr::VtArray<pxr::GfVec2f> texcoord;
    pxr::VtArray<int> faceVertexIndices;
    pxr::VtArray<int> faceVertexCounts;

    int rows = resolution;
    int cols = resolution;

    // Generate vertices
    for (int i = 0; i <= rows; ++i) {
        float v = static_cast<float>(i) / rows;
        float z = height * v;
        
        for (int j = 0; j <= cols; ++j) {
            float u = static_cast<float>(j) / cols;
            float theta = angle * u;
            
            // Calculate position
            float x = radius * std::cos(theta);
            float y = radius * std::sin(theta);
            
            points.push_back(pxr::GfVec3f(x, y, z));
            
            // Normal is pointing outward from the cylinder axis
            pxr::GfVec3f normal(x, y, 0);
            normal.Normalize();
            normals.push_back(normal);
            
            // UV coordinates: u along the arc, v along the height
            texcoord.push_back(pxr::GfVec2f(u, v));
        }
    }

    // Generate faces
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx0 = i * (cols + 1) + j;
            int idx1 = idx0 + 1;
            int idx2 = (i + 1) * (cols + 1) + j + 1;
            int idx3 = (i + 1) * (cols + 1) + j;
            
            faceVertexCounts.push_back(4);
            faceVertexIndices.push_back(idx0);
            faceVertexIndices.push_back(idx1);
            faceVertexIndices.push_back(idx2);
            faceVertexIndices.push_back(idx3);
        }
    }

    mesh->set_vertices(points);
    mesh->set_normals(normals);
    mesh->set_face_vertex_indices(faceVertexIndices);
    mesh->set_face_vertex_counts(faceVertexCounts);
    mesh->set_texcoords_array(texcoord);

    params.set_output("Geometry", std::move(geometry));
    return true;
}

NODE_DECLARATION_FUNCTION(create_spiral)
{
    b.add_input<int>("resolution").min(1).max(100).default_val(10);
    b.add_input<float>("R1").min(0.1).max(10).default_val(1);
    b.add_input<float>("R2").min(0.1).max(10).default_val(1);
    b.add_input<float>("Circle Count").min(0.1).max(10).default_val(2);
    b.add_input<float>("Height").min(0.1).max(10).default_val(1);
    b.add_output<Geometry>("Curve");
}

NODE_EXECUTION_FUNCTION(create_spiral)
{
    int resolution = params.get_input<int>("resolution");
    float R1 = params.get_input<float>("R1");
    float R2 = params.get_input<float>("R2");
    float circleCount = params.get_input<float>("Circle Count");
    float height = params.get_input<float>("Height");

    Geometry geometry;
    std::shared_ptr<CurveComponent> curve =
        std::make_shared<CurveComponent>(&geometry);
    geometry.attach_component(curve);

    pxr::VtArray<pxr::GfVec3f> points;

    float angleStep = circleCount * 2.0f * M_PI / resolution;
    float radiusIncrement = (R2 - R1) / resolution;
    float heightIncrement = height / resolution;

    for (int i = 0; i < resolution; ++i) {
        float angle = i * angleStep;
        float radius = R1 + radiusIncrement * i;
        float z = heightIncrement * i;
        pxr::GfVec3f point(
            radius * std::cos(angle), radius * std::sin(angle), z);
        points.push_back(point);
    }

    pxr::VtArray<pxr::GfVec3f> normals;

    for (int i = 0; i < resolution; ++i) {
        normals.push_back(pxr::GfVec3f(0.0f, 0.0f, 1.0f));
    }

    curve->set_vertices(points);
    curve->set_vert_count({ resolution });
    curve->set_curve_normals(normals);

    // Since a spiral is not periodic, we don't set a wrap attribute like we did
    // for the circle.

    params.set_output("Curve", std::move(geometry));
    return true;
}

NODE_DECLARATION_FUNCTION(create_uv_sphere)
{
    b.add_input<int>("segments").min(3).max(64).default_val(32);
    b.add_input<int>("rings").min(2).max(64).default_val(16);
    b.add_input<float>("radius").min(0.1).max(20).default_val(1.0);
    b.add_output<Geometry>("Geometry");
}

NODE_EXECUTION_FUNCTION(create_uv_sphere)
{
    int segments = params.get_input<int>("segments");
    int rings = params.get_input<int>("rings");
    float radius = params.get_input<float>("radius");

    Geometry geometry;
    std::shared_ptr<MeshComponent> mesh =
        std::make_shared<MeshComponent>(&geometry);
    geometry.attach_component(mesh);

    pxr::VtArray<pxr::GfVec3f> points;
    pxr::VtArray<pxr::GfVec3f> normals;
    pxr::VtArray<pxr::GfVec2f> texcoord;
    pxr::VtArray<int> faceVertexIndices;
    pxr::VtArray<int> faceVertexCounts;

    // Add top vertex
    points.push_back(pxr::GfVec3f(0, 0, radius));
    normals.push_back(pxr::GfVec3f(0, 0, 1));
    texcoord.push_back(pxr::GfVec2f(0.5f, 1.0f));

    // Generate vertices for each ring
    for (int ring = 0; ring < rings - 1; ++ring) {
        float phi = M_PI * (float)(ring + 1) / rings;
        float sinPhi = std::sin(phi);
        float cosPhi = std::cos(phi);
        float v = 1.0f - (float)(ring + 1) / rings;

        for (int segment = 0; segment < segments; ++segment) {
            float theta = 2.0f * M_PI * (float)segment / segments;
            float sinTheta = std::sin(theta);
            float cosTheta = std::cos(theta);
            float u = (float)segment / segments;

            float x = radius * sinPhi * cosTheta;
            float y = radius * sinPhi * sinTheta;
            float z = radius * cosPhi;

            points.push_back(pxr::GfVec3f(x, y, z));
            normals.push_back(pxr::GfVec3f(x / radius, y / radius, z / radius));
            texcoord.push_back(pxr::GfVec2f(u, v));
        }
    }

    // Add bottom vertex
    points.push_back(pxr::GfVec3f(0, 0, -radius));
    normals.push_back(pxr::GfVec3f(0, 0, -1));
    texcoord.push_back(pxr::GfVec2f(0.5f, 0.0f));

    // Add top cap faces
    for (int segment = 0; segment < segments; ++segment) {
        faceVertexCounts.push_back(3);
        faceVertexIndices.push_back(0);
        faceVertexIndices.push_back(1 + (segment + 1) % segments);
        faceVertexIndices.push_back(1 + segment);
    }

    // Add middle faces
    for (int ring = 0; ring < rings - 2; ++ring) {
        int ringStart = 1 + ring * segments;
        int nextRingStart = 1 + (ring + 1) * segments;
        for (int segment = 0; segment < segments; ++segment) {
            int nextSegment = (segment + 1) % segments;

            faceVertexCounts.push_back(4);
            faceVertexIndices.push_back(ringStart + segment);
            faceVertexIndices.push_back(ringStart + nextSegment);
            faceVertexIndices.push_back(nextRingStart + nextSegment);
            faceVertexIndices.push_back(nextRingStart + segment);
        }
    }

    // Add bottom cap faces
    int bottomVertex = points.size() - 1;
    int lastRingStart = 1 + (rings - 2) * segments;
    for (int segment = 0; segment < segments; ++segment) {
        faceVertexCounts.push_back(3);
        faceVertexIndices.push_back(bottomVertex);
        faceVertexIndices.push_back(lastRingStart + segment);
        faceVertexIndices.push_back(lastRingStart + (segment + 1) % segments);
    }

    assert(points.size() == normals.size());

    mesh->set_vertices(points);
    mesh->set_normals(normals);
    mesh->set_face_vertex_indices(faceVertexIndices);
    mesh->set_face_vertex_counts(faceVertexCounts);
    mesh->set_texcoords_array(texcoord);

    params.set_output("Geometry", std::move(geometry));
    return true;
}

NODE_DECLARATION_FUNCTION(create_ico_sphere)
{
    b.add_input<int>("subdivisions").min(0).max(5).default_val(2);
    b.add_input<float>("radius").min(0.1).max(20).default_val(1.0);
    b.add_output<Geometry>("Geometry");
}

NODE_EXECUTION_FUNCTION(create_ico_sphere)
{
    int subdivisions = params.get_input<int>("subdivisions");
    float radius = params.get_input<float>("radius");

    Geometry geometry;
    std::shared_ptr<MeshComponent> mesh =
        std::make_shared<MeshComponent>(&geometry);
    geometry.attach_component(mesh);

    std::vector<pxr::GfVec3f> vertices;
    std::vector<std::vector<int>> faces;

    // Create icosahedron base shape
    const float t = (1.0f + std::sqrt(5.0f)) / 2.0f;

    // Add initial 12 vertices of icosahedron
    vertices.push_back(pxr::GfVec3f(-1, t, 0).GetNormalized());
    vertices.push_back(pxr::GfVec3f(1, t, 0).GetNormalized());
    vertices.push_back(pxr::GfVec3f(-1, -t, 0).GetNormalized());
    vertices.push_back(pxr::GfVec3f(1, -t, 0).GetNormalized());

    vertices.push_back(pxr::GfVec3f(0, -1, t).GetNormalized());
    vertices.push_back(pxr::GfVec3f(0, 1, t).GetNormalized());
    vertices.push_back(pxr::GfVec3f(0, -1, -t).GetNormalized());
    vertices.push_back(pxr::GfVec3f(0, 1, -t).GetNormalized());

    vertices.push_back(pxr::GfVec3f(t, 0, -1).GetNormalized());
    vertices.push_back(pxr::GfVec3f(t, 0, 1).GetNormalized());
    vertices.push_back(pxr::GfVec3f(-t, 0, -1).GetNormalized());
    vertices.push_back(pxr::GfVec3f(-t, 0, 1).GetNormalized());

    // Add initial 20 faces
    faces.push_back({ 0, 11, 5 });
    faces.push_back({ 0, 5, 1 });
    faces.push_back({ 0, 1, 7 });
    faces.push_back({ 0, 7, 10 });
    faces.push_back({ 0, 10, 11 });

    faces.push_back({ 1, 5, 9 });
    faces.push_back({ 5, 11, 4 });
    faces.push_back({ 11, 10, 2 });
    faces.push_back({ 10, 7, 6 });
    faces.push_back({ 7, 1, 8 });

    faces.push_back({ 3, 9, 4 });
    faces.push_back({ 3, 4, 2 });
    faces.push_back({ 3, 2, 6 });
    faces.push_back({ 3, 6, 8 });
    faces.push_back({ 3, 8, 9 });

    faces.push_back({ 4, 9, 5 });
    faces.push_back({ 2, 4, 11 });
    faces.push_back({ 6, 2, 10 });
    faces.push_back({ 8, 6, 7 });
    faces.push_back({ 9, 8, 1 });

    // Helper function to get middle point of two vertices
    std::map<std::pair<int, int>, int> middlePointCache;
    auto getMiddlePoint = [&](int p1, int p2) -> int {
        // Ensure p1 <= p2 for consistent map keys
        bool swapped = false;
        if (p1 > p2) {
            std::swap(p1, p2);
            swapped = true;
        }

        std::pair<int, int> key(p1, p2);
        auto it = middlePointCache.find(key);
        if (it != middlePointCache.end()) {
            return it->second;
        }

        // Not in cache, calculate it
        pxr::GfVec3f middle = (vertices[p1] + vertices[p2]) * 0.5f;
        int i = vertices.size();
        vertices.push_back(middle.GetNormalized());
        middlePointCache[key] = i;
        return i;
    };

    // Perform subdivision
    for (int i = 0; i < subdivisions; i++) {
        std::vector<std::vector<int>> newFaces;
        for (const auto& face : faces) {
            int a = getMiddlePoint(face[0], face[1]);
            int b = getMiddlePoint(face[1], face[2]);
            int c = getMiddlePoint(face[2], face[0]);

            newFaces.push_back({ face[0], a, c });
            newFaces.push_back({ face[1], b, a });
            newFaces.push_back({ face[2], c, b });
            newFaces.push_back({ a, b, c });
        }
        faces = newFaces;
    }

    // Project vertices to sphere
    for (auto& v : vertices) {
        v *= radius;
    }

    // Convert to mesh format
    pxr::VtArray<pxr::GfVec3f> points(vertices.begin(), vertices.end());

    pxr::VtArray<int> faceVertexIndices;
    pxr::VtArray<int> faceVertexCounts;
    pxr::VtArray<pxr::GfVec3f> normals;
    pxr::VtArray<pxr::GfVec2f> texcoord;

    for (const auto& face : faces) {
        faceVertexCounts.push_back(face.size());
        for (int idx : face) {
            faceVertexIndices.push_back(idx);
        }
    }

    // Calculate normals (for a sphere, normals are just normalized positions)
    normals.resize(vertices.size());
    texcoord.resize(vertices.size());
    for (size_t i = 0; i < vertices.size(); i++) {
        pxr::GfVec3f normal = vertices[i].GetNormalized();
        normals[i] = normal;

        // Simple UV mapping based on spherical coordinates
        float u = 0.5f + std::atan2(normal[1], normal[0]) / (2.0f * M_PI);
        float v = 0.5f - std::asin(normal[2]) / M_PI;
        texcoord[i] = pxr::GfVec2f(u, v);
    }

    mesh->set_vertices(points);
    mesh->set_normals(normals);
    mesh->set_face_vertex_indices(faceVertexIndices);
    mesh->set_face_vertex_counts(faceVertexCounts);
    mesh->set_texcoords_array(texcoord);

    params.set_output("Geometry", std::move(geometry));
    return true;
}

NODE_DECLARATION_FUNCTION(create_point)
{
    b.add_input<float>("X").default_val(0.0f).min(-10.f).max(10.f);
    b.add_input<float>("Y").default_val(0.0f).min(-10.f).max(10.f);
    b.add_input<float>("Z").default_val(0.0f).min(-10.f).max(10.f);
    b.add_input<float>("Size").min(0.1f).max(10.0f).default_val(1.0f);
    b.add_output<Geometry>("Point");
}

NODE_EXECUTION_FUNCTION(create_point)
{
    float x = params.get_input<float>("X");
    float y = params.get_input<float>("Y");
    float z = params.get_input<float>("Z");
    float size = params.get_input<float>("Size");

    Geometry geometry;
    std::shared_ptr<PointsComponent> points =
        std::make_shared<PointsComponent>(&geometry);
    geometry.attach_component(points);

    pxr::VtArray<pxr::GfVec3f> vertices;
    vertices.push_back(pxr::GfVec3f(x, y, z));

    pxr::VtArray<float> widths;
    widths.push_back(size);

    points->set_vertices(vertices);
    points->set_normals({pxr::GfVec3f(0.0f, 0.0f, 1.0f)});
    points->set_width(widths);

    params.set_output("Point", std::move(geometry));
    return true;
}

NODE_DEF_CLOSE_SCOPE

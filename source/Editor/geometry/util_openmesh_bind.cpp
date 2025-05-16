#include "GCore/util_openmesh_bind.h"

#include "GCore/Components/MeshOperand.h"
USTC_CG_NAMESPACE_OPEN_SCOPE
std::shared_ptr<PolyMesh> operand_to_openmesh(Geometry* mesh_oeprand)
{
    auto openmesh = std::make_shared<PolyMesh>();
    auto topology = mesh_oeprand->get_component<MeshComponent>();

    // Get mesh data
    const auto& vertices = topology->get_vertices();
    for (const auto& vv : vertices) {
        OpenMesh::Vec3f v;
        v[0] = vv[0];
        v[1] = vv[1];
        v[2] = vv[2];
        openmesh->add_vertex(v);
    }

    auto faceVertexIndices = topology->get_face_vertex_indices();
    auto faceVertexCounts = topology->get_face_vertex_counts();
    auto normals = topology->get_normals();
    auto texcoords = topology->get_texcoords_array();
    auto colors = topology->get_display_color();

    bool hasNormals = !normals.empty();
    bool perVertexNormals = hasNormals && (normals.size() == vertices.size());
    bool hasTexcoords = !texcoords.empty();
    bool perVertexTexcoords =
        hasTexcoords && (texcoords.size() == vertices.size());
    bool hasColors = !colors.empty();
    bool perVertexColors = hasColors && (colors.size() == vertices.size());

    // Request vertex normals, texcoords, and colors only when they exist
    if (hasNormals)
        openmesh->request_vertex_normals();
    if (hasTexcoords)
        openmesh->request_vertex_texcoords2D();
    if (hasColors)
        openmesh->request_vertex_colors();

    int vertexIndex = 0;
    for (int i = 0; i < faceVertexCounts.size(); i++) {
        // Create a vector of vertex handles for the face
        std::vector<PolyMesh::VertexHandle> face_vhandles;
        for (int j = 0; j < faceVertexCounts[i]; j++) {
            int index = faceVertexIndices[vertexIndex];
            // Get the vertex handle from the index
            PolyMesh::VertexHandle vh = openmesh->vertex_handle(index);
            // Add it to the vector
            face_vhandles.push_back(vh);

            // Set normal if available
            if (hasNormals) {
                if (perVertexNormals) {
                    // Use per-vertex normals
                    OpenMesh::Vec3f n(
                        normals[index][0],
                        normals[index][1],
                        normals[index][2]);
                    openmesh->set_normal(vh, n);
                }
                else {
                    // Use per-face-vertex normals
                    OpenMesh::Vec3f n(
                        normals[vertexIndex][0],
                        normals[vertexIndex][1],
                        normals[vertexIndex][2]);
                    openmesh->set_normal(vh, n);
                }
            }

            // Set texcoords if available
            if (hasTexcoords) {
                if (perVertexTexcoords) {
                    // Use per-vertex texcoords
                    OpenMesh::Vec2f t(texcoords[index][0], texcoords[index][1]);
                    openmesh->set_texcoord2D(vh, t);
                }
                else {
                    // Use per-face-vertex texcoords
                    OpenMesh::Vec2f t(
                        texcoords[vertexIndex][0], texcoords[vertexIndex][1]);
                    openmesh->set_texcoord2D(vh, t);
                }
            }

            // Set colors if available
            if (hasColors) {
                if (perVertexColors) {
                    // Use per-vertex colors
                    OpenMesh::Vec3f c(
                        colors[index][0], colors[index][1], colors[index][2]);
                    openmesh->set_color(
                        vh,
                        PolyMesh::Color(
                            static_cast<unsigned char>(c[0] * 255),
                            static_cast<unsigned char>(c[1] * 255),
                            static_cast<unsigned char>(c[2] * 255)));
                }
                else {
                    // Use per-face-vertex colors
                    OpenMesh::Vec3f c(
                        colors[vertexIndex][0],
                        colors[vertexIndex][1],
                        colors[vertexIndex][2]);
                    openmesh->set_color(
                        vh,
                        PolyMesh::Color(
                            static_cast<unsigned char>(c[0] * 255),
                            static_cast<unsigned char>(c[1] * 255),
                            static_cast<unsigned char>(c[2] * 255)));
                }
            }

            vertexIndex++;
        }
        // Add the face to the mesh
        openmesh->add_face(face_vhandles);
    }

    if (hasNormals) {
        // Update or compute vertex normals if necessary
        openmesh->update_normals();
    }

    return openmesh;
}

std::shared_ptr<Geometry> openmesh_to_operand(PolyMesh* openmesh)
{
    // TODO: test
    auto geometry = std::make_shared<Geometry>();
    std::shared_ptr<MeshComponent> mesh =
        std::make_shared<MeshComponent>(geometry.get());
    geometry->attach_component(mesh);

    pxr::VtArray<pxr::GfVec3f> points;
    pxr::VtArray<int> faceVertexIndices;
    pxr::VtArray<int> faceVertexCounts;
    pxr::VtArray<pxr::GfVec3f> normals;
    pxr::VtArray<pxr::GfVec2f> texcoords;
    pxr::VtArray<pxr::GfVec3f> colors;

    bool hasNormals = openmesh->has_vertex_normals();
    bool hasTexcoords = openmesh->has_vertex_texcoords2D();
    bool hasColors = openmesh->has_vertex_colors();

    if (hasNormals && !openmesh->has_vertex_normals())
        openmesh->request_vertex_normals();
    if (hasTexcoords && !openmesh->has_vertex_texcoords2D())
        openmesh->request_vertex_texcoords2D();
    if (hasColors && !openmesh->has_vertex_colors())
        openmesh->request_vertex_colors();

    // Ensure normals are updated
    if (hasNormals)
        openmesh->update_normals();

    // Set the points
    for (const auto& v : openmesh->vertices()) {
        const auto& p = openmesh->point(v);
        points.push_back(pxr::GfVec3f(p[0], p[1], p[2]));

        // Add per-vertex normal if available
        if (hasNormals) {
            const auto& n = openmesh->normal(v);
            normals.push_back(pxr::GfVec3f(n[0], n[1], n[2]));
        }

        // Add per-vertex texcoord if available
        if (hasTexcoords) {
            const auto& t = openmesh->texcoord2D(v);
            texcoords.push_back(pxr::GfVec2f(t[0], t[1]));
        }

        // Add per-vertex color if available
        if (hasColors) {
            const auto& c = openmesh->color(v);
            colors.push_back(pxr::GfVec3f(c[0], c[1], c[2]));
        }
    }

    // Set the topology
    for (const auto& f : openmesh->faces()) {
        size_t count = 0;
        for (const auto& vf : f.vertices()) {
            faceVertexIndices.push_back(vf.idx());
            count += 1;
        }
        faceVertexCounts.push_back(count);
    }

    mesh->set_vertices(points);
    mesh->set_face_vertex_indices(faceVertexIndices);
    mesh->set_face_vertex_counts(faceVertexCounts);

    if (hasNormals) {
        mesh->set_normals(normals);
    }

    if (hasTexcoords) {
        mesh->set_texcoords_array(texcoords);
    }

    if (hasColors) {
        mesh->set_display_color(colors);
    }

    return geometry;
}

USTC_CG_NAMESPACE_CLOSE_SCOPE

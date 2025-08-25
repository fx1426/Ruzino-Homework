#include "GCore/Components/CurveComponent.h"
#include "GCore/Components/MeshComponent.h"
#include "geom_node_base.h"
#include "pxr/base/gf/matrix3f.h"
#include "pxr/base/gf/rotation.h"

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(curve_to_mesh)
{
    b.add_input<Geometry>("Curve");
    b.add_input<Geometry>("Profile Curve");

    b.add_output<Geometry>("Mesh");
}
NODE_EXECUTION_FUNCTION(curve_to_mesh)
{
    Geometry mesh_geom = Geometry::CreateMesh();

    auto curve_input = params.get_input<Geometry>("Curve");
    auto profile_curve_input = params.get_input<Geometry>("Profile Curve");

    auto curve = curve_input.get_component<CurveComponent>();
    auto profile_curve = profile_curve_input.get_component<CurveComponent>();

    // Suppose both curve are linear
    using namespace pxr;

    // Calculate the transformation

    // The curve must have a normal.
    std::vector<glm::vec3> curve_normals = curve->get_curve_normals();
    // Only rotation is needed here.

    auto guide_curve_verts = curve->get_vertices();
    auto profile_curve_verts = profile_curve->get_vertices();

    bool guide_curve_periodic = curve->get_periodic();

    auto vert_count = guide_curve_verts.size();

    auto mesh = mesh_geom.get_component<MeshComponent>();

    std::vector<glm::vec2> texcoords_array;
    std::vector<int> face_vertex_counts;
    std::vector<glm::vec3> vertices;
    std::vector<int> face_vertex_indices;
    std::vector<glm::vec3> normals;

    for (int i = 0; i < guide_curve_verts.size(); ++i) {
        glm::vec3 normal = normalize(curve_normals[i]);
        glm::vec3 tangent;

        int this_vert_id = i;
        int next_vert_id = (i + 1) % vert_count;
        int prev_vert_id = (i - 1) % vert_count;

        bool first_of_periodic = guide_curve_periodic && i == 0;

        bool last_of_periodic =
            guide_curve_periodic && i == guide_curve_verts.size() - 1;

        bool first_of_nonperiodic = !guide_curve_periodic && i == 0;

        bool last_of_nonperiodic =
            !guide_curve_periodic && i == guide_curve_verts.size() - 1;

        if (first_of_nonperiodic) {
            tangent = guide_curve_verts[1] - guide_curve_verts[0];
        }
        else if (last_of_nonperiodic) {
            tangent = guide_curve_verts[guide_curve_verts.size() - 1] -
                      guide_curve_verts[guide_curve_verts.size() - 2];
        }
        else {
            auto vec1 = guide_curve_verts[next_vert_id] -
                        guide_curve_verts[this_vert_id];
            auto vec2 = guide_curve_verts[this_vert_id] -
                        guide_curve_verts[prev_vert_id];

            auto l1 = length(vec1);
            auto l2 = length(vec2);
            vec1 = normalize(vec1);
            vec2 = normalize(vec2);
            auto weight_1 = l2 / (l1 + l2);

            tangent = weight_1 * vec2 + (1 - weight_1) * vec1;
        }
        tangent = normalize(tangent);
        auto bitangent = cross(normal, tangent);

        glm::mat3 tbn;

        tbn[2] = tangent;
        tbn[1] = bitangent;
        tbn[0] = normal;

        for (int j = 0; j < profile_curve_verts.size(); ++j) {
            auto new_pos = tbn * profile_curve_verts[j] + guide_curve_verts[i];
            vertices.push_back(new_pos);

            // Add normal for this vertex (transform profile normal by tbn)
            // Using profile position as approximation of normal direction
            glm::vec3 profile_normal = normalize(profile_curve_verts[j]);
            glm::vec3 transformed_normal = tbn * profile_normal;
            transformed_normal = normalize(transformed_normal);
            normals.push_back(transformed_normal);
        }
        // Removed per-iteration face count; faces will be built after vertex
        // generation.
    }

    // Build face indices and face counts from the swept vertices.
    int profileCount = profile_curve_verts.size();
    int rings = guide_curve_verts.size();
    for (int i = 0; i < rings - 1; i++) {
        int ring1Index = i * profileCount;
        int ring2Index = (i + 1) * profileCount;
        for (int j = 0; j < profileCount; j++) {
            int next_j = (j + 1) % profileCount;
            face_vertex_indices.push_back(ring1Index + j);
            face_vertex_indices.push_back(ring2Index + j);
            face_vertex_indices.push_back(ring2Index + next_j);
            face_vertex_indices.push_back(ring1Index + next_j);
            face_vertex_counts.push_back(4);
        }
    }
    mesh->set_vertices(vertices);
    mesh->set_face_vertex_counts(face_vertex_counts);
    mesh->set_face_vertex_indices(face_vertex_indices);
    mesh->set_normals(normals);  // Set the calculated normals
    mesh->set_texcoords_array(texcoords_array);

    params.set_output("Mesh", mesh_geom);
    return true;
}

NODE_DECLARATION_UI(curve_to_mesh);
NODE_DEF_CLOSE_SCOPE

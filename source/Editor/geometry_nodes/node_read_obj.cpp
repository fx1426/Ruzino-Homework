#include <pxr/base/gf/vec3f.h>
#include <pxr/base/vt/array.h>

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <filesystem>
#include <iostream>

#include "GCore/Components/MeshOperand.h"
#include "igl/readOBJ.h"
#include "nodes/core/def/node_def.hpp"

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(read_obj_std)
{
    b.add_input<std::string>("Path").default_val("Default");
    b.add_output<std::vector<std::vector<float>>>("Vertices");
    b.add_output<std::vector<std::vector<float>>>("Texture Coordinates");
    b.add_output<std::vector<std::vector<float>>>("Normals");
    b.add_output<std::vector<std::vector<int>>>("Faces");
    b.add_output<std::vector<std::vector<int>>>("Face Texture Coordinates");
    b.add_output<std::vector<std::vector<int>>>("Face Normals");
    // Function content omitted
}

NODE_EXECUTION_FUNCTION(read_obj_std)
{
    std::filesystem::path executable_path;

#ifdef _WIN32
    char p[MAX_PATH];
    GetModuleFileNameA(NULL, p, MAX_PATH);
    executable_path = std::filesystem::path(p).parent_path();
#else
    char p[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", p, PATH_MAX);
    if (count != -1) {
        p[count] = '\0';
        executable_path = std::filesystem::path(path).parent_path();
    }
    else {
        throw std::runtime_error("Failed to get executable path.");
    }
#endif

    auto path_str = params.get_input<std::string>("Path");
    std::filesystem::path abs_path;
    if (!path_str.empty()) {
        abs_path = std::filesystem::path(path_str);
    }
    else {
        std::cerr << "Path is empty." << std::endl;
        return false;
    }
    if (!abs_path.is_absolute()) {
        abs_path = executable_path / abs_path;
    }
    abs_path = abs_path.lexically_normal();
    std::vector<std::vector<float>> V;
    std::vector<std::vector<float>> TC;
    std::vector<std::vector<float>> N;
    std::vector<std::vector<int>> F;
    std::vector<std::vector<int>> FTC;
    std::vector<std::vector<int>> FN;
    // Function content omitted
    auto success = igl::readOBJ(abs_path.string(), V, TC, N, F, FTC, FN);

    if (success) {
        params.set_output("Vertices", std::move(V));
        params.set_output("Texture Coordinates", std::move(TC));
        params.set_output("Normals", std::move(N));
        params.set_output("Faces", std::move(F));
        params.set_output("Face Texture Coordinates", std::move(FTC));
        params.set_output("Face Normals", std::move(FN));
        return true;
    }
    else {
        return false;
    }
}

NODE_DECLARATION_UI(read_obj_std);

NODE_DECLARATION_FUNCTION(read_obj_eigen)
{
    b.add_input<std::string>("Path").default_val("Default");
    b.add_output<Eigen::MatrixXf>("Vertices");
    b.add_output<Eigen::MatrixXf>("Texture Coordinates");
    b.add_output<Eigen::MatrixXf>("Normals");
    b.add_output<Eigen::MatrixXi>("Faces");
    b.add_output<Eigen::MatrixXi>("Face Texture Coordinates");
    b.add_output<Eigen::MatrixXi>("Face Normals");
    // Function content omitted
}

NODE_EXECUTION_FUNCTION(read_obj_eigen)
{
    std::filesystem::path executable_path;

#ifdef _WIN32
    char p[MAX_PATH];
    GetModuleFileNameA(NULL, p, MAX_PATH);
    executable_path = std::filesystem::path(p).parent_path();
#else
    char p[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", p, PATH_MAX);
    if (count != -1) {
        p[count] = '\0';
        executable_path = std::filesystem::path(path).parent_path();
    }
    else {
        throw std::runtime_error("Failed to get executable path.");
    }
#endif

    auto path_str = params.get_input<std::string>("Path");
    std::filesystem::path abs_path;
    if (!path_str.empty()) {
        abs_path = std::filesystem::path(path_str);
    }
    else {
        std::cerr << "Path is empty." << std::endl;
        return false;
    }
    if (!abs_path.is_absolute()) {
        abs_path = executable_path / abs_path;
    }
    abs_path = abs_path.lexically_normal();
    Eigen::MatrixXf V;
    Eigen::MatrixXf TC;
    Eigen::MatrixXf N;
    Eigen::MatrixXi F;
    Eigen::MatrixXi FTC;
    Eigen::MatrixXi FN;
    // Function content omitted
    auto success = igl::readOBJ(abs_path.string(), V, TC, N, F, FTC, FN);

    if (success) {
        params.set_output("Vertices", std::move(V));
        params.set_output("Texture Coordinates", std::move(TC));
        params.set_output("Normals", std::move(N));
        params.set_output("Faces", std::move(F));
        params.set_output("Face Texture Coordinates", std::move(FTC));
        params.set_output("Face Normals", std::move(FN));
        return true;
    }
    else {
        return false;
    }
}

NODE_DECLARATION_UI(read_obj_eigen);

// 定义一个简单的网格类型
typedef OpenMesh::TriMesh_ArrayKernelT<> MyMesh;

NODE_DECLARATION_FUNCTION(read_obj_pxr)
{
    b.add_input<std::string>("Path").default_val("Default");
    b.add_output<pxr::VtVec3fArray>("Vertices");
    b.add_output<pxr::VtArray<int>>("FaceVertexCounts");
    b.add_output<pxr::VtArray<int>>("FaceVertexIndices");
    b.add_output<pxr::VtArray<pxr::GfVec3f>>("Normals");
    b.add_output<pxr::VtArray<pxr::GfVec2f>>("Texcoords");
}

NODE_EXECUTION_FUNCTION(read_obj_pxr)
{
    std::filesystem::path executable_path;

#ifdef _WIN32
    char p[MAX_PATH];
    GetModuleFileNameA(NULL, p, MAX_PATH);
    executable_path = std::filesystem::path(p).parent_path();
#else
    char p[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", p, PATH_MAX);
    if (count != -1) {
        p[count] = '\0';
        executable_path = std::filesystem::path(path).parent_path();
    }
    else {
        throw std::runtime_error("Failed to get executable path.");
    }
#endif

    auto path_str = params.get_input<std::string>("Path");
    std::filesystem::path abs_path;
    if (!path_str.empty()) {
        abs_path = std::filesystem::path(path_str);
    }
    else {
        std::cerr << "Path is empty." << std::endl;
        return false;
    }
    if (!abs_path.is_absolute()) {
        abs_path = executable_path / abs_path;
    }
    abs_path = abs_path.lexically_normal();

    MyMesh mesh;
    OpenMesh::IO::Options opt;
    opt += OpenMesh::IO::Options::VertexTexCoord;
    opt += OpenMesh::IO::Options::VertexNormal;

    // 尝试读取OBJ文件
    if (!OpenMesh::IO::read_mesh(
            mesh, std::string(abs_path.string().c_str()), opt)) {
        return false;
    }

    // 请求法线和纹理坐标
    if (opt.check(OpenMesh::IO::Options::VertexNormal)) {
        mesh.request_vertex_normals();
        if (!mesh.has_vertex_normals()) {
            mesh.update_normals();
        }
    }
    mesh.request_vertex_texcoords2D();

    // 转换顶点数据
    pxr::VtVec3fArray vertices(mesh.n_vertices());
    int vertex_idx = 0;
    for (auto vh : mesh.vertices()) {
        auto point = mesh.point(vh);
        vertices[vertex_idx] = pxr::GfVec3f(point[0], point[1], point[2]);
        vertex_idx++;
    }

    // 转换面数据
    const int vertices_per_face = 3;  // 三角形网格
    const int num_faces = mesh.n_faces();
    pxr::VtArray<int> faceVertexCounts(num_faces, vertices_per_face);
    pxr::VtArray<int> faceVertexIndices(num_faces * vertices_per_face);

    int face_idx = 0;
    for (auto fh : mesh.faces()) {
        int vertex_count = 0;
        for (auto fv_it = mesh.fv_iter(fh); fv_it.is_valid(); ++fv_it) {
            faceVertexIndices[face_idx * vertices_per_face + vertex_count] =
                fv_it->idx();
            vertex_count++;
        }
        face_idx++;
    }

    // 转换法线数据
    pxr::VtArray<pxr::GfVec3f> normals;
    if (mesh.has_vertex_normals()) {
        normals.resize(mesh.n_vertices());
        int normal_idx = 0;
        for (auto vh : mesh.vertices()) {
            auto normal = mesh.normal(vh);
            normals[normal_idx] = pxr::GfVec3f(normal[0], normal[1], normal[2]);
            normal_idx++;
        }
    }

    // 转换纹理坐标数据
    pxr::VtArray<pxr::GfVec2f> texcoords;
    if (mesh.has_vertex_texcoords2D()) {
        texcoords.resize(mesh.n_vertices());
        int texcoord_idx = 0;
        for (auto vh : mesh.vertices()) {
            auto texcoord = mesh.texcoord2D(vh);
            texcoords[texcoord_idx] = pxr::GfVec2f(texcoord[0], texcoord[1]);
            texcoord_idx++;
        }
    }

    // 设置输出
    params.set_output("Vertices", std::move(vertices));
    params.set_output("FaceVertexCounts", std::move(faceVertexCounts));
    params.set_output("FaceVertexIndices", std::move(faceVertexIndices));
    params.set_output("Normals", std::move(normals));
    params.set_output("Texcoords", std::move(texcoords));

    // 清理请求的属性
    mesh.release_vertex_normals();
    mesh.release_vertex_texcoords2D();

    return true;
}

NODE_DECLARATION_UI(read_obj_pxr);

NODE_DEF_CLOSE_SCOPE

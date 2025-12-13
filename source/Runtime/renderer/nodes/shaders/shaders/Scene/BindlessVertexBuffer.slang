
import Scene.SceneTypes;
import Scene.Scene;

[[vk::binding(0, 1)]]
ByteAddressBuffer t_BindlessBuffers[] : register(t0, space1);

StructuredBuffer<GeometryInstanceData> instanceDescBuffer;
StructuredBuffer<MeshDesc> meshDescBuffer;

StaticVertexData load_vertex(MeshDesc mesh_data, uint vertex_id)
{
    var vb_offset = mesh_data.vbOffset;
    var normal_offset = mesh_data.normalOffset;
    var bindlessIndex = mesh_data.bindlessIndex;

    // Add memory barrier before accessing non-uniform resource
    // This is critical for AMD GPU cache coherency
    DeviceMemoryBarrier();

    ByteAddressBuffer vertex_buffer =
        t_BindlessBuffers[NonUniformResourceIndex(bindlessIndex)];
    var id = vertex_id;
    if (mesh_data.ibOffset > 0) {
        vertex_id = vertex_buffer.Load(mesh_data.ibOffset + vertex_id * 4);
    }

    float3 pos = asfloat(vertex_buffer.Load3(3 * vertex_id * 4 + vb_offset));

    float3 normal;

    if (mesh_data.normalInterpolation == InterpolationType::FaceVarying) {
        normal = asfloat(vertex_buffer.Load3(3 * id * 4 + normal_offset));
    }
    else {
        normal =
            asfloat(vertex_buffer.Load3(3 * vertex_id * 4 + normal_offset));
    }

    if (normal_offset == 0)
        normal = float3(0, 1, 0);

    float4 tangent = float4(1, 0, 0, 1);
    if (mesh_data.tangentOffset != 0) {
        if (mesh_data.tangentInterpolation == InterpolationType::FaceVarying) {
            tangent = asfloat(
                vertex_buffer.Load4(4 * id * 4 + mesh_data.tangentOffset));
        }
        else {
            tangent = asfloat(vertex_buffer.Load4(
                4 * vertex_id * 4 + mesh_data.tangentOffset));
        }
    }

    float2 texCrd;

    if (mesh_data.texCrdInterpolation == InterpolationType::FaceVarying) {
        texCrd =
            asfloat(vertex_buffer.Load2(2 * id * 4 + mesh_data.texCrdOffset));
    }
    else
        texCrd = asfloat(
            vertex_buffer.Load2(2 * vertex_id * 4 + mesh_data.texCrdOffset));

    uint subset_material_id = -1;
    if (mesh_data.subsetMatIdOffset != 0) {
        subset_material_id =
            vertex_buffer.Load(mesh_data.subsetMatIdOffset + id * 4);
    }

    StaticVertexData data =
        StaticVertexData(pos, normal, tangent, texCrd, 0, subset_material_id);
    return data;
}

VertexInfo get_interpolated_vertex(
    uint instance_id,
    uint primitive_id,
    float2 baricentric2)
{
    // Ensure memory coherency for AMD GPUs when accessing structured buffers
    // DeviceMemoryBarrier is valid in ray tracing shaders
    DeviceMemoryBarrier();

    GeometryInstanceData instance = instanceDescBuffer[instance_id];
    MeshDesc mesh = meshDescBuffer[instance.geometryID];

    uint i0 = primitive_id * 3;
    uint i1 = primitive_id * 3 + 1;
    uint i2 = primitive_id * 3 + 2;

    StaticVertexData v[3];

    v[0] = load_vertex(mesh, i0);
    v[1] = load_vertex(mesh, i1);
    v[2] = load_vertex(mesh, i2);

    float4x4 worldMat = transpose(instance.transform);
    float3x3 worldInvTransposeMat = transpose(inverse(float3x3(worldMat)));

    float3 baricentric = float3(
        1.0 - baricentric2.x - baricentric2.y, baricentric2.x, baricentric2.y);
    var data = computeVertexData(
        v, worldMat, worldInvTransposeMat, false, baricentric);

    data.subsetMaterialID = v[0].subset_material_id;
    return data;
}

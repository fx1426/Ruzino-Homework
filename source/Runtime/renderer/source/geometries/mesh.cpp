//
// Copyright 2020 Pixar
//
// Licensed under the Apache License, Version 2.0 (the "Apache License")
// with the following modification; you may not use this file except in
// compliance with the Apache License and the following modification to it:
// Section 6. Trademarks. is deleted and replaced with:
//
// 6. Trademarks. This License does not grant permission to use the trade
//    names, trademarks, service marks, or product names of the Licensor
//    and its affiliates, except as required to comply with Section 4(c) of
//    the License and to reproduce the content of the NOTICE file.
//
// You may obtain a copy of the Apache License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the Apache License with the above modification is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the Apache License for the specific
// language governing permissions and limitations under the Apache License.
//
// #define __GNUC__
#include "mesh.h"

#include "../instancer.h"
#include "../renderParam.h"
#include "Logger/Logger.h"
#include "Scene/SceneTypes.slang"
#include "material/material.h"
#include "nvrhi/utils.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/imaging/hd/extComputationUtils.h"
#include "pxr/imaging/hd/instancer.h"
#include "pxr/imaging/hd/meshUtil.h"
#include "pxr/imaging/hd/smoothNormals.h"

USTC_CG_NAMESPACE_OPEN_SCOPE
class Hd_USTC_CG_RenderParam;
using namespace pxr;
Hd_USTC_CG_Mesh::Hd_USTC_CG_Mesh(const SdfPath& id)
    : HdMesh(id),
      _cullStyle(HdCullStyleDontCare),
      _doubleSided(false),
      _normalsValid(false),
      _adjacencyValid(false),
      _refined(false)
{
    // create model buffer (constant buffer, CPU writable)
    auto device = RHI::get_device();
    nvrhi::BufferDesc buffer_desc =
        nvrhi::BufferDesc{}
            .setByteSize(sizeof(GfMatrix4f))
            .setStructStride(sizeof(GfMatrix4f))
            .setInitialState(nvrhi::ResourceStates::ShaderResource)
            .setCpuAccess(nvrhi::CpuAccessMode::Write)
            .setDebugName("modelBuffer");
}

Hd_USTC_CG_Mesh::~Hd_USTC_CG_Mesh()
{
}

HdDirtyBits Hd_USTC_CG_Mesh::GetInitialDirtyBitsMask() const
{
    int mask =
        HdChangeTracker::Clean | HdChangeTracker::InitRepr |
        HdChangeTracker::DirtyPoints | HdChangeTracker::DirtyTopology |
        HdChangeTracker::DirtyTransform | HdChangeTracker::DirtyVisibility |
        HdChangeTracker::DirtyCullStyle | HdChangeTracker::DirtyDoubleSided |
        HdChangeTracker::DirtyDisplayStyle | HdChangeTracker::DirtySubdivTags |
        HdChangeTracker::DirtyPrimvar | HdChangeTracker::DirtyNormals |
        HdChangeTracker::DirtyInstancer | HdChangeTracker::DirtyMaterialId;

    return (HdDirtyBits)mask;
}

HdDirtyBits Hd_USTC_CG_Mesh::_PropagateDirtyBits(HdDirtyBits bits) const
{
    return bits;
}

TfTokenVector Hd_USTC_CG_Mesh::_UpdateComputedPrimvarSources(
    HdSceneDelegate* sceneDelegate,
    HdDirtyBits dirtyBits)
{
    HD_TRACE_FUNCTION();

    const SdfPath& id = GetId();

    // Get all the dirty computed primvars
    HdExtComputationPrimvarDescriptorVector dirtyCompPrimvars;
    for (size_t i = 0; i < HdInterpolationCount; ++i) {
        HdExtComputationPrimvarDescriptorVector compPrimvars;
        auto interp = static_cast<HdInterpolation>(i);
        compPrimvars =
            sceneDelegate->GetExtComputationPrimvarDescriptors(GetId(), interp);

        for (const auto& pv : compPrimvars) {
            if (HdChangeTracker::IsPrimvarDirty(dirtyBits, id, pv.name)) {
                dirtyCompPrimvars.emplace_back(pv);
            }
        }
    }

    if (dirtyCompPrimvars.empty()) {
        return TfTokenVector();
    }

    HdExtComputationUtils::ValueStore valueStore =
        HdExtComputationUtils::GetComputedPrimvarValues(
            dirtyCompPrimvars, sceneDelegate);

    TfTokenVector compPrimvarNames;
    // Update local primvar map and track the ones that were computed
    for (const auto& compPrimvar : dirtyCompPrimvars) {
        const auto it = valueStore.find(compPrimvar.name);
        if (!TF_VERIFY(it != valueStore.end())) {
            continue;
        }

        compPrimvarNames.emplace_back(compPrimvar.name);
        _primvarSourceMap[compPrimvar.name] = { it->second,
                                                compPrimvar.interpolation };
    }

    return compPrimvarNames;
}

void Hd_USTC_CG_Mesh::_UpdatePrimvarSources(
    HdSceneDelegate* sceneDelegate,
    HdDirtyBits dirtyBits,
    HdRenderParam* param)
{
    HD_TRACE_FUNCTION();
    const SdfPath& id = GetId();

    HdPrimvarDescriptorVector primvars;
    for (size_t i = 0; i < HdInterpolationCount; ++i) {
        auto interp = static_cast<HdInterpolation>(i);
        primvars = GetPrimvarDescriptors(sceneDelegate, interp);
        for (const HdPrimvarDescriptor& pv : primvars) {
            log::info("Checking primvar %s", pv.name.GetText());

            if (HdChangeTracker::IsPrimvarDirty(dirtyBits, id, pv.name) &&
                pv.name != HdTokens->points) {
                log::info("primvar %s is dirty", pv.name.GetText());
                _primvarSourceMap[pv.name] = {
                    GetPrimvar(sceneDelegate, pv.name), interp
                };
            }
        }
    }
}

void Hd_USTC_CG_Mesh::create_gpu_resources(Hd_USTC_CG_RenderParam* render_param)
{
    auto device = RHI::get_device();

    auto copy_commandlist =
        device->createCommandList({ .enableImmediateExecution = false });

    auto descriptor_table =
        render_param->InstanceCollection->get_buffer_descriptor_table();

    unsigned index_buffer_offset = 0;
    unsigned normal_buffer_offset = 0;
    unsigned tangent_buffer_offset = 0;
    unsigned texcoord_buffer_offset = 0;
    unsigned subset_mat_id_offset = 0;

    unsigned total_buffer_size = points.size() * 3 * sizeof(float);
    index_buffer_offset = total_buffer_size;
    total_buffer_size += triangulatedIndices.size() * 3 * sizeof(uint);
    normal_buffer_offset = total_buffer_size;
    total_buffer_size += normals.size() * 3 * sizeof(float);
    texcoord_buffer_offset = total_buffer_size;

    VtVec2fArray texcoords;

    InterpolationType texCrdInterpolation = InterpolationType::Vertex;
    for (auto pv : _primvarSourceMap) {
        if (pv.first == pxr::TfToken("UVMap") ||
            pv.first == pxr::TfToken("st")) {
            texcoords = pv.second.data.Get<VtVec2fArray>();
            if (pv.second.interpolation == HdInterpolationFaceVarying) {
                texCrdInterpolation = InterpolationType::FaceVarying;
            }
            else {
                texCrdInterpolation = InterpolationType::Vertex;
            }
        }
    }

    total_buffer_size += texcoords.size() * 2 * sizeof(float);

    VtArray<unsigned> subset_material_id;

    if (_primvarSourceMap.find(pxr::TfToken("subset_material_id")) !=
        _primvarSourceMap.end()) {
        subset_mat_id_offset = total_buffer_size;
        subset_material_id =
            _primvarSourceMap[pxr::TfToken("subset_material_id")]
                .data.Get<VtArray<unsigned>>();
        total_buffer_size +=
            _primvarSourceMap[pxr::TfToken("subset_material_id")]
                .data.Get<VtArray<unsigned>>()
                .size() *
            sizeof(int);
    }

    nvrhi::BufferDesc desc =
        nvrhi::BufferDesc{}
            .setCanHaveRawViews(true)
            .setByteSize(total_buffer_size)
            .setIsVertexBuffer(true)
            .setInitialState(nvrhi::ResourceStates::ShaderResource)
            .setCpuAccess(nvrhi::CpuAccessMode::None)
            .setIsAccelStructBuildInput(true)
            .setKeepInitialState(true)
            .setDebugName("vertexBuffer");

    vertexBuffer = device->createBuffer(desc);

    copy_commandlist->open();

    copy_commandlist->writeBuffer(
        vertexBuffer, points.data(), points.size() * 3 * sizeof(float), 0);

    copy_commandlist->writeBuffer(
        vertexBuffer,
        triangulatedIndices.data(),
        triangulatedIndices.size() * 3 * sizeof(uint),
        index_buffer_offset);

    if (!normals.empty()) {
        copy_commandlist->writeBuffer(
            vertexBuffer,
            normals.data(),
            normals.size() * 3 * sizeof(float),
            normal_buffer_offset);
    }

    if (!texcoords.empty()) {
        copy_commandlist->writeBuffer(
            vertexBuffer,
            texcoords.data(),
            texcoords.size() * 2 * sizeof(float),
            texcoord_buffer_offset);
    }
    if (!subset_material_id.empty()) {
        copy_commandlist->writeBuffer(
            vertexBuffer,
            subset_material_id.data(),
            subset_material_id.size() * sizeof(int),
            subset_mat_id_offset);
    }

    copy_commandlist->close();

    {
        std::lock_guard lock(execution_launch_mutex);
        device->executeCommandList(copy_commandlist);

        nvrhi::rt::AccelStructDesc blas_desc;
        nvrhi::rt::GeometryDesc geometry_desc;
        geometry_desc.geometryType = nvrhi::rt::GeometryType::Triangles;
        nvrhi::rt::GeometryTriangles triangles;
        triangles.setVertexBuffer(vertexBuffer)
            .setVertexOffset(0)
            .setIndexBuffer(vertexBuffer)
            .setIndexOffset(index_buffer_offset)
            .setIndexCount(triangulatedIndices.size() * 3)
            .setVertexCount(points.size())
            .setVertexStride(3 * sizeof(float))
            .setVertexFormat(nvrhi::Format::RGB32_FLOAT)
            .setIndexFormat(nvrhi::Format::R32_UINT);
        geometry_desc.setTriangles(triangles);
        blas_desc.addBottomLevelGeometry(geometry_desc);
        blas_desc.isTopLevel = false;
        BLAS = device->createAccelStruct(blas_desc);

        auto m_CommandList = device->createCommandList();

        m_CommandList->open();
        nvrhi::utils::BuildBottomLevelAccelStruct(
            m_CommandList, BLAS, blas_desc);
        m_CommandList->close();
        device->executeCommandList(m_CommandList);
        device->waitForIdle();
        device->runGarbageCollection();

        descriptor_handle = descriptor_table->CreateDescriptorHandle(
            nvrhi::BindingSetItem::RawBuffer_SRV(0, vertexBuffer));
    }

    MeshDesc mesh_desc;
    mesh_desc.vbOffset = 0;
    mesh_desc.ibOffset = index_buffer_offset;
    mesh_desc.normalOffset = normal_buffer_offset;
    mesh_desc.tangentOffset = tangent_buffer_offset;
    mesh_desc.texCrdOffset = texcoord_buffer_offset;
    mesh_desc.subsetMatIdOffset = subset_mat_id_offset;
    mesh_desc.bindlessIndex = descriptor_handle.Get();

    mesh_desc.texCrdInterpolation = texCrdInterpolation;
    mesh_desc.normalInterpolation = normals.size() == points.size()
                                        ? InterpolationType::Vertex
                                        : InterpolationType::FaceVarying;

    mesh_desc_buffer = render_param->InstanceCollection->mesh_pool.allocate(1);
    mesh_desc_buffer->write_data(&mesh_desc);
}

void Hd_USTC_CG_Mesh::updateTLAS(
    Hd_USTC_CG_RenderParam* render_param,
    HdSceneDelegate* sceneDelegate,
    HdDirtyBits* dirtyBits)
{
    _UpdateInstancer(sceneDelegate, dirtyBits);
    const SdfPath& id = GetId();

    HdInstancer::_SyncInstancerAndParents(
        sceneDelegate->GetRenderIndex(), GetInstancerId());

    VtMatrix4dArray transforms;
    if (!GetInstancerId().IsEmpty()) {
        // Retrieve instance transforms from the instancer.
        HdRenderIndex& renderIndex = sceneDelegate->GetRenderIndex();
        HdInstancer* instancer = renderIndex.GetInstancer(GetInstancerId());
        transforms = static_cast<Hd_USTC_CG_Instancer*>(instancer)
                         ->ComputeInstanceTransforms(GetId());
    }
    else {
        // If there's no instancer, add a single instance with transform
        // I.
        transforms.push_back(GfMatrix4d(1.0f));
    }

    auto& rt_instance_pool = render_param->InstanceCollection->rt_instance_pool;
    std::vector<nvrhi::rt::InstanceDesc> instances;
    instances.resize(transforms.size());

    rt_instanceBuffer = rt_instance_pool.allocate(instances.size());

    instanceBuffer = render_param->InstanceCollection->instance_pool.allocate(
        transforms.size());

    auto material_id = GetMaterialId();

    log::info("Material id: %s", material_id.GetText());

    Hd_USTC_CG_Material* material = (*render_param->material_map)[material_id];

    for (int i = 0; i < transforms.size(); ++i) {
        // Combine the local transform and the instance transform.

        GfMatrix4f mat = transform * GfMatrix4f(transforms[i]);
        GfMatrix4f mat_transposed = mat.GetTranspose();

        nvrhi::rt::InstanceDesc instanceDesc;
        instanceDesc.blasDeviceAddress = BLAS->getDeviceAddress();
        instanceDesc.instanceMask = 1;
        instanceDesc.flags =
            nvrhi::rt::InstanceFlags::TriangleFrontCounterclockwise;

        memcpy(
            instanceDesc.transform,
            mat_transposed.data(),
            sizeof(nvrhi::rt::AffineTransform));

        instanceDesc.instanceID = instanceBuffer->index() + i;
        instances[i] = instanceDesc;

        GeometryInstanceData instance_data;
        instance_data.geometryID = mesh_desc_buffer->index();
        if (material) {
            material->ensure_material_data_handle(render_param);
            instance_data.materialID = material->GetMaterialLocation();
        }
        else {
            instance_data.materialID = -1;
        }
        memcpy(&instance_data.transform, mat.data(), sizeof(pxr::GfMatrix4f));
        instanceBuffer->write_data(&instance_data, i);
    }
    render_param->InstanceCollection->set_require_rebuild_tlas();
    rt_instanceBuffer->write_data(instances.data());

    draw_indirect =
        render_param->InstanceCollection->draw_indirect_pool.allocate(1);
    nvrhi::DrawIndirectArguments args;

    args.vertexCount = triangulatedIndices.size() * 3;
    args.instanceCount = instances.size();
    args.startVertexLocation = 0;
    args.startInstanceLocation = instanceBuffer->index();

    draw_indirect->write_data(&args);
    nvrhi::DrawIndirectArguments dbg_args;

    draw_indirect->read_data(&dbg_args);

    std::cout << "draw_indirect: " << std::endl;
    std::cout << "vertexCount: " << dbg_args.vertexCount << std::endl;
    std::cout << "instanceCount: " << dbg_args.instanceCount << std::endl;

    std::cout << "startVertexLocation: " << dbg_args.startVertexLocation
              << std::endl;
    std::cout << "startInstanceLocation: " << dbg_args.startInstanceLocation
              << std::endl;
}

void Hd_USTC_CG_Mesh::_InitRepr(
    const TfToken& reprToken,
    HdDirtyBits* dirtyBits)
{
}

void Hd_USTC_CG_Mesh::_SetMaterialId(
    HdSceneDelegate* delegate,
    Hd_USTC_CG_Mesh* rprim)
{
    SdfPath const& newMaterialId = delegate->GetMaterialId(rprim->GetId());
    if (rprim->GetMaterialId() != newMaterialId) {
        rprim->SetMaterialId(newMaterialId);
    }
}

void Hd_USTC_CG_Mesh::Sync(
    HdSceneDelegate* sceneDelegate,
    HdRenderParam* renderParam,
    HdDirtyBits* dirtyBits,
    const TfToken& reprToken)
{
    _dirtyBits = *dirtyBits;
    HD_TRACE_FUNCTION();
    HF_MALLOC_TAG_FUNCTION();

    _MeshReprConfig::DescArray descs = _GetReprDesc(reprToken);

    const SdfPath& id = GetId();
    std::string path = id.GetText();

    if (HdChangeTracker::IsVisibilityDirty(*dirtyBits, id)) {
        _sharedData.visible = sceneDelegate->GetVisible(id);
    }

    if (*dirtyBits & HdChangeTracker::DirtyMaterialId) {
        _SetMaterialId(sceneDelegate, this);
    }

    bool requires_rebuild_blas =
        HdChangeTracker::IsPrimvarDirty(*dirtyBits, id, HdTokens->points) ||
        HdChangeTracker::IsTopologyDirty(*dirtyBits, id);

    bool requires_rebuild_tlas =
        requires_rebuild_blas ||
        HdChangeTracker::IsInstancerDirty(*dirtyBits, id) ||
        HdChangeTracker::IsTransformDirty(*dirtyBits, id) ||
        HdChangeTracker::IsVisibilityDirty(*dirtyBits, id);

    if (HdChangeTracker::IsPrimvarDirty(*dirtyBits, id, HdTokens->points)) {
        VtValue value = sceneDelegate->Get(id, HdTokens->points);
        points = value.Get<VtVec3fArray>();

        _normalsValid = false;
    }

    if (!points.empty()) {
        if (HdChangeTracker::IsPrimvarDirty(
                *dirtyBits, id, HdTokens->normals) ||
            HdChangeTracker::IsPrimvarDirty(*dirtyBits, id, HdTokens->widths) ||
            HdChangeTracker::IsPrimvarDirty(
                *dirtyBits, id, HdTokens->primvar)) {
            _UpdatePrimvarSources(sceneDelegate, *dirtyBits, renderParam);
        }

        if (HdChangeTracker::IsTopologyDirty(*dirtyBits, id)) {
            topology = GetMeshTopology(sceneDelegate);

            HdMeshUtil meshUtil(&topology, GetId());
            meshUtil.ComputeTriangleIndices(
                &triangulatedIndices, &trianglePrimitiveParams);

            auto& geom_subsets = topology.GetGeomSubsets();

            for (auto& primvar : _primvarSourceMap) {
                if (primvar.second.interpolation ==
                    HdInterpolationFaceVarying) {
                    VtValue value = primvar.second.data;

                    if (value.IsArrayValued()) {
                        if (value.IsHolding<VtVec3fArray>()) {
                            if (value.Get<VtVec3fArray>().size() !=
                                topology.GetFaceVertexIndices().size()) {
                                log::error(
                                    "FaceVarying primvar size mismatch: %s, "
                                    "expected %d, have %d",
                                    primvar.first.GetText(),
                                    topology.GetFaceVertexIndices().size(),
                                    value.Get<VtVec2fArray>().size());
                            }
                            meshUtil.ComputeTriangulatedFaceVaryingPrimvar(
                                value.Get<VtVec3fArray>().data(),
                                value.GetArraySize(),
                                HdTypeFloatVec3,
                                &primvar.second.data);
                        }
                        else if (value.IsHolding<VtVec2fArray>()) {
                            assert(
                                value.Get<VtVec2fArray>().size() ==
                                topology.GetFaceVertexIndices().size());
                            meshUtil.ComputeTriangulatedFaceVaryingPrimvar(
                                value.Get<VtVec2fArray>().data(),
                                value.GetArraySize(),
                                HdTypeFloatVec2,
                                &primvar.second.data);
                        }
                        else if (value.IsHolding<VtVec4fArray>()) {
                            assert(
                                value.Get<VtVec4fArray>().size() ==
                                topology.GetFaceVertexIndices().size());
                            log::info(
                                "Get a VtVec4fArray, named %s",
                                primvar.first.GetText());
                            meshUtil.ComputeTriangulatedFaceVaryingPrimvar(
                                value.Get<VtVec4fArray>().data(),
                                value.GetArraySize(),
                                HdTypeFloatVec4,
                                &primvar.second.data);
                        }
                    }

                    if (primvar.second.data.GetArraySize() !=
                        triangulatedIndices.size() * 3) {
                        log::error(
                            "FaceVarying primvar size mismatch: %s, expected "
                            "%d, have %d",
                            primvar.first.GetText(),
                            triangulatedIndices.size() * 3,
                            primvar.second.data.GetArraySize());
                    }
                }

                //// Then make them per-vertex

                // if (primvar.second.interpolation ==
                //     HdInterpolationFaceVarying) {
                //     auto value = primvar.second.data;
                //     if (value.IsArrayValued()) {
                //         if (value.IsHolding<VtVec3fArray>()) {
                //             VtVec3fArray vec3fArray =
                //             value.Get<VtVec3fArray>(); VtVec3fArray
                //             newVec3fArray =
                //                 VtVec3fArray(points.size());
                //             for (int i = 0; i < triangulatedIndices.size();
                //                  i += 1) {
                //                 newVec3fArray[triangulatedIndices[i][0]] =
                //                     vec3fArray[i * 3];
                //                 newVec3fArray[triangulatedIndices[i][1]] =
                //                     vec3fArray[i * 3 + 1];
                //                 newVec3fArray[triangulatedIndices[i][2]] =
                //                     vec3fArray[i * 3 + 2];
                //             }

                //            primvar.second.data = VtValue(newVec3fArray);
                //        }
                //        else if (value.IsHolding<VtVec2fArray>()) {
                //            VtVec2fArray vec2fArray =
                //            value.Get<VtVec2fArray>(); VtVec2fArray
                //            newVec2fArray =
                //                VtVec2fArray(points.size());
                //            for (int i = 0; i < triangulatedIndices.size();
                //                 i += 1) {
                //                newVec2fArray[triangulatedIndices[i][0]] =
                //                    vec2fArray[i * 3];
                //                newVec2fArray[triangulatedIndices[i][1]] =
                //                    vec2fArray[i * 3 + 1];
                //                newVec2fArray[triangulatedIndices[i][2]] =
                //                    vec2fArray[i * 3 + 2];
                //            }

                //            primvar.second.data = VtValue(newVec2fArray);
                //        }
                //    }
                //}
            }

            if (!geom_subsets.empty()) {
                std::unordered_map<int, int> subset_material_id_map;

                for (auto& subset : geom_subsets) {
                    auto face_ids = subset.indices;

                    auto material_map =
                        static_cast<Hd_USTC_CG_RenderParam*>(renderParam)
                            ->material_map;

                    auto p = material_map->find(subset.materialId);

                    if (p == material_map->end()) {
                        log::error(
                            "Material not found for subset %s",
                            subset.materialId.GetText());
                        continue;
                    }

                    auto material_id = (*p).second->GetMaterialLocation();
                    for (auto face_id : face_ids) {
                        subset_material_id_map[face_id] = material_id;
                    }
                }

                VtArray<unsigned> material_id_primvars;
                material_id_primvars.resize(triangulatedIndices.size() * 3, 0u);

                assert(
                    triangulatedIndices.size() ==
                    trianglePrimitiveParams.size());

                for (int i = 0; i < triangulatedIndices.size(); i++) {
                    material_id_primvars[i * 3] = subset_material_id_map
                        [HdMeshUtil::DecodeFaceIndexFromCoarseFaceParam(
                            trianglePrimitiveParams[i])];
                    material_id_primvars[i * 3 + 1] = subset_material_id_map
                        [HdMeshUtil::DecodeFaceIndexFromCoarseFaceParam(
                            trianglePrimitiveParams[i])];
                    material_id_primvars[i * 3 + 2] = subset_material_id_map
                        [HdMeshUtil::DecodeFaceIndexFromCoarseFaceParam(
                            trianglePrimitiveParams[i])];
                }

                _primvarSourceMap[TfToken("subset_material_id")] = {
                    VtValue(material_id_primvars), HdInterpolationFaceVarying
                };
            }

            _normalsValid = false;
            _adjacencyValid = false;
        }
        if (HdChangeTracker::IsInstancerDirty(*dirtyBits, id) ||
            HdChangeTracker::IsTransformDirty(*dirtyBits, id)) {
            // TODO: fill instance matrix buffe
            // r
            transform = GfMatrix4f(sceneDelegate->GetTransform(id));
        }

        if (!_normalsValid) {
            VtValue normal_primvar;

            if (_primvarSourceMap.find(HdTokens->normals) !=
                _primvarSourceMap.end()) {
                normal_primvar = _primvarSourceMap[HdTokens->normals].data;
            }
            else {
                normal_primvar = GetNormals(sceneDelegate);
                log::info(GetId().GetAsString().c_str());
            }

            if (normal_primvar.IsEmpty() ||
                (normal_primvar.IsArrayValued() &&
                 normal_primvar.GetArraySize() == 1)) {
                // If there are no normals authored, we need to compute
                // them. This is the case for example when the normals
                // are not authored in the USD file, but are computed by
                // the renderer. We compute the normals here and store
                // them in the normals member variable.

                if (!_adjacencyValid) {
                    _adjacency.BuildAdjacencyTable(&topology);
                    _adjacencyValid = true;
                    // If we rebuilt the adjacency table, force a
                    // rebuild of normals.
                    _normalsValid = false;
                }
                normals = Hd_SmoothNormals::ComputeSmoothNormals(
                    &_adjacency, points.size(), points.cdata());
                assert(points.size() == normals.size());
            }
            else {
                // If normals are authored, we use them.

                normals = normal_primvar.Get<VtVec3fArray>();
            }

            _normalsValid = true;
        }
        _UpdateComputedPrimvarSources(sceneDelegate, *dirtyBits);
        if (!points.empty()) {
            if (requires_rebuild_blas) {
                create_gpu_resources(
                    static_cast<Hd_USTC_CG_RenderParam*>(renderParam));
            }

            if (requires_rebuild_tlas) {
                if (IsVisible()) {
                    updateTLAS(
                        static_cast<Hd_USTC_CG_RenderParam*>(renderParam),
                        sceneDelegate,
                        dirtyBits);
                }
            }
        }

        *dirtyBits &= ~HdChangeTracker::AllSceneDirtyBits;
    }
}

void Hd_USTC_CG_Mesh::Finalize(HdRenderParam* renderParam)
{
    vertexBuffer = nullptr;

    BLAS = nullptr;
    instanceBuffer = nullptr;
    rt_instanceBuffer = nullptr;
    mesh_desc_buffer = nullptr;
    draw_indirect = nullptr;
}

USTC_CG_NAMESPACE_CLOSE_SCOPE

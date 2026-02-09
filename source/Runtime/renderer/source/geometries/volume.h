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
#ifndef Hd_RUZINO_VOLUME_H
#define Hd_RUZINO_VOLUME_H
// OpenVDB and NanoVDB headers - assume available

#include <openvdb/io/File.h>
#include <openvdb/openvdb.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "../api.h"
#include "DescriptorTableManager.h"
#include "nvrhi/nvrhi.h"
#include "pxr/base/gf/bbox3d.h"
#include "pxr/base/gf/matrix4f.h"
#include "pxr/imaging/hd/volume.h"
#include "pxr/pxr.h"

RUZINO_NAMESPACE_OPEN_SCOPE
class Hd_RUZINO_RenderParam;
using namespace pxr;

// Volume field data structure
struct VolumeFieldData {
    std::string name;
    std::string filePath;
    std::string gridName;   // For OpenVDB grids
    TfToken fieldPrimType;  // OpenVDBAsset, Field3DAsset, etc.

    // Loaded data info
    bool isLoaded = false;
    GfBBox3d boundingBox;
    GfVec3i dimensions;
    GfVec3f voxelSize;
    size_t dataSize = 0;

    // NanoVDB data for GPU upload
    std::vector<uint8_t> nanoVDBData;

    // GPU resources - for PNanoVDB.h usage
    nvrhi::BufferHandle gpuBuffer;
    DescriptorHandle gpuBufferHandle;
    nvrhi::TextureHandle texture3D;

    // Default constructor
    VolumeFieldData() = default;

    // Move constructor and assignment
    VolumeFieldData(VolumeFieldData&& other) noexcept = default;
    VolumeFieldData& operator=(VolumeFieldData&& other) noexcept = default;

    // Delete copy constructor and assignment
    VolumeFieldData(const VolumeFieldData&) = delete;
    VolumeFieldData& operator=(const VolumeFieldData&) = delete;
};

// Volume format types USD supports
enum class VolumeFormat {
    OpenVDB,  // .vdb files
    Field3D,  // .f3d files
    Raw,      // Raw binary data
    Unknown
};

class HD_RUZINO_API Hd_RUZINO_Volume final : public HdVolume {
   public:
    HF_MALLOC_TAG_NEW("new Hd_RUZINO_Volume");

    Hd_RUZINO_Volume(const SdfPath& id);
    ~Hd_RUZINO_Volume() override;

    HdDirtyBits GetInitialDirtyBitsMask() const override;

    void Sync(
        HdSceneDelegate* sceneDelegate,
        HdRenderParam* renderParam,
        HdDirtyBits* dirtyBits,
        const TfToken& reprToken) override;
    void Finalize(HdRenderParam* renderParam) override;

    // GPU resource creation
    void CreateGPUResources(class Hd_RUZINO_RenderParam* renderParam);

    // Convenience methods for later integration
    const GfBBox3d& GetBoundingBox() const
    {
        return _boundingBox;
    }
    const std::unordered_map<TfToken, VolumeFieldData, TfToken::HashFunctor>&
    GetFields() const
    {
        return _fields;
    }

    // For future NanoVDB/shader integration
    struct VolumeGPUData {
        GfMatrix4f transform;
        GfBBox3d boundingBox;
        GfVec3i dimensions;
        GfVec3f voxelSize;
        uint32_t fieldCount;
        uint32_t densityFieldIndex;
        uint32_t temperatureFieldIndex;
        uint32_t velocityFieldIndex;
    };

    const VolumeGPUData& GetGPUData() const
    {
        return _gpuData;
    }

   protected:
    void _InitRepr(const TfToken& reprToken, HdDirtyBits* dirtyBits) override;
    HdDirtyBits _PropagateDirtyBits(HdDirtyBits bits) const override;

   private:
    // Volume loading methods
    void _LoadVolumeFields(HdSceneDelegate* sceneDelegate);
    VolumeFormat _DetectVolumeFormat(const std::string& filePath);
    bool _LoadOpenVDB(
        const std::string& filePath,
        const std::string& gridName,
        VolumeFieldData& fieldData);
    bool _LoadField3D(
        const std::string& filePath,
        const std::string& fieldName,
        VolumeFieldData& fieldData);
    bool _LoadRawVolume(
        const std::string& filePath,
        const GfVec3i& dimensions,
        VolumeFieldData& fieldData);

    void _UpdateBoundingBox();
    void _PrepareGPUData();

    // Volume data
    std::unordered_map<TfToken, VolumeFieldData, TfToken::HashFunctor> _fields;
    GfBBox3d _boundingBox;
    GfMatrix4f _transform;
    VolumeGPUData _gpuData;

    // State tracking
    bool _fieldsLoaded;
    bool _boundingBoxValid;

    // This class does not support copying
    Hd_RUZINO_Volume(const Hd_RUZINO_Volume&) = delete;
    Hd_RUZINO_Volume& operator=(const Hd_RUZINO_Volume&) = delete;
};

RUZINO_NAMESPACE_CLOSE_SCOPE

#endif  // Hd_RUZINO_VOLUME_H

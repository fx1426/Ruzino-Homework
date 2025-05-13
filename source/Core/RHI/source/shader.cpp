#include "RHI/ShaderFactory/shader.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <queue>
#include <string>

#include "RHI/ResourceManager/resource_allocator.hpp"
#include "RHI/internal/resources.hpp"
#include "shaderCompiler.h"
#include "slang-com-ptr.h"
#include "slang.h"

USTC_CG_NAMESPACE_OPEN_SCOPE

std::string ShaderFactory::shader_search_path = "";

ProgramDesc Program::get_desc() const
{
    return desc;
}

nvrhi::ShaderDesc Program::get_shader_desc() const
{
    ShaderDesc desc;

    desc.shaderType = this->desc.shaderType;
    desc.entryName = this->desc.entry_name;
    desc.debugName =
        std::to_string(reinterpret_cast<long long>(getBufferPointer()));
    return desc;
}

void const* Program::getBufferPointer() const
{
    return blob->getBufferPointer();
}

size_t Program::getBufferSize() const
{
    return blob->getBufferSize();
}

const ShaderReflectionInfo& Program::get_reflection_info() const
{
    return reflection_info;
}

ProgramDesc& ProgramDesc::set_path(const std::filesystem::path& path)
{
    this->path = path;
    // #ifdef _DEBUG
    update_last_write_time(path);
    // #endif
    return *this;
}

ProgramDesc& ProgramDesc::set_shader_type(nvrhi::ShaderType shaderType)
{
    this->shaderType = shaderType;
    return *this;
}

ProgramDesc& ProgramDesc::set_entry_name(const std::string& entry_name)
{
    this->entry_name = entry_name;
    // #ifdef _DEBUG
    update_last_write_time(path);
    // #endif

    return *this;
}
namespace fs = std::filesystem;

void ProgramDesc::update_last_write_time(const std::filesystem::path& path)
{
    auto full_path =
        std::filesystem::path(ShaderFactory::shader_search_path) / path;
    if (fs::exists(full_path)) {
        auto possibly_newer_lastWriteTime = fs::last_write_time(full_path);
        if (possibly_newer_lastWriteTime > lastWriteTime) {
            lastWriteTime = possibly_newer_lastWriteTime;
        }
    }
    else {
        lastWriteTime = {};
    }
}

std::string ProgramDesc::get_profile() const
{
    switch (shaderType) {
        case nvrhi::ShaderType::None: break;
        case nvrhi::ShaderType::Compute: return "cs_6_6";
        case nvrhi::ShaderType::Vertex: return "vs_6_6";
        case nvrhi::ShaderType::Hull: return "hs_6_6";
        case nvrhi::ShaderType::Domain: return "ds_6_6";
        case nvrhi::ShaderType::Geometry: return "gs_6_6";
        case nvrhi::ShaderType::Pixel: return "ps_6_6";
        case nvrhi::ShaderType::Amplification: return "as_6_6";
        case nvrhi::ShaderType::Mesh: return "ms_6_6";
        case nvrhi::ShaderType::AllGraphics: return "lib_6_6";
        case nvrhi::ShaderType::RayGeneration: return "rg_6_6";
        case nvrhi::ShaderType::AnyHit: return "ah_6_6";
        case nvrhi::ShaderType::ClosestHit: return "ch_6_6";
        case nvrhi::ShaderType::Miss: return "ms_6_6";
        case nvrhi::ShaderType::Intersection: return "is_6_6";
        case nvrhi::ShaderType::Callable: return "lib_6_6";
        case nvrhi::ShaderType::AllRayTracing: return "lib_6_6";
        case nvrhi::ShaderType::All: return "lib_6_6";
    }

    // Default return value for cases not handled explicitly
    return "lib_6_6";
}
class GlobalSessionPool {
   private:
    std::mutex sessionsMutex;
    std::condition_variable sessionAvailable;
    size_t maxSessions;
    std::unordered_map<std::thread::id, Slang::ComPtr<slang::IGlobalSession>>
        activeThreadSessions;
    std::queue<Slang::ComPtr<slang::IGlobalSession>> cachedSessions;
    size_t totalSessionCount = 0;  // Tracks both active and cached sessions

    Slang::ComPtr<slang::IGlobalSession> createNewSession()
    {
        Slang::ComPtr<slang::IGlobalSession> session;
        slang::createGlobalSession(session.writeRef());
        SlangShaderCompiler::addHLSLPrelude(session);
        SlangShaderCompiler::addCPPPrelude(session);
        return session;
    }

   public:
    GlobalSessionPool() : maxSessions(16)
    {
    }

    Slang::ComPtr<slang::IGlobalSession> getSession()
    {
        std::thread::id threadId = std::this_thread::get_id();
        std::unique_lock<std::mutex> lock(sessionsMutex);

        // Check if this thread already has a session
        auto it = activeThreadSessions.find(threadId);
        if (it != activeThreadSessions.end()) {
            return it->second;
        }

        // Wait until we can create or reuse a session
        while (totalSessionCount >= maxSessions && cachedSessions.empty()) {
            sessionAvailable.wait(lock);
        }

        Slang::ComPtr<slang::IGlobalSession> session;

        // Reuse a cached session if available
        if (!cachedSessions.empty()) {
            session = cachedSessions.front();
            cachedSessions.pop();
        }
        else {
            // Create a new session
            session = createNewSession();
            totalSessionCount++;
        }

        // Associate session with this thread
        activeThreadSessions[threadId] = session;

        return session;
    }

    void releaseSession(std::thread::id threadId)
    {
        std::lock_guard<std::mutex> lock(sessionsMutex);

        auto it = activeThreadSessions.find(threadId);
        if (it != activeThreadSessions.end()) {
            // Move session to cache
            cachedSessions.push(it->second);
            activeThreadSessions.erase(it);

            // Notify waiting threads
            sessionAvailable.notify_one();
        }
    }

    ~GlobalSessionPool()
    {
        // Clean up all sessions
        std::lock_guard<std::mutex> lock(sessionsMutex);
        activeThreadSessions.clear();
        while (!cachedSessions.empty()) {
            cachedSessions.pop();
        }
        totalSessionCount = 0;
    }
};

GlobalSessionPool globalSessionPool;

Slang::ComPtr<slang::IGlobalSession> getGlobalSession()
{
    return globalSessionPool.getSession();
}

void releaseGlobalSession()
{
    globalSessionPool.releaseSession(std::this_thread::get_id());
}

static nvrhi::ResourceType convertBindingTypeToResourceType(
    slang::BindingType bindingType,
    SlangResourceShape resource_shape)
{
    using namespace nvrhi;
    using namespace slang;

    auto ret = ResourceType::None;
    switch (bindingType) {
        case BindingType::Sampler: ret = ResourceType::Sampler; break;
        case BindingType::Texture:
        case BindingType::CombinedTextureSampler:
        case BindingType::InputRenderTarget:
            ret = ResourceType::Texture_SRV;
            break;
        case BindingType::MutableTexture:
            ret = ResourceType::Texture_UAV;
            break;
        case BindingType::TypedBuffer:
        case BindingType::MutableTypedBuffer:
            ret = ResourceType::TypedBuffer_SRV;
            break;
        case BindingType::RawBuffer: ret = ResourceType::RawBuffer_SRV; break;
        case BindingType::MutableRawBuffer:
            ret = ResourceType::RawBuffer_UAV;
            break;
        case BindingType::ConstantBuffer:
        case BindingType::ParameterBlock:
            ret = ResourceType::ConstantBuffer;
            break;
        case BindingType::RayTracingAccelerationStructure:
            ret = ResourceType::RayTracingAccelStruct;
            break;
        case BindingType::PushConstant:
            ret = ResourceType::PushConstants;
            break;
    }

    if (resource_shape == SLANG_STRUCTURED_BUFFER) {
        if (ret == ResourceType::RawBuffer_SRV) {
            ret = ResourceType::StructuredBuffer_SRV;
        }
        else if (ret == ResourceType::RawBuffer_UAV) {
            ret = ResourceType::StructuredBuffer_UAV;
        }
    }

    return ret;
}

void ShaderFactory::modify_vulkan_binding_shift(
    nvrhi::BindingLayoutItem& item) const
{
    switch (item.type) {
        case nvrhi::ResourceType::None: break;
        case nvrhi::ResourceType::Texture_SRV: item.slot -= SRV_OFFSET; break;
        case nvrhi::ResourceType::Texture_UAV: item.slot -= UAV_OFFSET; break;
        case nvrhi::ResourceType::TypedBuffer_SRV:
            item.slot -= SRV_OFFSET;
            break;
        case nvrhi::ResourceType::TypedBuffer_UAV:
            item.slot -= UAV_OFFSET;
            break;
        case nvrhi::ResourceType::StructuredBuffer_SRV:
            item.slot -= SRV_OFFSET;
            break;
        case nvrhi::ResourceType::StructuredBuffer_UAV:
            item.slot -= UAV_OFFSET;
            break;
        case nvrhi::ResourceType::RawBuffer_SRV: item.slot -= SRV_OFFSET; break;
        case nvrhi::ResourceType::RawBuffer_UAV: item.slot -= UAV_OFFSET; break;
        case nvrhi::ResourceType::ConstantBuffer:
            item.slot -= CONSTANT_BUFFER_OFFSET;
            break;
        case nvrhi::ResourceType::VolatileConstantBuffer:
            item.slot -= CONSTANT_BUFFER_OFFSET;
            break;
        case nvrhi::ResourceType::Sampler: item.slot -= SAMPLER_OFFSET; break;
        case nvrhi::ResourceType::RayTracingAccelStruct:
            item.slot -= SRV_OFFSET;
            break;
    }
}
ShaderReflectionInfo ShaderFactory::shader_reflect(
    slang::IComponentType* component,
    nvrhi::ShaderType shader_type,
    slang::IBlob** diagnostic) const
{
    ShaderReflectionInfo ret;
    std::map<std::string, std::tuple<unsigned, unsigned>>& binding_locations =
        ret.binding_locations;

    slang::ShaderReflection* programReflection =
        component->getLayout(0, diagnostic);

    // slang::EntryPointReflection* entryPoint =
    //     programReflection->findEntryPointByName(entryPointName);
    auto parameterCount = programReflection->getParameterCount();
    auto g_layout = programReflection->getGlobalParamsTypeLayout();
    auto binding_set_count = g_layout->getDescriptorSetCount();
    // auto parameterCount = entryPoint->getParameterCount();
    nvrhi::BindingLayoutDescVector& layout_vector = ret.binding_spaces;

    std::vector<unsigned> indices;

    for (int pp = 0; pp < parameterCount; ++pp) {
        slang::VariableLayoutReflection* parameter =
            programReflection->getParameterByIndex(pp);

        slang::TypeLayoutReflection* typeLayout = parameter->getTypeLayout();
        slang::TypeReflection* type_reflection = parameter->getType();
        SlangResourceShape resource_shape = type_reflection->getResourceShape();
        auto d_set_count = typeLayout->getDescriptorSetCount();

        slang::ParameterCategory category = parameter->getCategory();
        std::string name = parameter->getName();

        auto index = parameter->getBindingIndex();
        auto space = parameter->getBindingSpace() +
                     parameter->getOffset(
                         SLANG_PARAMETER_CATEGORY_SUB_ELEMENT_REGISTER_SPACE);

        auto bindingRangeCount = typeLayout->getBindingRangeCount();
        assert(bindingRangeCount == 1);
        slang::BindingType type = typeLayout->getBindingRangeType(0);

        nvrhi::BindingLayoutItem item;
        item.type = convertBindingTypeToResourceType(type, resource_shape);
        item.slot = index;

        if (RHI::get_backend() == nvrhi::GraphicsAPI::VULKAN) {
            modify_vulkan_binding_shift(item);
        }
        if (layout_vector.size() < space + 1) {
            layout_vector.resize(space + 1);
            indices.resize(space + 1, 0);
        }

        binding_locations[name] = std::make_tuple(space, indices[space]++);

        layout_vector[space].addItem(item);
        layout_vector[space].visibility = shader_type;
    }

    return ret;
}

// Function to convert ShaderType to SlangStage
SlangStage ConvertShaderTypeToSlangStage(nvrhi::ShaderType shaderType)
{
    using namespace nvrhi;
    switch (shaderType) {
        case ShaderType::Vertex: return SLANG_STAGE_VERTEX;
        case ShaderType::Hull: return SLANG_STAGE_HULL;
        case ShaderType::Domain: return SLANG_STAGE_DOMAIN;
        case ShaderType::Geometry: return SLANG_STAGE_GEOMETRY;
        case ShaderType::Pixel:
            return SLANG_STAGE_FRAGMENT;  // alias for SLANG_STAGE_PIXEL
        case ShaderType::Amplification: return SLANG_STAGE_AMPLIFICATION;
        case ShaderType::Mesh: return SLANG_STAGE_MESH;
        case ShaderType::Compute: return SLANG_STAGE_COMPUTE;
        case ShaderType::RayGeneration: return SLANG_STAGE_RAY_GENERATION;
        case ShaderType::AnyHit: return SLANG_STAGE_ANY_HIT;
        case ShaderType::ClosestHit: return SLANG_STAGE_CLOSEST_HIT;
        case ShaderType::Miss: return SLANG_STAGE_MISS;
        case ShaderType::Intersection: return SLANG_STAGE_INTERSECTION;
        case ShaderType::Callable: return SLANG_STAGE_CALLABLE;
        default: return SLANG_STAGE_NONE;
    }
}

nvrhi::ShaderHandle ShaderFactory::compile_shader(
    const std::string& entryName,
    nvrhi::ShaderType shader_type,
    std::filesystem::path shader_path,
    ShaderReflectionInfo& reflection_info,
    std::string& error_string,
    const std::vector<ShaderMacro>& macro_defines,
    const std::string& source_code) const
{
    ProgramDesc program_desc;
    program_desc.set_entry_name(entryName);

    if (shader_path != "") {
        program_desc.set_path(shader_path);
    }
    for (const auto& macro_define : macro_defines) {
        program_desc.define(macro_define.name, macro_define.definition);
    }
    program_desc.shaderType = shader_type;
    if (!source_code.empty()) {
        program_desc.source_code = { source_code };
    }

    ProgramHandle shader_compiled;

    if (resource_allocator) {
        shader_compiled = resource_allocator->create(program_desc);
    }
    else {
        shader_compiled = createProgram(program_desc);
    }

    if (!shader_compiled->get_error_string().empty()) {
        error_string = shader_compiled->get_error_string();
        shader_compiled = nullptr;
        return nullptr;
    }

    nvrhi::ShaderDesc desc = shader_compiled->get_shader_desc();

    reflection_info = shader_compiled->get_reflection_info();

    ShaderHandle shader;
    if (resource_allocator) {
        shader = resource_allocator->create(
            desc,
            shader_compiled->getBufferPointer(),
            shader_compiled->getBufferSize());
    }
    else {
        shader = device->createShader(
            desc,
            shader_compiled->getBufferPointer(),
            shader_compiled->getBufferSize());
    }

    if (resource_allocator) {
        resource_allocator->destroy(shader_compiled);
    }
    else {
        shader_compiled = nullptr;
    }

    return shader;
}

ProgramHandle ShaderFactory::compile_cpu_executable(
    const std::string& entryName,
    nvrhi::ShaderType shader_type,
    std::filesystem::path shader_path,
    ShaderReflectionInfo& reflection_info,
    std::string& error_string,
    const std::vector<ShaderMacro>& macro_defines,
    const std::string& source_code)
{
    ProgramDesc desc;

    if (shader_path != "") {
        desc.set_path(shader_path);
    }
    desc.set_entry_name(entryName);

    for (const auto& macro_define : macro_defines) {
        desc.define(macro_define.name, macro_define.definition);
    }
    desc.shaderType = shader_type;
    desc.source_code = { source_code };

    ProgramHandle program_handle;
    program_handle = ProgramHandle::Create(new Program());

    program_handle->desc = desc;

    SlangCompileTarget target = SLANG_SHADER_HOST_CALLABLE;

    SlangCompile(
        desc.path,
        desc.source_code,
        desc.entry_name.c_str(),
        desc.shaderType,
        desc.get_profile().c_str(),
        desc.macros,
        program_handle->reflection_info,
        program_handle->blob,
        program_handle->library,
        program_handle->error_string,
        target);

    reflection_info = program_handle->get_reflection_info();
    error_string = program_handle->get_error_string();

    return program_handle;
}

void ShaderFactory::populate_vk_options(
    std::vector<slang::CompilerOptionEntry>& vk_compiler_options)
{
    vk_compiler_options.push_back(
        { slang::CompilerOptionName::VulkanUseEntryPointName,
          slang::CompilerOptionValue{ slang::CompilerOptionValueKind::Int,
                                      1 } });
    vk_compiler_options.push_back(
        { slang::CompilerOptionName::VulkanBindShiftAll,
          slang::CompilerOptionValue{
              slang::CompilerOptionValueKind::Int, 2, SRV_OFFSET } });
    vk_compiler_options.push_back(
        { slang::CompilerOptionName::VulkanBindShiftAll,
          slang::CompilerOptionValue{
              slang::CompilerOptionValueKind::Int, 1, SAMPLER_OFFSET } });
    vk_compiler_options.push_back(
        { slang::CompilerOptionName::VulkanBindShiftAll,
          slang::CompilerOptionValue{ slang::CompilerOptionValueKind::Int,
                                      3,
                                      CONSTANT_BUFFER_OFFSET } });
    vk_compiler_options.push_back(
        { slang::CompilerOptionName::VulkanBindShiftAll,
          slang::CompilerOptionValue{
              slang::CompilerOptionValueKind::Int, 0, UAV_OFFSET } });
}

#define CHECK_REPORTED_ERROR()                                           \
    if (SLANG_FAILED(result)) {                                          \
        if (diagnostics) {                                               \
            error_string = (const char*)diagnostics->getBufferPointer(); \
        }                                                                \
        return;                                                          \
    }                                                                    \
    else if (diagnostics) {                                              \
        log::warning((const char*)diagnostics->getBufferPointer());      \
    }

void ShaderFactory::SlangCompile(
    const std::filesystem::path& path,
    const std::vector<std::string>& sourceCodes,
    const char* entryPoint,
    nvrhi::ShaderType shaderType,
    const char* profile,
    const std::vector<ShaderMacro>& defines,
    ShaderReflectionInfo& shader_reflection,
    Slang::ComPtr<ISlangBlob>& ppResultBlob,
    Slang::ComPtr<ISlangSharedLibrary>& ppSharedLirary,
    std::string& error_string,
    SlangCompileTarget target,
    Slang::ComPtr<slang::IComponentType>* linkedProgram1) const
{
    auto stage = ConvertShaderTypeToSlangStage(shaderType);

    auto global_session = getGlobalSession();

    auto profile_id = global_session->findProfile(profile);

    std::vector<slang::CompilerOptionEntry> vk_compiler_options;

    if (target == SLANG_SPIRV) {
        populate_vk_options(vk_compiler_options);
    }

    slang::TargetDesc desc;
    desc.format = target;
    desc.profile = profile_id;
    if (target == SLANG_SPIRV)
        desc.flags = SLANG_TARGET_FLAG_GENERATE_WHOLE_PROGRAM |
                     SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY;
    else if (target == SLANG_DXIL) {
        // Use appropriate flags for DXIL
        desc.flags = SLANG_TARGET_FLAG_GENERATE_WHOLE_PROGRAM;
    }
    std::vector<slang::PreprocessorMacroDesc> macros;

    for (const auto& define : defines) {
        macros.push_back({ define.name.c_str(), define.definition.c_str() });
    }

    Slang::ComPtr<slang::ISession> p_compile_session;

    slang::SessionDesc compile_session_desc;
    compile_session_desc.targets = &desc;
    compile_session_desc.targetCount = 1;

    compile_session_desc.preprocessorMacros = macros.data();
    compile_session_desc.preprocessorMacroCount =
        static_cast<SlangInt>(macros.size());

    std::vector<std::string> searchPaths = { shader_search_path };
    searchPaths.push_back("./");
    searchPaths.push_back(shader_search_path + "/shaders/");
    for (auto& search_path : search_paths) {
        searchPaths.push_back(search_path);
    }

    std::vector<const char*> slangSearchPaths;
    for (auto& path : searchPaths) {
        slangSearchPaths.push_back(path.data());
    }
    compile_session_desc.searchPaths = slangSearchPaths.data();
    compile_session_desc.searchPathCount = (SlangInt)slangSearchPaths.size();

    if (target == SLANG_SPIRV) {
        compile_session_desc.compilerOptionEntries = vk_compiler_options.data();

        compile_session_desc.compilerOptionEntryCount =
            static_cast<SlangInt>(vk_compiler_options.size());
    }
    SlangResult result;

    result = global_session->createSession(
        compile_session_desc, p_compile_session.writeRef());

    assert(result == SLANG_OK);

    Slang::ComPtr<slang::IBlob> diagnostics;

    if (target == SLANG_HOST_EXECUTABLE) {
        // result = SlangShaderCompiler::addCPPHeaderInclude(slangRequest);
        assert(result == SLANG_OK);
    }

    static std::atomic<unsigned> shader_id = 0;

    auto load_module_from_source =
        [&](const std::string& sourceCode,
            slang::ISession* session) -> slang::IModule* {
        auto id = shader_id++;
        return session->loadModuleFromSourceString(
            (std::to_string(id) + path.filename().generic_string()).c_str(),
            (std::to_string(id) + path.generic_string()).c_str(),
            sourceCode.c_str(),
            diagnostics.writeRef());
    };

    auto load_module_from_path =
        [&](const std::filesystem::path& modulePath,
            slang::ISession* session) -> slang::IModule* {
        return session->loadModule(
            modulePath.generic_string().c_str(), diagnostics.writeRef());
    };

    std::vector<Slang::ComPtr<slang::IModule>> modules;
    bool loaded_successfully = false;

    // Try to load from source code if provided
    if (!sourceCodes.empty()) {
        for (const auto& sourceCode : sourceCodes) {
            if (sourceCode.empty())
                continue;

            auto m =
                load_module_from_source(sourceCode, p_compile_session.get());
            if (m) {
                modules.emplace_back(m);
                loaded_successfully = true;
            }

            else {
                if (diagnostics) {
                    error_string = (const char*)diagnostics->getBufferPointer();
                }
            }
        }
    }

    // Additionally try loading from path if provided (not exclusively)
    if (!path.empty()) {
        auto m = load_module_from_path(path, p_compile_session.get());
        if (m) {
            modules.emplace_back(m);
            loaded_successfully = true;
        }
        else {
            if (diagnostics) {
                error_string = (const char*)diagnostics->getBufferPointer();
            }
        }
    }

    // Report error only if nothing could be loaded
    if (!loaded_successfully) {
        if (diagnostics) {
            error_string = (const char*)diagnostics->getBufferPointer();
        }
        return;
    }

    std::vector<slang::IComponentType*> components;

    for (auto& module : modules) {
        components.push_back(module.get());
    }

    Slang::ComPtr<slang::IEntryPoint> entry;

    if (!std::string(entryPoint).empty()) {
        CHECK_REPORTED_ERROR();

        result = modules[0]->findAndCheckEntryPoint(
            entryPoint, stage, entry.writeRef(), diagnostics.writeRef());
        CHECK_REPORTED_ERROR();
        components.push_back(entry.get());
    }

    Slang::ComPtr<slang::IComponentType> program;
    result = p_compile_session->createCompositeComponentType(
        components.data(),
        components.size(),
        program.writeRef(),
        diagnostics.writeRef());

    CHECK_REPORTED_ERROR();

    Slang::ComPtr<slang::IComponentType> linkedProgram;

    result = program->link(linkedProgram.writeRef(), diagnostics.writeRef());

    CHECK_REPORTED_ERROR();

    shader_reflection =
        shader_reflect(linkedProgram.get(), shaderType, diagnostics.writeRef());
    CHECK_REPORTED_ERROR();

    if (target == SLANG_SHADER_HOST_CALLABLE) {
        result = linkedProgram->getEntryPointHostCallable(
            0, 0, ppSharedLirary.writeRef(), diagnostics.writeRef());
        CHECK_REPORTED_ERROR();
        assert(result == SLANG_OK);
    }
    else {
        result = linkedProgram->getTargetCode(
            0, ppResultBlob.writeRef(), diagnostics.writeRef());
        CHECK_REPORTED_ERROR();
        assert(ppResultBlob);
        assert(result == SLANG_OK);
    }

    releaseGlobalSession();
}

ProgramHandle ShaderFactory::createProgram(const ProgramDesc& desc) const
{
    ProgramHandle ret;
    ret = ProgramHandle::Create(new Program());

    ret->desc = desc;

    SlangCompileTarget target =
        (RHI::get_backend() == nvrhi::GraphicsAPI::VULKAN) ? SLANG_SPIRV
                                                           : SLANG_DXIL;

    SlangCompile(
        desc.path,
        desc.source_code,
        desc.entry_name.c_str(),
        desc.shaderType,
        desc.get_profile().c_str(),
        desc.macros,
        ret->reflection_info,
        ret->blob,
        ret->library,
        ret->error_string,
        target,
        std::addressof(ret->linkedProgram));

    return ret;
}

USTC_CG_NAMESPACE_CLOSE_SCOPE

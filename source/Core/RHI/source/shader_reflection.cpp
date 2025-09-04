#include "RHI/ShaderFactory/shader_reflection.hpp"

#include <spdlog/spdlog.h>
#include "RHI/ShaderFactory/shader.hpp"

USTC_CG_NAMESPACE_OPEN_SCOPE

const nvrhi::BindingLayoutDescVector&
ShaderReflectionInfo::get_binding_layout_descs() const
{
    return binding_spaces;
}

unsigned ShaderReflectionInfo::get_binding_space(const std::string& path)
{
    std::string normalized_path = normalize_path(path);
    
    // First try exact match
    auto it = binding_locations.find(normalized_path);
    if (it != binding_locations.end()) {
        return std::get<0>(it->second);
    }
    
    // If not found and it's an array access, try fallback to base name
    std::string base_name = resolve_base_name(normalized_path);
    if (base_name != normalized_path) {
        it = binding_locations.find(base_name);
        if (it != binding_locations.end()) {
            return std::get<0>(it->second);
        }
    }
    
    // spdlog::warn("Binding space not found: %s (base: %s)", path.c_str(), base_name.c_str());
    return -1;
}

unsigned ShaderReflectionInfo::get_binding_location(const std::string& path)
{
    std::string normalized_path = normalize_path(path);
    
    // First try exact match
    auto it = binding_locations.find(normalized_path);
    if (it != binding_locations.end()) {
        return std::get<1>(it->second);
    }
    
    // If not found and it's an array access, try fallback to base name
    std::string base_name = resolve_base_name(normalized_path);
    if (base_name != normalized_path) {
        it = binding_locations.find(base_name);
        if (it != binding_locations.end()) {
            return std::get<1>(it->second);
        }
    }
    
    spdlog::error("Binding location not found: %s (base: %s)", path.c_str(), base_name.c_str());
    return -1;
}

nvrhi::ResourceType ShaderReflectionInfo::get_binding_type(const std::string& path)
{
    std::string normalized_path = normalize_path(path);
    
    // First try exact match
    auto it = binding_locations.find(normalized_path);
    if (it != binding_locations.end()) {
        return binding_spaces[std::get<0>(it->second)]
            .bindings[std::get<1>(it->second)]
            .type;
    }
    
    // If not found and it's an array access, try fallback to base name
    std::string base_name = resolve_base_name(normalized_path);
    if (base_name != normalized_path) {
        it = binding_locations.find(base_name);
        if (it != binding_locations.end()) {
            return binding_spaces[std::get<0>(it->second)]
                .bindings[std::get<1>(it->second)]
                .type;
        }
    }
    
    spdlog::error("Binding type not found: %s (base: %s)", path.c_str(), base_name.c_str());
    return nvrhi::ResourceType::None;
}

bool ShaderReflectionInfo::has_binding(const std::string& path) const
{
    std::string normalized_path = normalize_path(path);
    
    // First try exact match
    if (binding_locations.find(normalized_path) != binding_locations.end()) {
        return true;
    }
    
    // If not found and it's an array access, try fallback to base name
    std::string base_name = resolve_base_name(normalized_path);
    if (base_name != normalized_path) {
        return binding_locations.find(base_name) != binding_locations.end();
    }
    
    return false;
}

std::string ShaderReflectionInfo::resolve_base_name(const std::string& path) const
{
    // Find the first occurrence of '[' to get the base name for arrays
    // For now, we prioritize array access over member access
    size_t bracket_pos = path.find('[');
    size_t dot_pos = path.find('.');
    
    if (bracket_pos != std::string::npos && (dot_pos == std::string::npos || bracket_pos < dot_pos)) {
        // Array access found first
        return path.substr(0, bracket_pos);
    } else if (dot_pos != std::string::npos) {
        // Member access found
        return path.substr(0, dot_pos);
    }
    
    // No special characters found, return the whole path
    return path;
}

std::string ShaderReflectionInfo::extract_array_indices(const std::string& path) const
{
    // Extract all array indices and member accesses for future use
    size_t pos = path.find_first_of("[.");
    if (pos == std::string::npos) {
        return "";
    }
    return path.substr(pos);
}

bool ShaderReflectionInfo::is_array_access(const std::string& path) const
{
    return path.find('[') != std::string::npos;
}

std::string ShaderReflectionInfo::normalize_path(const std::string& path) const
{
    // Handle various path formats and normalize them
    std::string normalized = path;
    
    // Replace consecutive dots with single dot
    size_t pos = 0;
    while ((pos = normalized.find("..", pos)) != std::string::npos) {
        normalized.replace(pos, 2, ".");
        pos += 1;
    }
    
    // Remove leading/trailing dots
    if (!normalized.empty() && normalized.front() == '.') {
        normalized.erase(0, 1);
    }
    if (!normalized.empty() && normalized.back() == '.') {
        normalized.pop_back();
    }
    
    return normalized;
}

ShaderReflectionInfo ShaderReflectionInfo::operator+(
    const ShaderReflectionInfo& other) const
{
    ShaderReflectionInfo result;
    result.binding_spaces = binding_spaces;

    auto larger_size =
        std::max(binding_spaces.size(), other.binding_spaces.size());

    result.binding_spaces.resize(larger_size);

    auto r_size = other.binding_spaces.size();
    for (int i = 0; i < r_size; ++i) {
        auto& r_space = other.binding_spaces[i];
        auto& l_space = result.binding_spaces[i];

        l_space.visibility = l_space.visibility | r_space.visibility;
    }

    for (const auto& [name, location] : other.binding_locations) {
        auto r_space_id = std::get<0>(location);
        auto r_location_id = std::get<1>(location);

        nvrhi::BindingLayoutItem r_binding_item =
            other.binding_spaces[r_space_id].bindings[r_location_id];

        // search in the first binding layout
        auto l_space_id = r_space_id;
        auto& l_space = result.binding_spaces[r_space_id];

        auto pos = std::find(
            l_space.bindings.begin(), l_space.bindings.end(), r_binding_item);

        if (pos == l_space.bindings.end()) {
            l_space.bindings.push_back(r_binding_item);
            unsigned new_l_location = l_space.bindings.size() - 1;
            result.binding_locations[name] =
                std::make_tuple(l_space_id, new_l_location);
        }
        else {
            result.binding_locations[name] =
                std::make_tuple(l_space_id, pos - l_space.bindings.begin());
        }
    }

    return result;
}

ShaderReflectionInfo& ShaderReflectionInfo::operator+=(
    const ShaderReflectionInfo& other)
{
    *this = *this + other;
    return *this;
}

std::ostream& operator<<(std::ostream& os, const ShaderReflectionInfo& info)
{
    // print binding layout using binding locations
    for (const std::pair<const std::string, std::tuple<unsigned, unsigned>>&
             binding_location : info.binding_locations) {
        os << binding_location.first << " : ";
        auto space_id = std::get<0>(binding_location.second);
        auto location_id = std::get<1>(binding_location.second);

        os << "space: " << space_id << ", location: " << location_id << "; "
           << std::endl;
    }
    return os;
}

USTC_CG_NAMESPACE_CLOSE_SCOPE

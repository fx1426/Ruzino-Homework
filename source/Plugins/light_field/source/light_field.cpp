#include "light_field/light_field.h"

#include "glm/vec3.hpp"

USTC_CG_NAMESPACE_OPEN_SCOPE
std::vector<glm::vec3> g_light_field_lens_locations;

std::vector<glm::vec3> get_light_field_lens_locations()
{
    return g_light_field_lens_locations;
}

std::vector<glm::vec3> set_light_field_lens_locations(
    const std::vector<glm::vec3>& lens_locations)
{
    g_light_field_lens_locations = lens_locations;
    return g_light_field_lens_locations;
}

USTC_CG_NAMESPACE_CLOSE_SCOPE

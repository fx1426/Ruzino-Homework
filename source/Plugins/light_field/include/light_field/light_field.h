#pragma once

#include "api.h"
#include "glm/vec3.hpp"

USTC_CG_NAMESPACE_OPEN_SCOPE

LIGHT_FIELD_API std::vector<glm::vec3> get_light_field_lens_locations();
LIGHT_FIELD_API std::vector<glm::vec3> set_light_field_lens_locations(
    const std::vector<glm::vec3>& lens_locations);

USTC_CG_NAMESPACE_CLOSE_SCOPE

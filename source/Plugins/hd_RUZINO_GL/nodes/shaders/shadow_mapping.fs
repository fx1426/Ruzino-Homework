#version 430 core

uniform mat4 light_view;
uniform mat4 light_projection;
in vec3 vertexPosition;
layout(location = 0) out float shadow_map0;

void main() {
    vec4 clipPos = light_projection * light_view * (vec4(vertexPosition, 1.0));
    float depth = clipPos.z / clipPos.w;
    shadow_map0 = depth * 0.5 + 0.5; // [-1, 1] to [0, 1]
}
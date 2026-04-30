// #define __GNUC__

#include "../camera.h"
#include "../light.h"
#include "nodes/core/def/node_def.hpp"
#include "pxr/base/gf/frustum.h"
#include "pxr/base/gf/matrix4f.h"
#include "pxr/imaging/glf/simpleLight.h"
#include "pxr/imaging/hd/tokens.h"
#include "render_node_base.h"
#include "rich_type_buffer.hpp"
#include "utils/draw_fullscreen.h"
NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(deferred_lighting)
{
    b.add_input<GLTextureHandle>("Position");
    b.add_input<GLTextureHandle>("diffuseColor");
    b.add_input<GLTextureHandle>("MetallicRoughness");
    b.add_input<GLTextureHandle>("Normal");
    b.add_input<GLTextureHandle>("Shadow Maps");
    b.add_input<GLTextureHandle>("AO").optional(true);

    b.add_input<int>("Shadow Mode").default_val(0).min(0).max(1);
    b.add_input<bool>("Use SSAO").default_val(false);
    b.add_input<float>("PCSS Light Size").default_val(0.01f).min(0.001f).max(0.08f);
    b.add_input<int>("PCSS Blocker Samples").default_val(16).min(1).max(64);
    b.add_input<int>("PCSS Filter Samples").default_val(32).min(1).max(64);
    b.add_input<int>("Render Mode").default_val(0).min(0).max(1);
    b.add_input<int>("Toon Bands").default_val(4).min(2).max(8);
    b.add_input<float>("Specular Threshold").default_val(0.45f).min(0.0f).max(1.0f);
    b.add_input<float>("Rim Strength").default_val(0.35f).min(0.0f).max(2.0f);
    b.add_input<float>("Rim Power").default_val(3.0f).min(0.25f).max(8.0f);
    b.add_input<float>("Outline Width").default_val(1.5f).min(0.0f).max(8.0f);
    b.add_input<float>("Normal Edge Threshold").default_val(0.25f).min(0.01f).max(1.0f);
    b.add_input<float>("Depth Edge Threshold").default_val(0.02f).min(0.001f).max(0.2f);
    b.add_input<std::string>("Lighting Shader")
        .default_val("shaders/blinn_phong.fs");
    b.add_output<GLTextureHandle>("Color");
}

struct LightInfo {
    GfMatrix4f light_projection;
    GfMatrix4f light_view;
    GfVec3f position;
    float radius;
    GfVec3f luminance;
    int shadow_map_id;
};

NODE_EXECUTION_FUNCTION(deferred_lighting)
{
    // Fetch all the information

    auto position_texture = params.get_input<GLTextureHandle>("Position");
    auto diffuseColor_texture =
        params.get_input<GLTextureHandle>("diffuseColor");

    auto metallic_roughness =
        params.get_input<GLTextureHandle>("MetallicRoughness");
    auto normal_texture = params.get_input<GLTextureHandle>("Normal");

    auto shadow_maps = params.get_input<GLTextureHandle>("Shadow Maps");
    auto shadow_mode = params.get_input<int>("Shadow Mode");
    auto use_ssao = params.get_input<bool>("Use SSAO");
    auto pcss_light_size = params.get_input<float>("PCSS Light Size");
    auto pcss_blocker_samples = params.get_input<int>("PCSS Blocker Samples");
    auto pcss_filter_samples = params.get_input<int>("PCSS Filter Samples");
    auto render_mode = params.get_input<int>("Render Mode");
    auto toon_bands = params.get_input<int>("Toon Bands");
    auto specular_threshold = params.get_input<float>("Specular Threshold");
    auto rim_strength = params.get_input<float>("Rim Strength");
    auto rim_power = params.get_input<float>("Rim Power");
    auto outline_width = params.get_input<float>("Outline Width");
    auto normal_edge_threshold = params.get_input<float>("Normal Edge Threshold");
    auto depth_edge_threshold = params.get_input<float>("Depth Edge Threshold");
    auto has_ao_input = use_ssao && params.has_input("AO");

    Hd_RUZINO_Camera* free_camera = get_free_camera(params);
    // Creating output textures.
    auto size = position_texture->desc.size;
    GLTextureDesc color_output_desc;
    color_output_desc.format = HdFormatFloat32Vec4;
    color_output_desc.size = size;
    auto color_texture = resource_allocator.create(color_output_desc);

    unsigned int VBO, VAO;
    CreateFullScreenVAO(VAO, VBO);

    auto shaderPath = params.get_input<std::string>("Lighting Shader");

    GLShaderDesc shader_desc;
    shader_desc.set_vertex_path(
        std::filesystem::path(RENDER_NODES_FILES_DIR) /
        std::filesystem::path("shaders/fullscreen.vs"));

    shader_desc.set_fragment_path(
        std::filesystem::path(RENDER_NODES_FILES_DIR) /
        std::filesystem::path(shaderPath));
    auto shader = resource_allocator.create(shader_desc);
    GLuint framebuffer;
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture2D(
        GL_FRAMEBUFFER,
        GL_COLOR_ATTACHMENT0,
        GL_TEXTURE_2D,
        color_texture->texture_id,
        0);

    glClearColor(0.f, 0.f, 0.f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    shader->shader.use();
    shader->shader.setVec2("iResolution", size);

    shader->shader.setInt("diffuseColorSampler", 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, diffuseColor_texture->texture_id);

    shader->shader.setInt("normalMapSampler", 1);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, normal_texture->texture_id);

    shader->shader.setInt("metallicRoughnessSampler", 2);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, metallic_roughness->texture_id);

    shader->shader.setInt("shadow_maps", 3);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D_ARRAY, shadow_maps->texture_id);

    shader->shader.setInt("position", 4);
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, position_texture->texture_id);

    if (has_ao_input) {
        auto ao_texture = params.get_input<GLTextureHandle>("AO");
        shader->shader.setInt("aoSampler", 5);
        glActiveTexture(GL_TEXTURE5);
        glBindTexture(GL_TEXTURE_2D, ao_texture->texture_id);
    }

    shader->shader.setInt("shadowMode", shadow_mode);
    shader->shader.setBool("useSSAO", has_ao_input);
    shader->shader.setFloat("pcssLightSize", pcss_light_size);
    shader->shader.setInt("pcssBlockerSamples", pcss_blocker_samples);
    shader->shader.setInt("pcssFilterSamples", pcss_filter_samples);
    shader->shader.setInt("renderMode", render_mode);
    shader->shader.setInt("toonBands", toon_bands);
    shader->shader.setFloat("specularThreshold", specular_threshold);
    shader->shader.setFloat("rimStrength", rim_strength);
    shader->shader.setFloat("rimPower", rim_power);
    shader->shader.setFloat("outlineWidth", outline_width);
    shader->shader.setFloat("normalEdgeThreshold", normal_edge_threshold);
    shader->shader.setFloat("depthEdgeThreshold", depth_edge_threshold);

    GfVec3f camPos =
        GfMatrix4f(free_camera->GetTransform()).ExtractTranslation();
    shader->shader.setVec3("camPos", camPos);

    GLuint lightBuffer;
    glGenBuffers(1, &lightBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, lightBuffer);
    glViewport(0, 0, size[0], size[1]);
    std::vector<LightInfo> light_vector;

    for (int i = 0; i < lights.size(); ++i) {
        if (light_vector.size() >= 4) {
            break;
        }

        if (!lights[i]->GetId().IsEmpty()) {
            GlfSimpleLight light_params =
                lights[i]->Get(HdTokens->params).Get<GlfSimpleLight>();
            auto diffuse4 = light_params.GetDiffuse();
            pxr::GfVec3f diffuse3(diffuse4[0], diffuse4[1], diffuse4[2]);
            auto position4 = light_params.GetPosition();
            pxr::GfVec3f position3(position4[0], position4[1], position4[2]);

            if (lights[i]->GetLightType() == HdPrimTypeTokens->sphereLight &&
                lights[i]->Get(HdLightTokens->radius).IsHolding<float>()) {
                auto radius =
                    lights[i]->Get(HdLightTokens->radius).Get<float>();

                GfFrustum frustum;
                auto light_view_mat = GfMatrix4f().SetLookAt(
                    position3, GfVec3f(0, 0, 0), GfVec3f(0, 0, 1));
                frustum.SetPerspective(120.f, 1.0, 1, 25.f);
                auto light_projection_mat =
                    GfMatrix4f(frustum.ComputeProjectionMatrix());
                // Keep the light matrices matched with shadow_mapping.
                light_vector.emplace_back(
                    light_projection_mat, light_view_mat, position3, radius,
                    diffuse3, i);
            }

            // You can add directional light here, and also the corresponding
            // shadow map calculation part.
        }
    }

    shader->shader.setInt("light_count", static_cast<int>(light_vector.size()));

    glBufferData(
        GL_SHADER_STORAGE_BUFFER,
        light_vector.size() * sizeof(LightInfo),
        light_vector.data(),
        GL_STATIC_DRAW);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, lightBuffer);

    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    DestroyFullScreenVAO(VAO, VBO);

    auto shader_error = shader->shader.get_error();

    resource_allocator.destroy(shader);
    glDeleteBuffers(1, &lightBuffer);
    glDeleteFramebuffers(1, &framebuffer);
    params.set_output("Color", color_texture);

    if (!shader_error.empty()) {
        return false;
    }
    return true;
}

NODE_DECLARATION_UI(deferred_lighting);
NODE_DEF_CLOSE_SCOPE

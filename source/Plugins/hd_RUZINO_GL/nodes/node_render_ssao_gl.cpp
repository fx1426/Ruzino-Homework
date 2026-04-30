

#include "../camera.h"
#include "../light.h"
#include "nodes/core/def/node_def.hpp"
#include "pxr/base/gf/matrix4f.h"
#include "pxr/imaging/hd/tokens.h"
#include "render_node_base.h"
#include "rich_type_buffer.hpp"
#include "utils/draw_fullscreen.h"
NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(ssao)
{
    b.add_input<GLTextureHandle>("Position");
    b.add_input<GLTextureHandle>("Normal");
    b.add_input<GLTextureHandle>("Depth");

    b.add_input<float>("SSAO Radius").default_val(0.5f).min(0.01f).max(10.0f);
    b.add_input<float>("SSAO Strength").default_val(1.0f).min(0.0f).max(4.0f);
    b.add_input<int>("SSAO Samples").default_val(32).min(1).max(64);
    b.add_input<std::string>("Shader").default_val("shaders/ssao.fs");
    b.add_output<GLTextureHandle>("AO");
}

NODE_EXECUTION_FUNCTION(ssao)
{
    auto position = params.get_input<GLTextureHandle>("Position");
    auto normal = params.get_input<GLTextureHandle>("Normal");
    auto depth = params.get_input<GLTextureHandle>("Depth");
    auto ssao_radius = params.get_input<float>("SSAO Radius");
    auto ssao_strength = params.get_input<float>("SSAO Strength");
    auto ssao_samples = params.get_input<int>("SSAO Samples");

    auto size = position->desc.size;
    Hd_RUZINO_Camera* free_camera = get_free_camera(params);

    unsigned int VBO, VAO;

    CreateFullScreenVAO(VAO, VBO);

    GLTextureDesc texture_desc;
    texture_desc.size = size;
    texture_desc.format = HdFormatFloat32;
    auto ao_texture = resource_allocator.create(texture_desc);

    auto shaderPath = params.get_input<std::string>("Shader");

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
        ao_texture->texture_id,
        0);

    glClearColor(0.f, 0.f, 0.f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    shader->shader.use();
    shader->shader.setVec2("iResolution", size);

    shader->shader.setInt("positionSampler", 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, position->texture_id);

    shader->shader.setInt("normalSampler", 1);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, normal->texture_id);

    shader->shader.setInt("depthSampler", 2);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, depth->texture_id);

    shader->shader.setMat4("view", GfMatrix4f(free_camera->_viewMatrix));
    shader->shader.setMat4(
        "projection", GfMatrix4f(free_camera->_projMatrix));
    shader->shader.setFloat("ssaoRadius", ssao_radius);
    shader->shader.setFloat("ssaoStrength", ssao_strength);
    shader->shader.setInt("ssaoSamples", ssao_samples);

    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    DestroyFullScreenVAO(VAO, VBO);

    auto shader_error = shader->shader.get_error();

    resource_allocator.destroy(shader);
    glDeleteFramebuffers(1, &framebuffer);

    params.set_output("AO", ao_texture);
    if (!shader_error.empty()) {
        return false;
    }
    return true;
}

NODE_DECLARATION_UI(ssao);
NODE_DEF_CLOSE_SCOPE

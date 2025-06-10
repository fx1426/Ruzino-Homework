#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>

// Use the actual OpenUSD internal namespace
using namespace pxr;

// Comparison operators for GfVec2f
inline bool operator<(const GfVec2f& lhs, const GfVec2f& rhs)
{
    if (lhs[0] != rhs[0])
        return lhs[0] < rhs[0];
    return lhs[1] < rhs[1];
}

inline bool operator>(const GfVec2f& lhs, const GfVec2f& rhs)
{
    return rhs < lhs;
}

// Comparison operators for GfVec3f
inline bool operator<(const GfVec3f& lhs, const GfVec3f& rhs)
{
    if (lhs[0] != rhs[0])
        return lhs[0] < rhs[0];
    if (lhs[1] != rhs[1])
        return lhs[1] < rhs[1];
    return lhs[2] < rhs[2];
}

inline bool operator>(const GfVec3f& lhs, const GfVec3f& rhs)
{
    return rhs < lhs;
}

#include <algorithm>

#include "nodes/core/socket_trait.inl"
template<>
struct ValueTrait<pxr::GfVec3f> {
    static constexpr bool has_min = true;
    static constexpr bool has_max = true;
    static constexpr bool has_default = true;
};
template<>
struct ValueTrait<pxr::GfVec2f> {
    static constexpr bool has_min = true;
    static constexpr bool has_max = true;
    static constexpr bool has_default = true;
};

#include "GPUContext/compute_context.hpp"
#define FFX_CPU
static float fs2S;
static float hdr10S;
static uint32_t ctl[24 * 4];

static void LpmSetupOut(uint32_t i, uint32_t* v)
{
    for (int j = 0; j < 4; ++j) {
        ctl[i * 4 + j] = v[j];
    }
}

#include "lpm/ffx_common_types.h"
#include "lpm/ffx_core_cpu.h"
#include "lpm/ffx_lpm.h"
#include "lpm/ffx_lpm_host.h"
#include "lpm/ffx_lpm_private.h"
#include "nodes/core/def/node_def.hpp"
#include "render_node_base.h"

NODE_DEF_OPEN_SCOPE

struct Storage {
    constexpr static bool has_storage = true;

    bool isInitialized = false;
};

NODE_DECLARATION_FUNCTION(lpm)
{
    b.add_input<nvrhi::TextureHandle>("Input Color");
    b.add_input<float>("Shoulder").min(0.0f).max(2.0f).default_val(1.0f);
    b.add_input<float>("Soft Gap").min(0.0f).max(1.0f).default_val(0.0f);
    b.add_input<float>("HDR Max").min(1.0f).max(10000.0f).default_val(1000.0f);
    b.add_input<float>("LPM Exposure").min(-10.0f).max(10.0f).default_val(0.0f);
    b.add_input<float>("Contrast").min(0.0f).max(2.0f).default_val(1.0f);
    b.add_input<float>("Shoulder Contrast")
        .min(0.0f)
        .max(2.0f)
        .default_val(1.0f);
    b.add_input<pxr::GfVec3f>("Saturation")
        .min(GfVec3f(0.0f, 0.0f, 0.0f))
        .max(GfVec3f(2.0f, 2.0f, 2.0f))
        .default_val(GfVec3f(1.0f, 1.0f, 1.0f));
    b.add_input<pxr::GfVec3f>("Crosstalk")
        .min(GfVec3f(0.0f, 0.0f, 0.0f))
        .max(GfVec3f(1.0f, 1.0f, 1.0f))
        .default_val(GfVec3f(0.0f, 0.0f, 0.0f));
    b.add_input<int>("Color Space")
        .min(0)
        .max(2)
        .default_val(0);  // 0=REC709, 1=P3, 2=REC2020
    b.add_input<int>("Display Mode")
        .min(0)
        .max(5)
        .default_val(0);  // 0=LDR, 1=FSHDR_2084, etc.
    b.add_input<float>("Display Min Luminance")
        .min(0.0f)
        .max(1.0f)
        .default_val(0.0f);
    b.add_input<float>("Display Max Luminance")
        .min(1.0f)
        .max(10000.0f)
        .default_val(1000.0f);
    b.add_input<pxr::GfVec2f>("Display Red Primary")
        .min(GfVec2f(0.0f, 0.0f))
        .max(GfVec2f(1.0f, 1.0f))
        .default_val(GfVec2f(0.64f, 0.33f));
    b.add_input<pxr::GfVec2f>("Display Green Primary")
        .min(GfVec2f(0.0f, 0.0f))
        .max(GfVec2f(1.0f, 1.0f))
        .default_val(GfVec2f(0.30f, 0.60f));
    b.add_input<pxr::GfVec2f>("Display Blue Primary")
        .min(GfVec2f(0.0f, 0.0f))
        .max(GfVec2f(1.0f, 1.0f))
        .default_val(GfVec2f(0.15f, 0.06f));
    b.add_input<pxr::GfVec2f>("Display White Point")
        .min(GfVec2f(0.0f, 0.0f))
        .max(GfVec2f(1.0f, 1.0f))
        .default_val(GfVec2f(0.3127f, 0.3290f));

    b.add_output<nvrhi::TextureHandle>("Output Color");
}
NODE_EXECUTION_FUNCTION(lpm)
{
    ProgramDesc cs_program_desc;
    cs_program_desc.shaderType = nvrhi::ShaderType::Compute;
    cs_program_desc.set_path("../lpm/ffx_lpm_filter_pass.slang")
        .set_entry_name("main");
    std::vector<ShaderMacro> macros;
    // -DFFX_GPU=1 -DFFX_HLSL=1 -DFFX_HLSL_SM=65
    macros.push_back({ "FFX_HALF", "0" });
    macros.push_back({ "FFX_GPU", "1" });       // Enable GPU mode
    macros.push_back({ "FFX_HLSL", "1" });      // Enable HLSL mode
    macros.push_back({ "FFX_HLSL_SM", "65" });  // Set shader model to 6.5

    cs_program_desc.define(macros);
    ProgramHandle cs_program = resource_allocator.create(cs_program_desc);
    MARK_DESTROY_NVRHI_RESOURCE(cs_program);
    CHECK_PROGRAM_ERROR(cs_program);

    auto inputColor = params.get_input<nvrhi::TextureHandle>("Input Color");
    auto shoulder = params.get_input<float>("Shoulder");
    auto softGap = params.get_input<float>("Soft Gap");
    auto hdrMax = params.get_input<float>("HDR Max");
    auto lpmExposure = params.get_input<float>("LPM Exposure");
    auto contrast = params.get_input<float>("Contrast");
    auto shoulderContrast = params.get_input<float>("Shoulder Contrast");
    auto saturation = params.get_input<GfVec3f>("Saturation");
    auto crosstalk = params.get_input<GfVec3f>("Crosstalk");
    auto colorSpace = params.get_input<int>("Color Space");
    auto displayMode = params.get_input<int>("Display Mode");
    auto displayMinLuminance = params.get_input<float>("Display Min Luminance");
    auto displayMaxLuminance = params.get_input<float>("Display Max Luminance");
    auto displayRedPrimary = params.get_input<GfVec2f>("Display Red Primary");
    auto displayGreenPrimary =
        params.get_input<GfVec2f>("Display Green Primary");
    auto displayBluePrimary = params.get_input<GfVec2f>("Display Blue Primary");
    auto displayWhitePoint = params.get_input<GfVec2f>("Display White Point");

    // Create output texture with same format as input
    auto inputDesc = inputColor->getDesc();
    auto outputColor = resource_allocator.create(inputDesc);

    ProgramVars program_vars(resource_allocator, cs_program);
    program_vars["r_input_color"] = inputColor;
    program_vars["rw_output_color"] = outputColor;

    // Create constant buffer using FFX LPM constants structure
    LpmConstants lpmConstants = {};

    // Set display mode
    lpmConstants.displayMode = static_cast<uint32_t>(
        displayMode);  // Convert parameters to FFX types for calculation
    FfxFloat32x3 ffxSaturation;
    ffxSaturation[0] = saturation[0];
    ffxSaturation[1] = saturation[1];
    ffxSaturation[2] = saturation[2];

    FfxFloat32x3 ffxCrosstalk;
    ffxCrosstalk[0] = crosstalk[0];
    ffxCrosstalk[1] = crosstalk[1];
    ffxCrosstalk[2] = crosstalk[2];

    FfxFloat32x2 displayMinMaxLuminance;
    displayMinMaxLuminance[0] = displayMinLuminance;
    displayMinMaxLuminance[1] = displayMaxLuminance;

    FfxFloat32x2 fs2R, fs2G, fs2B, fs2W;
    if (displayMode != 0)  // Not LDR mode
    {
        fs2R[0] = displayRedPrimary[0];
        fs2R[1] = displayRedPrimary[1];
        fs2G[0] = displayGreenPrimary[0];
        fs2G[1] = displayGreenPrimary[1];
        fs2B[0] = displayBluePrimary[0];
        fs2B[1] = displayBluePrimary[1];
        fs2W[0] = displayWhitePoint[0];
        fs2W[1] = displayWhitePoint[1];
    }

    // Calculate required scalars for HDR modes
    if (displayMode == 1 || displayMode == 3)  // FSHDR_2084 or HDR10_2084
    {
        hdr10S = LpmHdr10RawScalar(displayMaxLuminance);
    }
    else if (displayMode == 2 || displayMode == 4)  // FSHDR_SCRGB or
                                                    // HDR10_SCRGB
    {
        if (displayMode == 2)  // HDR10_SCRGB
            hdr10S = LpmHdr10ScrgbScalar(displayMaxLuminance);
        else  // FSHDR_SCRGB
            fs2S = LpmFs2ScrgbScalar(displayMinLuminance, displayMaxLuminance);
    }

    // Clear the control array before calculation
    memset(ctl, 0, sizeof(ctl));

    // Calculate LPM constants based on color space and display mode
    FfxLpmColorSpace ffxColorSpace = static_cast<FfxLpmColorSpace>(colorSpace);
    FfxLpmDisplayMode ffxDisplayMode =
        static_cast<FfxLpmDisplayMode>(displayMode);

    switch (ffxColorSpace) {
        case FfxLpmColorSpace::FFX_LPM_ColorSpace_REC709: {
            switch (ffxDisplayMode) {
                case FfxLpmDisplayMode::FFX_LPM_DISPLAYMODE_LDR: {
                    FfxCalculateLpmConsts(
                        shoulder != 1.0f,
                        LPM_CONFIG_709_709,
                        LPM_COLORS_709_709,
                        softGap,
                        hdrMax,
                        lpmExposure,
                        contrast,
                        shoulderContrast,
                        ffxSaturation,
                        ffxCrosstalk);
                } break;
                case FfxLpmDisplayMode::FFX_LPM_DISPLAYMODE_FSHDR_2084: {
                    FfxCalculateLpmConsts(
                        shoulder != 1.0f,
                        LPM_CONFIG_FS2RAWPQ_709,
                        LPM_COLORS_FS2RAWPQ_709,
                        softGap,
                        hdrMax,
                        lpmExposure,
                        contrast,
                        shoulderContrast,
                        ffxSaturation,
                        ffxCrosstalk);
                } break;
                case FfxLpmDisplayMode::FFX_LPM_DISPLAYMODE_FSHDR_SCRGB: {
                    FfxCalculateLpmConsts(
                        shoulder != 1.0f,
                        LPM_CONFIG_FS2SCRGB_709,
                        LPM_COLORS_FS2SCRGB_709,
                        softGap,
                        hdrMax,
                        lpmExposure,
                        contrast,
                        shoulderContrast,
                        ffxSaturation,
                        ffxCrosstalk);
                } break;
                case FfxLpmDisplayMode::FFX_LPM_DISPLAYMODE_HDR10_2084: {
                    FfxCalculateLpmConsts(
                        shoulder != 1.0f,
                        LPM_CONFIG_HDR10RAW_709,
                        LPM_COLORS_HDR10RAW_709,
                        softGap,
                        hdrMax,
                        lpmExposure,
                        contrast,
                        shoulderContrast,
                        ffxSaturation,
                        ffxCrosstalk);
                } break;
                case FfxLpmDisplayMode::FFX_LPM_DISPLAYMODE_HDR10_SCRGB: {
                    FfxCalculateLpmConsts(
                        shoulder != 1.0f,
                        LPM_CONFIG_HDR10SCRGB_709,
                        LPM_COLORS_HDR10SCRGB_709,
                        softGap,
                        hdrMax,
                        lpmExposure,
                        contrast,
                        shoulderContrast,
                        ffxSaturation,
                        ffxCrosstalk);
                } break;
            }
        } break;
        case FfxLpmColorSpace::FFX_LPM_ColorSpace_P3: {
            switch (ffxDisplayMode) {
                case FfxLpmDisplayMode::FFX_LPM_DISPLAYMODE_LDR: {
                    FfxCalculateLpmConsts(
                        shoulder != 1.0f,
                        LPM_CONFIG_709_P3,
                        LPM_COLORS_709_P3,
                        softGap,
                        hdrMax,
                        lpmExposure,
                        contrast,
                        shoulderContrast,
                        ffxSaturation,
                        ffxCrosstalk);
                } break;
                case FfxLpmDisplayMode::FFX_LPM_DISPLAYMODE_FSHDR_2084: {
                    FfxCalculateLpmConsts(
                        shoulder != 1.0f,
                        LPM_CONFIG_FS2RAWPQ_P3,
                        LPM_COLORS_FS2RAWPQ_P3,
                        softGap,
                        hdrMax,
                        lpmExposure,
                        contrast,
                        shoulderContrast,
                        ffxSaturation,
                        ffxCrosstalk);
                } break;
                case FfxLpmDisplayMode::FFX_LPM_DISPLAYMODE_FSHDR_SCRGB: {
                    FfxCalculateLpmConsts(
                        shoulder != 1.0f,
                        LPM_CONFIG_FS2SCRGB_P3,
                        LPM_COLORS_FS2SCRGB_P3,
                        softGap,
                        hdrMax,
                        lpmExposure,
                        contrast,
                        shoulderContrast,
                        ffxSaturation,
                        ffxCrosstalk);
                } break;
                case FfxLpmDisplayMode::FFX_LPM_DISPLAYMODE_HDR10_2084: {
                    FfxCalculateLpmConsts(
                        shoulder != 1.0f,
                        LPM_CONFIG_HDR10RAW_P3,
                        LPM_COLORS_HDR10RAW_P3,
                        softGap,
                        hdrMax,
                        lpmExposure,
                        contrast,
                        shoulderContrast,
                        ffxSaturation,
                        ffxCrosstalk);
                } break;
                case FfxLpmDisplayMode::FFX_LPM_DISPLAYMODE_HDR10_SCRGB: {
                    FfxCalculateLpmConsts(
                        shoulder != 1.0f,
                        LPM_CONFIG_HDR10SCRGB_P3,
                        LPM_COLORS_HDR10SCRGB_P3,
                        softGap,
                        hdrMax,
                        lpmExposure,
                        contrast,
                        shoulderContrast,
                        ffxSaturation,
                        ffxCrosstalk);
                } break;
            }
        } break;
        case FfxLpmColorSpace::FFX_LPM_ColorSpace_REC2020: {
            switch (ffxDisplayMode) {
                case FfxLpmDisplayMode::FFX_LPM_DISPLAYMODE_LDR: {
                    FfxCalculateLpmConsts(
                        shoulder != 1.0f,
                        LPM_CONFIG_709_2020,
                        LPM_COLORS_709_2020,
                        softGap,
                        hdrMax,
                        lpmExposure,
                        contrast,
                        shoulderContrast,
                        ffxSaturation,
                        ffxCrosstalk);
                } break;
                case FfxLpmDisplayMode::FFX_LPM_DISPLAYMODE_FSHDR_2084: {
                    FfxCalculateLpmConsts(
                        shoulder != 1.0f,
                        LPM_CONFIG_FS2RAWPQ_2020,
                        LPM_COLORS_FS2RAWPQ_2020,
                        softGap,
                        hdrMax,
                        lpmExposure,
                        contrast,
                        shoulderContrast,
                        ffxSaturation,
                        ffxCrosstalk);
                } break;
                case FfxLpmDisplayMode::FFX_LPM_DISPLAYMODE_FSHDR_SCRGB: {
                    FfxCalculateLpmConsts(
                        shoulder != 1.0f,
                        LPM_CONFIG_FS2SCRGB_2020,
                        LPM_COLORS_FS2SCRGB_2020,
                        softGap,
                        hdrMax,
                        lpmExposure,
                        contrast,
                        shoulderContrast,
                        ffxSaturation,
                        ffxCrosstalk);
                } break;
                case FfxLpmDisplayMode::FFX_LPM_DISPLAYMODE_HDR10_2084: {
                    FfxCalculateLpmConsts(
                        shoulder != 1.0f,
                        LPM_CONFIG_HDR10RAW_2020,
                        LPM_COLORS_HDR10RAW_2020,
                        softGap,
                        hdrMax,
                        lpmExposure,
                        contrast,
                        shoulderContrast,
                        ffxSaturation,
                        ffxCrosstalk);
                } break;
                case FfxLpmDisplayMode::FFX_LPM_DISPLAYMODE_HDR10_SCRGB: {
                    FfxCalculateLpmConsts(
                        shoulder != 1.0f,
                        LPM_CONFIG_HDR10SCRGB_2020,
                        LPM_COLORS_HDR10SCRGB_2020,
                        softGap,
                        hdrMax,
                        lpmExposure,
                        contrast,
                        shoulderContrast,
                        ffxSaturation,
                        ffxCrosstalk);
                } break;
            }
        } break;
        default: break;
    }

    // Copy the calculated control data to the constants structure
    memcpy(lpmConstants.ctl, ctl, sizeof(ctl));

    // Set display mode
    lpmConstants.displayMode = static_cast<uint32_t>(displayMode);

    // Populate LPM constants with configuration flags
    uint32_t outCon, outSoft, outCon2, outClip, outScaleOnly;

    // Determine configuration based on color space and display mode
    bool con = false, soft = false, con2 = false, clip = false,
         scaleOnly = false;

    switch (ffxColorSpace) {
        case FfxLpmColorSpace::FFX_LPM_ColorSpace_REC709:
            switch (ffxDisplayMode) {
                case FfxLpmDisplayMode::FFX_LPM_DISPLAYMODE_LDR:
                    // LPM_CONFIG_709_709: false, false, false, false, false
                    break;
                case FfxLpmDisplayMode::FFX_LPM_DISPLAYMODE_FSHDR_2084:
                    // LPM_CONFIG_FS2RAWPQ_709: false, false, true, true, false
                    con2 = true;
                    clip = true;
                    break;
                case FfxLpmDisplayMode::FFX_LPM_DISPLAYMODE_FSHDR_SCRGB:
                    // LPM_CONFIG_FS2SCRGB_709: false, false, false, false, true
                    scaleOnly = true;
                    break;
                case FfxLpmDisplayMode::FFX_LPM_DISPLAYMODE_HDR10_2084:
                    // LPM_CONFIG_HDR10RAW_709: false, false, true, true, false
                    con2 = true;
                    clip = true;
                    break;
                case FfxLpmDisplayMode::FFX_LPM_DISPLAYMODE_HDR10_SCRGB:
                    // LPM_CONFIG_HDR10SCRGB_709: false, false, false, false,
                    // true
                    scaleOnly = true;
                    break;
            }
            break;
        case FfxLpmColorSpace::FFX_LPM_ColorSpace_P3:
            switch (ffxDisplayMode) {
                case FfxLpmDisplayMode::FFX_LPM_DISPLAYMODE_LDR:
                    // LPM_CONFIG_709_P3: true, true, false, false, false
                    con = true;
                    soft = true;
                    break;
                case FfxLpmDisplayMode::FFX_LPM_DISPLAYMODE_FSHDR_2084:
                    // LPM_CONFIG_FS2RAWPQ_P3: true, true, true, false, false
                    con = true;
                    soft = true;
                    con2 = true;
                    break;
                case FfxLpmDisplayMode::FFX_LPM_DISPLAYMODE_FSHDR_SCRGB:
                    // LPM_CONFIG_FS2SCRGB_P3: true, true, true, false, false
                    con = true;
                    soft = true;
                    con2 = true;
                    break;
                case FfxLpmDisplayMode::FFX_LPM_DISPLAYMODE_HDR10_2084:
                    // LPM_CONFIG_HDR10RAW_P3: false, false, true, true, false
                    con2 = true;
                    clip = true;
                    break;
                case FfxLpmDisplayMode::FFX_LPM_DISPLAYMODE_HDR10_SCRGB:
                    // LPM_CONFIG_HDR10SCRGB_P3: false, false, true, false,
                    // false
                    con2 = true;
                    break;
            }
            break;
        case FfxLpmColorSpace::FFX_LPM_ColorSpace_REC2020:
            switch (ffxDisplayMode) {
                case FfxLpmDisplayMode::FFX_LPM_DISPLAYMODE_LDR:
                    // LPM_CONFIG_709_2020: true, true, false, false, false
                    con = true;
                    soft = true;
                    break;
                case FfxLpmDisplayMode::FFX_LPM_DISPLAYMODE_FSHDR_2084:
                    // LPM_CONFIG_FS2RAWPQ_2020: true, true, true, false, false
                    con = true;
                    soft = true;
                    con2 = true;
                    break;
                case FfxLpmDisplayMode::FFX_LPM_DISPLAYMODE_FSHDR_SCRGB:
                    // LPM_CONFIG_FS2SCRGB_2020: true, true, true, false, false
                    con = true;
                    soft = true;
                    con2 = true;
                    break;
                case FfxLpmDisplayMode::FFX_LPM_DISPLAYMODE_HDR10_2084:
                    // LPM_CONFIG_HDR10RAW_2020: false, false, false, false,
                    // true
                    scaleOnly = true;
                    break;
                case FfxLpmDisplayMode::FFX_LPM_DISPLAYMODE_HDR10_SCRGB:
                    // LPM_CONFIG_HDR10SCRGB_2020: false, false, true, false,
                    // false
                    con2 = true;
                    break;
            }
            break;
    }

    FfxPopulateLpmConsts(
        con,
        soft,
        con2,
        clip,
        scaleOnly,
        outCon,
        outSoft,
        outCon2,
        outClip,
        outScaleOnly);

    lpmConstants.shoulder = (shoulderContrast != 1.0f) ? 1 : 0;
    lpmConstants.con = outCon;
    lpmConstants.soft = outSoft;
    lpmConstants.con2 = outCon2;
    lpmConstants.clip = outClip;
    lpmConstants.scaleOnly = outScaleOnly;
    lpmConstants.pad = 0;

    auto params_cb = create_constant_buffer(params, lpmConstants);
    MARK_DESTROY_NVRHI_RESOURCE(params_cb);
    program_vars["cbLPM"] = params_cb;

    auto image_size =
        GfVec2i(inputColor->getDesc().width, inputColor->getDesc().height);

    // Create linear clamp sampler
    nvrhi::SamplerDesc samplerDesc;
    samplerDesc.addressU = nvrhi::SamplerAddressMode::Clamp;
    samplerDesc.addressV = nvrhi::SamplerAddressMode::Clamp;
    samplerDesc.addressW = nvrhi::SamplerAddressMode::Clamp;
    auto linearClampSampler = resource_allocator.create(samplerDesc);
    MARK_DESTROY_NVRHI_RESOURCE(linearClampSampler);
    program_vars["s_LinearClamp"] = linearClampSampler;

    program_vars.finish_setting_vars();

    ComputeContext context(resource_allocator, program_vars);
    context.finish_setting_pso();

    context.begin();
    context.dispatch({}, program_vars, image_size[0], 16, image_size[1], 16);
    context.finish();

    params.set_output("Output Color", outputColor);
}

NODE_DECLARATION_UI(lpm);

NODE_DEF_CLOSE_SCOPE

#include "path.h"

#include <algorithm>
#include <limits>
#include <random>

#include "../surfaceInteraction.h"
RUZINO_NAMESPACE_OPEN_SCOPE
using namespace pxr;

VtValue PathIntegrator::Li(const GfRay& ray, std::default_random_engine& random)
{
    std::uniform_real_distribution<float> uniform_dist(
        0.0f, 1.0f - std::numeric_limits<float>::epsilon());
    std::function<float()> uniform_float = [&uniform_dist, &random]() {
        return uniform_dist(random);
    };

    auto color = EstimateOutGoingRadiance(ray, uniform_float, 0);

    return VtValue(GfVec3f(color[0], color[1], color[2]));
}

GfVec3f PathIntegrator::EstimateOutGoingRadiance(
    const GfRay& ray,
    const std::function<float()>& uniform_float,
    int recursion_depth)
{
    GfVec3f radiance(0.0f);
    GfVec3f throughput(1.0f);
    GfRay currentRay = ray;

    const int maxDepth = 5;
    const int rrStartDepth = 3;
    const float eps = 0.0001f;

    for (int depth = 0; depth < maxDepth; ++depth) {
        SurfaceInteraction si;

        if (!Intersect(currentRay, si)) {
            if (depth == 0) {
                GfVec3f lightHitPos;
                GfVec3f lightRadiance = IntersectLights(currentRay, lightHitPos);
                if (lightRadiance[0] > 0.0f ||
                    lightRadiance[1] > 0.0f ||
                    lightRadiance[2] > 0.0f) {
                    radiance += GfCompMult(throughput, lightRadiance);
                } else {
                    radiance += GfCompMult(throughput, IntersectDomeLight(currentRay));
                }
            }
            break;
        }

        if (GfDot(si.shadingNormal, currentRay.GetDirection()) > 0.0f) {
            si.flipNormal();
            si.PrepareTransforms();
        }

        GfVec3f direct = EstimateDirectLight(si, uniform_float);
        radiance += GfCompMult(throughput, direct);

        GfVec3f wi;
        float pdf = 0.0f;
        GfVec3f brdf = si.Sample(wi, pdf, uniform_float);

        if (pdf <= 0.0f) {
            break;
        }

        float cosTheta = GfDot(si.shadingNormal, wi);
        if (cosTheta <= 0.0f) {
            break;
        }

        throughput = GfCompMult(throughput, brdf) * (cosTheta / pdf);

        if (depth >= rrStartDepth) {
            float p = std::max(throughput[0], std::max(throughput[1], throughput[2]));
            p = std::min(0.95f, std::max(0.05f, p));

            if (uniform_float() > p) {
                break;
            }

            throughput /= p;
        }

        currentRay = GfRay(
            si.position + eps * si.geometricNormal,
            wi.GetNormalized());
    }

    return radiance;
}


RUZINO_NAMESPACE_CLOSE_SCOPE

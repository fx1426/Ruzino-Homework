#include <pxr/base/gf/matrix4f.h>
#include <pxr/base/gf/rotation.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdSkel/animQuery.h>
#include <pxr/usd/usdSkel/cache.h>
#include <pxr/usd/usdSkel/skeleton.h>

#include <memory>

#include "GCore/Components/MaterialComponent.h"
#include "GCore/Components/MeshOperand.h"
#include "GCore/Components/SkelComponent.h"
#include "GCore/Components/XformComponent.h"
#include "geom_node_base.h"
#include "pxr/usd/usdSkel/animation.h"
#include "pxr/usd/usdSkel/bindingAPI.h"
#include "pxr/usd/usdSkel/skeletonQuery.h"


NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(get_current_usd_path)
{
//    b.add_input<bool>("Print log").default_val(false);
    b.add_output<std::string>("USD file path");
}
NODE_EXECUTION_FUNCTION(get_current_usd_path)
{
    auto& global_payload = params.get_global_payload<GeomPayload&>();
//    bool log = params.get_input<bool>("Print log");
//    if (log) {
//            log::warning("Path in the payload is %s", global_payload.stage_filepath_.c_str());
//    }
    params.set_output("USD file path", global_payload.stage_filepath_);
    return true;
}
NODE_DECLARATION_UI(get_current_usd_path);
//NODE_DECLARATION_REQUIRED(get_current_usd_path);
NODE_DEF_CLOSE_SCOPE
#include <pxr/usd/usd/stage.h>

#include "GCore/Components/MeshComponent.h"
#include "geom_node_base.h"

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
    //            spdlog::warn("Path in the payload is %s",
    //            global_payload.stage_filepath_.c_str());
    //    }
    params.set_output("USD file path", global_payload.stage_filepath_);
    return true;
}
NODE_DECLARATION_UI(get_current_usd_path);
// NODE_DECLARATION_REQUIRED(get_current_usd_path);
NODE_DEF_CLOSE_SCOPE

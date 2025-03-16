USTC_CG_ADD_LIB(
    polyscope_widget
    SHARED
    PUBLIC_LIBS polyscope stage GUI usd usdGeom usdShade stage_listener
    PRIVATE_LIBS Logger RHI
    COMPILE_DEFS NOMINMAX=1 USTC_CG_BUILD_MODULE=1
)
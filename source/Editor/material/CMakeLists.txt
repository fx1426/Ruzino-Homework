if(USTC_CG_WITH_OPENUSD)

USTC_CG_ADD_LIB(
	MCore
	SHARED
	PRIVATE_LIBS 
		usd
		usdImaging
		nodes_system
		usdMtlx
		nodes_ui_imgui
		blueprints
		MaterialXGenShader

	COMPILE_DEFS
		NOMINMAX=1
)

endif()
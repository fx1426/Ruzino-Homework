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

	COMPILE_DEFS
		NOMINMAX=1
		USTC_CG_BUILD_MODULE=1
)

endif()
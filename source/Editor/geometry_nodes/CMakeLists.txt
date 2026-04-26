# Build compile definitions list
set(GEOMETRY_NODES_COMPILE_DEFS NOMINMAX)
if(ENABLE_GEOM_USD_EXTENSION)
	list(APPEND GEOMETRY_NODES_COMPILE_DEFS GEOM_USD_EXTENSION=1)
endif()

# Add shader directory definition
list(APPEND GEOMETRY_NODES_COMPILE_DEFS GEOM_NODES_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/")

add_nodes(
	TARGET_NAME geometry_nodes 
	SRC_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
	DEP_LIBS stage nodes_system usd geometry usdShade Eigen3::Eigen autodiff igl::core igl_restricted::triangle RZFemBem
	COMPILE_DEFS ${GEOMETRY_NODES_COMPILE_DEFS}
)

target_link_libraries(node_points_to_mesh PRIVATE usd hioOpenVDB)

add_library(tetgen STATIC ${CMAKE_CURRENT_SOURCE_DIR}/tetgen/tetgen.cxx ${CMAKE_CURRENT_SOURCE_DIR}/tetgen/tetgen.h ${CMAKE_CURRENT_SOURCE_DIR}/tetgen/predicates.cxx)
set_target_properties(tetgen PROPERTIES ${OUTPUT_DIR})
if(MSVC)
	target_compile_options(tetgen PRIVATE /wd4477)
endif()
target_link_libraries(node_tetgen PRIVATE tetgen)
target_include_directories(node_tetgen PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tetgen)

option(ENABLE_HW6_ARAP_OPENMP "Enable OpenMP acceleration for the HW6 ARAP local phase." ON)
if(ENABLE_HW6_ARAP_OPENMP)
	find_package(OpenMP)
	if(OpenMP_CXX_FOUND)
		target_link_libraries(hw6_arap PRIVATE OpenMP::OpenMP_CXX)
		target_compile_definitions(hw6_arap PRIVATE HW6_ARAP_ENABLE_OPENMP=1)
	else()
		message(WARNING "OpenMP was requested for hw6_arap, but CMake could not find OpenMP. The node will fall back to serial execution.")
	endif()
endif()

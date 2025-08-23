file(GLOB test_sources ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

foreach(source ${test_sources})
    UCG_ADD_APP(
        SRC ${source}
        LIBS usdview_widget nodes_system GUI usd nodes_ui_imgui geometry MCore usdImaging usdImagingGL stb_image
    )
endforeach()

add_dependencies(headless_render hd_USTC_CG)
add_dependencies(headless_render render_nodes)

add_dependencies(USTC_CG geometry_nodes)
add_dependencies(USTC_CG basic_nodes)
add_dependencies(USTC_CG optimization)
# add back the renderer
add_dependencies(USTC_CG hd_USTC_CG)
add_dependencies(USTC_CG render_nodes)
# add_dependencies(USTC_CG gl_based_render_nodes)
 add_dependencies(USTC_CG RZFemBemNodes)


if (USTC_CG_WITH_POLYSCOPE)
target_link_libraries(USTC_CG_polyscope PUBLIC polyscope_widget)

add_dependencies(USTC_CG_polyscope geometry_nodes)
add_dependencies(USTC_CG_polyscope basic_nodes)
add_dependencies(USTC_CG_polyscope polyscope_nodes)
add_dependencies(USTC_CG_polyscope optimization)

if (ENABLE_SUBMISSION)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../../submissions_inner)
        message(STATUS "[Inner submissions found]")
    
    add_dependencies(USTC_CG_polyscope submissions_inner_nodes)
    add_dependencies(USTC_CG submissions_inner_nodes)
    endif ()
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../../../submissions)
        message(STATUS "[Outer submissions found]")
    add_dependencies(USTC_CG_polyscope submissions_nodes)
    add_dependencies(USTC_CG submissions_nodes)
    endif ()
endif ()
endif (USTC_CG_WITH_POLYSCOPE)

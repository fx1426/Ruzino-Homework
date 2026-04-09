# CMakeLists.txt for GUI Python module
cmake_minimum_required(VERSION 3.12)
project(gui_python)

# Find required packages
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Find nanobind
if(NOT TARGET nanobind AND NOT TARGET nanobind-static)
    set(_ruzino_nanobind_config_dir
        "${CMAKE_SOURCE_DIR}/external/nanobind/cmake")

    if(EXISTS "${_ruzino_nanobind_config_dir}/nanobind-config.cmake")
        list(APPEND CMAKE_PREFIX_PATH "${_ruzino_nanobind_config_dir}")
    else()
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -m nanobind --cmake_dir
            RESULT_VARIABLE _ruzino_nanobind_result
            OUTPUT_STRIP_TRAILING_WHITESPACE
            OUTPUT_VARIABLE NB_DIR
            ERROR_QUIET)
        if(_ruzino_nanobind_result EQUAL 0 AND NB_DIR)
            list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
        endif()
    endif()

    find_package(nanobind CONFIG REQUIRED)
endif()

# Add the GUI library path (adjust as needed)
# set(GUI_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include")
# set(GUI_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../lib") 

# Create the Python module
nanobind_add_module(gui gui.cpp)

# Link against the GUI library
# target_link_libraries(gui PRIVATE gui_lib)

# Include directories
target_include_directories(gui PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    # Add other include paths as needed
)

# Compiler features
target_compile_features(gui PRIVATE cxx_std_17)

# Install the module
install(TARGETS gui LIBRARY DESTINATION .)

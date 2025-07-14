cmake_minimum_required(VERSION 3.29.6)

#
# Project details
#
project(
  "USTC_CG_Framework3D"
  VERSION 1.0
  LANGUAGES C CXX
)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set the relative path to the project root
file(RELATIVE_PATH RELATIVE_TO_PROJROOT ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
set(USTC_CG_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE STRING "Root directory of the project")

# Set Python variables BEFORE including any subdirectories
set(Python3_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SDK/python")
message(STATUS "Python3_ROOT_DIR is set to ${Python3_ROOT_DIR}")
set(Python3_FIND_STRATEGY LOCATION)
set(Python3_FIND_REGISTRY NEVER)
set(Python3_FIND_FRAMEWORK NEVER)

# Find Python early to ensure it's available for all subdirectories
find_package(Python3 COMPONENTS Development Interpreter REQUIRED)
message(STATUS "Found Python3 version: ${Python3_VERSION}")
message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
# Export Python3_EXECUTABLE to parent scope
set(Python3_EXECUTABLE ${Python3_EXECUTABLE} PARENT_SCOPE)

# Set up environment for nanobind to find the same Python
# Remove the Python3 to Python mapping, let nanobind find Python itself
set(Python_ROOT_DIR "${Python3_ROOT_DIR}")
set(Python_FIND_STRATEGY LOCATION)
set(Python_FIND_REGISTRY NEVER)
set(Python_FIND_FRAMEWORK NEVER)
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)


include(cmake/AddLibrary.cmake)
include(cmake/AddNodes.cmake)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug")
endif()

# For testing
enable_testing()
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/googletest)
set_target_properties(gtest PROPERTIES FOLDER "ThirdParty")
set_target_properties(gtest_main PROPERTIES FOLDER "ThirdParty")
set_target_properties(gmock PROPERTIES FOLDER "ThirdParty")
set_target_properties(gmock_main PROPERTIES FOLDER "ThirdParty")

set_target_properties(gtest PROPERTIES ${OUTPUT_DIR})
set_target_properties(gtest_main PROPERTIES ${OUTPUT_DIR})

option(USTC_CG_WITH_CUDA OFF)
option(USTC_CG_WITH_TORCH OFF)

if(USTC_CG_WITH_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 86)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
else()
  add_compile_definitions(USTC_CG_WITH_CUDA=0)
endif()

add_compile_definitions(BOOST_PYTHON_NO_LIB=1)

message(STATUS "Started CMake for ${PROJECT_NAME} v${PROJECT_VERSION}...\n")
set(BUILD_SHARED_LIBS ON)

# On mac, the cpp standard should >= 20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# if(MSVC)
#   add_compile_options(/utf-8)
# endif()

if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fms-extensions")
endif()


if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
  set(SDK_FOLDER ${CMAKE_BUILD_TYPE})
elseif(${CMAKE_BUILD_TYPE} MATCHES "RelWithDebInfo")
  set(SDK_FOLDER ${CMAKE_BUILD_TYPE})
else()
  set(SDK_FOLDER "Release")
endif()

message("CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}. Selecting SDK from ${SDK_FOLDER}")

if(USTC_CG_WITH_TORCH)
  set(Torch_DIR SDK/${SDK_FOLDER}/libtorch/share/cmake/Torch)
  find_package(Torch REQUIRED)
  add_compile_definitions(USTC_CG_WITH_TORCH=1)
else()
  add_compile_definitions(USTC_CG_WITH_TORCH=0)
endif()


set(pxr_DIR SDK/OpenUSD/${SDK_FOLDER})
set(FRAMEWORK3D_DIR ${USTC_CG_ROOT_DIR})
message("Looking for OpenUSD in ${pxr_DIR}")
find_package(pxr)

if(pxr_FOUND)
  set_target_properties(tf PROPERTIES INTERFACE_COMPILE_DEFINITIONS "TBB_SUPPRESS_DEPRECATED_MESSAGES=1")
endif()

if(pxr_FOUND)
  message("Found OpenUSD version ${pxr_VERSION}")
  find_package(OpenGL REQUIRED)
  set(USTC_CG_WITH_OPENUSD ON)

  if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    if(Python3_LIBRARY MATCHES "_d.lib$")
      set_target_properties(usd PROPERTIES INTERFACE_COMPILE_DEFINITIONS "TBB_USE_DEBUG=1; BOOST_DEBUG_PYTHON")
    else()
      set_target_properties(usd PROPERTIES INTERFACE_COMPILE_DEFINITIONS "TBB_USE_DEBUG=1")
    endif()
  endif()
else()
  set(USTC_CG_WITH_OPENUSD OFF)
endif()

# Include Slang
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if(WIN32)
  find_package(NVAPI REQUIRED)
endif()

set(SLANG_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/SDK/slang)
find_package(Slang REQUIRED COMPONENTS slang slang-rt gfx)

set(AgilitySDK_ROOT_DIR SDK/d3d12)
find_package(AgilitySDK REQUIRED)
add_subdirectory(external)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/source/Core)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/source/Editor)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/source/Runtime)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/source/Plugins)

add_subdirectory(tests)

option(ENABLE_SUBMISSION "Enable USTC Homework Submission" OFF)
# submission directory
if (ENABLE_SUBMISSION)
    set(editor_dir ${CMAKE_CURRENT_SOURCE_DIR}/source/Editor)
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/submissions_inner)
      message(STATUS [./SUBMISSION FOUND])
      add_subdirectory(submissions_inner)
    endif ()
    
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../submissions)
      message(STATUS [../SUBMISSION FOUND])
      get_filename_component(
        ABS_SUBMISSION
        ${CMAKE_CURRENT_SOURCE_DIR}/../submissions
        ABSOLUTE
        BASE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      )
      message(STATUS "[ABS DIR] " ${ABS_SUBMISSION})
    
      add_subdirectory(${ABS_SUBMISSION} "OuterSubmissions")
    endif ()
endif ()
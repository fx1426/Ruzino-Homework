include(FindPackageHandleStandardArgs)

# Define search paths based on user input and environment variables
set(AgilitySDK_SEARCH_DIR ${AgilitySDK_ROOT_DIR})
set(_agilitysdk_hint_paths
    ${AgilitySDK_SEARCH_DIR}/build/native/include
    ${AgilitySDK_SEARCH_DIR}/include
    D:/Windows Kits/10/Include
    C:/Program Files (x86)/Windows Kits/10/Include
)

if(DEFINED ENV{INCLUDE})
    file(TO_CMAKE_PATH "$ENV{INCLUDE}" _agilitysdk_env_include)
    list(APPEND _agilitysdk_hint_paths ${_agilitysdk_env_include})
endif()

file(GLOB _agilitysdk_windows_sdk_versions
    LIST_DIRECTORIES true
    "D:/Windows Kits/10/Include/*"
    "C:/Program Files (x86)/Windows Kits/10/Include/*")
list(APPEND _agilitysdk_hint_paths ${_agilitysdk_windows_sdk_versions})
##################################
# Find the AgilitySDK include dir
##################################

message("Searching for Agility SDK include directories...")
  
find_path(AgilitySDK_INCLUDE_DIRS d3d12.h
    PATHS ${_agilitysdk_hint_paths}
    PATH_SUFFIXES um)

find_package_handle_standard_args(AgilitySDK REQUIRED_VARS AgilitySDK_INCLUDE_DIRS)

##################################
# Create targets
##################################

message ("Agility SDK found. Include directory: ${AgilitySDK_INCLUDE_DIRS}")

if(NOT CMAKE_VERSION VERSION_LESS 3.0 AND AgilitySDK_FOUND)
add_library(agility_sdk INTERFACE)
set_target_properties(agility_sdk PROPERTIES
          INTERFACE_INCLUDE_DIRECTORIES  ${AgilitySDK_INCLUDE_DIRS}
          IMPORTED_IMPLIB  ${AgilitySDK_LIBRARIES}
          IMPORTED_LOCATION  ${AgilitySDK_LIBRARIES})
endif()

cmake_minimum_required(VERSION 3.10)

# set the project name
project(VACUUMMS)

option(BUILD_SHARED_LIBS "shared libs " ON)
option(BUILD_STATIC_LIBS "static libs " OFF)
option(BUILD_CUDA_COMPONENTS "FVI and PMDGPU require cuda" OFF)
option(BUILD_TIFF_UTILS "some utils require libtiff" OFF)

message("got value for BUILD_CUDA_COMPONENTS=${BUILD_CUDA_COMPONENTS}")
message("got value for BUILD_TIFF_UTILS=${BUILD_TIFF_UTILS}")

# Find X11. Usually installed to system area, but when it's not, need to find it.
find_package(X11)
message("X11_INCLUDE_DIR set to ${X11_INCLUDE_DIR}")
message("X11_FOUND set to ${X11_FOUND}")
message("X11_X11_LIB set to ${X11_X11_LIB}")
message("X11_SM_FOUND set to ${X11_SM_FOUND}")
message("X11_SM_LIB set to ${X11_SM_LIB}")
message("X11_Xext_LIB set to ${X11_Xext_LIB}")
message("X11_ICE_LIB set to ${X11_ICE_LIB}")
message("X11_LIBRARIES set to ${X11_LIBRARIES}")

#FTW
#list(PREPEND X11_LIBRARIES ${X11_ICE_LIB})
#message("X11_LIBRARIES now set to ${X11_LIBRARIES}")

#list(PREPEND X11_INCLUDE_DIR ${X11_ICE_INCLUDE_PATH})
#message("X11_INCLUDE_DIR now set to ${X11_INCLUDE_DIR}")

#add_library(X11 SHARED IMPORTED)
#set_target_properties(X11 PROPERTIES IMPORTED_LOCATION "${X11_LIBRARIES}")
#add_library(SM SHARED IMPORTED)
#set_target_properties(SM PROPERTIES IMPORTED_LOCATION "${X11_SM_LIB}")
#add_library(ICE SHARED IMPORTED)
#set_target_properties(ICE PROPERTIES IMPORTED_LOCATION "${X11_ICE_INCLUDE_PATH}")
#add_library(Xext SHARED IMPORTED)
#set_target_properties(Xext PROPERTIES IMPORTED_LOCATION "${X11_Xext_LIB}")


# Import the tiff library. This will make it available for linking. 
if (BUILD_TIFF_UTILS)
    find_package(TIFF)
    add_library(tiff SHARED IMPORTED)
    set_target_properties(tiff PROPERTIES IMPORTED_LOCATION "${TIFF_LIBRARIES}")
endif()

if (BUILD_CUDA_COMPONENTS)
    find_package(CUDA)
endif()

add_subdirectory(libraries)
message("added subdirectory lbiraries")
list(APPEND EXTRA_lIBS ftw_general )
add_subdirectory(utils)
add_subdirectory(applications)



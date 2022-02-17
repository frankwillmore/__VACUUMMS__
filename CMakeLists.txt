cmake_minimum_required(VERSION 3.10)

# set the project name
project(VACUUMMS)

option(BUILD_SHARED_LIBS "shared libs " ON)
option(BUILD_STATIC_LIBS "static libs " OFF)
option(BUILD_CUDA_COMPONENTS "FVI and PMDGPU require cuda" ON)
option(BUILD_TIFF_UTILS "some utils require libtiff" ON)

message("got value for BUILD_CUDA_COMPONENTS=${BUILD_CUDA_COMPONENTS}")
message("got value for BUILD_TIFF_UTILS=${BUILD_TIFF_UTILS}")

# Import the tiff library. This will make it available for linking. 
if (BUILD_TIFF_UTILS)
    find_package(TIFF)
    add_library(tiff SHARED IMPORTED)
    set_target_properties(tiff PROPERTIES IMPORTED_LOCATION ${TIFF_LIBRARIES})
endif()

add_subdirectory(libraries)
message("added subdirectory lbiraries")
list(APPEND EXTRA_lIBS ftw_general )
add_subdirectory(utils)
add_subdirectory(applications)



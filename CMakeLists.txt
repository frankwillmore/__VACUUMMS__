cmake_minimum_required(VERSION 3.10)

# set the project name
project(VACUUMMS)

option(BUILD_SHARED_LIBS "shared libs " ON)
option(BUILD_STATIC_LIBS "static libs " OFF)
option(BUILD_CUDA_COMPONENTS "FVI and PMDGPU require cuda" OFF)
option(BUILD_TIFF_UTILS "some utils require libtiff" ON)

add_subdirectory(libraries)
list(APPEND EXTRA_lIBS ftw_general)
add_subdirectory(utils)
add_subdirectory(applications)



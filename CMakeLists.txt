cmake_minimum_required(VERSION 3.10)

# set the project name
project(VACUUMMS)

option(USE_MYJUNK "Use junk provided" ON)
option(BUILD_SHARED_LIBS "shared libs " ON)
option(BUILD_STATIC_LIBS "static libs " OFF)
option(BUILD_CUDA_UTILS "FVI utils require cuda" OFF)
option(BUILD_TIFF_UTILS "FVI utils require cuda" ON)

add_subdirectory(libraries)
list(APPEND EXTRA_lIBS ftw_general)
add_subdirectory(utils)
add_subdirectory(applications)



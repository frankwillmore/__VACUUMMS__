cmake_minimum_required(VERSION 3.10)

# set the project name
project(VACUUMMS)

option(USE_MYJUNK "Use junk provided" ON)
option(BUILD_SHARED_LIBS "shared libs " ON)
option(BUILD_STATIC_LIBS "static libs " OFF)

add_subdirectory(libraries)
add_subdirectory(utils)

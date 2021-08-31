cmake_minimum_required(VERSION 3.10)

# set the project name
#project(VACUUMMS)
#project(utils)
#project(avg)

option(USE_MYJUNK "Use junk provided" ON)
option(BUILD_SHARED_LIBS "shared libs " ON)
option(BUILD_STATIC_LIBS "static libs " OFF)

# add the executable
add_executable(ddx "${PROJECT_SOURCE_DIR}/utils/avg.c")
target_include_directories(ddx PUBLIC "${PROJECT_SOURCE_DIR}/include")





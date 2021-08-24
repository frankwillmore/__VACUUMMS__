cmake_minimum_required(VERSION 3.10)

# set the project name
#project(VACUUMMS)
project(utils)
#project(avg)

# add the executable
add_executable(avg "${PROJECT_SOURCE_DIR}/src/utils/avg.c")

target_include_directories(avg PUBLIC
                           "${PROJECT_SOURCE_DIR}/include"
                           )

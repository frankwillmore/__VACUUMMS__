cmake_minimum_required(VERSION 3.10)

# set the project name
#project(VACUUMMS)
project(utils)
#project(avg)

option(USE_MYJUNK "Use junk provided" ON)
option(BUILD_SHARED_LIBS "shared libs " ON)
option(BUILD_STATIC_LIBS "static libs " OFF)

# add the executable
add_executable(avg "${PROJECT_SOURCE_DIR}/src/utils/avg.c")

target_include_directories(avg PUBLIC
                           "${PROJECT_SOURCE_DIR}/include"
                           )

add_library(general SHARED ${PROJECT_SOURCE_DIR}/libraries/general/ftw_prng.c)
set_target_properties(general PROPERTIES
                      POSITION_INDEPENDENT_CODE ${BUILD_SHARED_LIBS}
                      )
#../libraries/general/ftw_cav_parser.c           ../libraries/general/ftw_rng2.c
#../libraries/general/ftw_command_line_parser.c  ../libraries/general/ftw_rng.c
#../libraries/general/ftw_config_parser.c        ../libraries/general/ftw_std.c
#../libraries/general/ftw_param.c                ../libraries/general/test.c

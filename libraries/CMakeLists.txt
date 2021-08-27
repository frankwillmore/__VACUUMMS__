cmake_minimum_required(VERSION 3.10)

# set the project name
#project(VACUUMMS)
#project(utils)
#project(avg)

#option(USE_MYJUNK "Use junk provided" ON)
option(BUILD_SHARED_LIBS "shared libs " ON)
option(BUILD_STATIC_LIBS "static libs " OFF)


add_library(general SHARED
      ${PROJECT_SOURCE_DIR}/libraries/general/ftw_prng.c
      ${PROJECT_SOURCE_DIR}/libraries/general/ftw_cav_parser.c
      ${PROJECT_SOURCE_DIR}/libraries/general/ftw_cav_parser.c
      ${PROJECT_SOURCE_DIR}/libraries/general/ftw_rng2.c
      ${PROJECT_SOURCE_DIR}/libraries/general/ftw_cav_parser.c
      ${PROJECT_SOURCE_DIR}/libraries/general/ftw_rng2.c
      ${PROJECT_SOURCE_DIR}/libraries/general/ftw_command_line_parser.c  
      ${PROJECT_SOURCE_DIR}/libraries/general/ftw_rng.c
      ${PROJECT_SOURCE_DIR}/libraries/general/ftw_config_parser.c
      ${PROJECT_SOURCE_DIR}/libraries/general/ftw_std.c
      ${PROJECT_SOURCE_DIR}/libraries/general/ftw_param.c
      ${PROJECT_SOURCE_DIR}/libraries/general/test.c
)

set_target_properties(general PROPERTIES
                      POSITION_INDEPENDENT_CODE ${BUILD_SHARED_LIBS}
                      )

target_include_directories(general PUBLIC
                           "${PROJECT_SOURCE_DIR}/include"
                           )

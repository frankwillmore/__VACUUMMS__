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

add_library(ftw_pov SHARED
    ${PROJECT_SOURCE_DIR}/libraries/ftw_pov/ftw_povheader.c
)
target_include_directories(ftw_pov PUBLIC
    "${PROJECT_SOURCE_DIR}/include"
)

# optional libs

if (BUILD_TIFF_UTILS)
    find_package(TIFF)
    add_library(ftw_tiff SHARED
        ${PROJECT_SOURCE_DIR}/libraries/ftw_tiff/ftw_makeTIFF.c
    )
    target_include_directories(ftw_tiff PUBLIC
                           "${TIFF_INCLUDE_DIRS}"
                           )
    target_include_directories(ftw_tiff PUBLIC
                           "${PROJECT_SOURCE_DIR}/include"
                           )
endif()

if (BUILD_CUDA_UTILS)
   add_library(ftw_cuda SHARED
      ${PROJECT_SOURCE_DIR}/libraries/ftw_cuda/ftw_cuda_utils.cu
      ${PROJECT_SOURCE_DIR}/libraries/ftw_cuda/ftw_gfg2fvi.cu
   )
endif()

install(TARGETS general ftw_pov ftw_tiff ftw_cuda 
        CONFIGURATIONS Release 
        RUNTIME DESTINATION Release/lib)


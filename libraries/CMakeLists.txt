cmake_minimum_required(VERSION 3.10)

option(BUILD_SHARED_LIBS "shared libs " ON)
option(BUILD_STATIC_LIBS "static libs " OFF)

add_library(ftw_general SHARED
      ${PROJECT_SOURCE_DIR}/libraries/ftw_general/ftw_prng.c
      ${PROJECT_SOURCE_DIR}/libraries/ftw_general/ftw_cav_parser.c
      ${PROJECT_SOURCE_DIR}/libraries/ftw_general/ftw_cav_parser.c
      ${PROJECT_SOURCE_DIR}/libraries/ftw_general/ftw_rng2.c
      ${PROJECT_SOURCE_DIR}/libraries/ftw_general/ftw_cav_parser.c
      ${PROJECT_SOURCE_DIR}/libraries/ftw_general/ftw_rng2.c
      ${PROJECT_SOURCE_DIR}/libraries/ftw_general/ftw_command_line_parser.c  
      ${PROJECT_SOURCE_DIR}/libraries/ftw_general/ftw_rng.c
      ${PROJECT_SOURCE_DIR}/libraries/ftw_general/ftw_config_parser.c
      ${PROJECT_SOURCE_DIR}/libraries/ftw_general/ftw_std.c
      ${PROJECT_SOURCE_DIR}/libraries/ftw_general/ftw_param.c
)

list(APPEND EXTRA_LIBS ftw_general)

set_target_properties(ftw_general PROPERTIES
    POSITION_INDEPENDENT_CODE ${BUILD_SHARED_LIBS}
    )

target_include_directories(ftw_general PUBLIC
    "${PROJECT_SOURCE_DIR}/include"
    )

# ftw_pov

add_library(ftw_pov SHARED
    ${PROJECT_SOURCE_DIR}/libraries/ftw_pov/ftw_povheader.c
)
target_include_directories(ftw_pov PUBLIC
    "${PROJECT_SOURCE_DIR}/include"
)

# optional libs

if (BUILD_TIFF_UTILS)
    add_library(ftw_tiff SHARED
        ${PROJECT_SOURCE_DIR}/libraries/ftw_tiff/ftw_makeTIFF.c
    )
    list(APPEND EXTRA_LIBS ftw_tiff)

    target_link_libraries(ftw_tiff PUBLIC tiff)

    set_target_properties(ftw_tiff PROPERTIES
        POSITION_INDEPENDENT_CODE ${BUILD_SHARED_LIBS}
        )
    target_include_directories(ftw_tiff PUBLIC
        "${TIFF_INCLUDE_DIRS}"
        )
    target_include_directories(ftw_tiff PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
        )
endif() # BUILD_TIFF_UTILS


if (BUILD_CUDA_COMPONENTS)
   cuda_add_library(ftw_cuda SHARED
      ${PROJECT_SOURCE_DIR}/libraries/ftw_cuda/ftw_cuda_utils.cu
      ${PROJECT_SOURCE_DIR}/libraries/ftw_cuda/ftw_gfg2fvi.cu
   )
   target_include_directories(ftw_cuda PUBLIC
      "${PROJECT_SOURCE_DIR}/include"
   )
endif()

install(TARGETS ftw_general ftw_pov
        DESTINATION lib)

if (BUILD_TIFF_UTILS)
    install(TARGETS ftw_tiff 
        DESTINATION lib)
endif()

if (BUILD_CUDA_COMPONENTS)
    install(TARGETS ftw_cuda
        DESTINATION lib)
endif()


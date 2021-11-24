cmake_minimum_required(VERSION 3.10)

option(BUILD_SHARED_LIBS "shared libs " ON)
option(BUILD_STATIC_LIBS "static libs " OFF)

add_executable(rattle "${PROJECT_SOURCE_DIR}/applications/rattle/rattle.c")
target_include_directories(rattle PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(rattle PUBLIC ftw_general m pthread)

add_executable(ddx "${PROJECT_SOURCE_DIR}/applications/ddx/ddx.c" "${PROJECT_SOURCE_DIR}/applications/ddx/io_setup.c" )
target_include_directories(ddx PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(ddx PUBLIC ftw_general m pthread)

add_executable(ljx 
    "${PROJECT_SOURCE_DIR}/applications/ljx/ljx_main.c"
    "${PROJECT_SOURCE_DIR}/applications/ljx/command_line_parser.c"
    "${PROJECT_SOURCE_DIR}/applications/ljx/io_setup.c"
    "${PROJECT_SOURCE_DIR}/applications/ljx/energy.c"
    "${PROJECT_SOURCE_DIR}/applications/ljx/graphics.c"
)
target_include_directories(ljx PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(ljx PUBLIC ftw_general m pthread X11)

add_executable(center "${PROJECT_SOURCE_DIR}/applications/center/center.c")
target_include_directories(center PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(center PUBLIC ftw_general m pthread)

# Needs CUDA
#add_executable(fv "${PROJECT_SOURCE_DIR}/applications/fv/fv.cu")
#target_include_directories(fv PUBLIC "${PROJECT_SOURCE_DIR}/include")
#target_link_libraries(fv PUBLIC ftw_general m pthread)

add_executable(mfp
    "${PROJECT_SOURCE_DIR}/applications/mfp/mfp.c"
    "${PROJECT_SOURCE_DIR}/applications/mfp/io_setup.c"
)
target_include_directories(mfp PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(mfp PUBLIC ftw_general m pthread)

add_executable(pddx "${PROJECT_SOURCE_DIR}/applications/ddx/ddx.c" "${PROJECT_SOURCE_DIR}/applications/ddx/io_setup.c" )
target_include_directories(pddx PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pddx PUBLIC ftw_general m pthread)

add_executable(hs 
    "${PROJECT_SOURCE_DIR}/applications/hs/hs_main.c" 
    "${PROJECT_SOURCE_DIR}/applications/hs/io_setup.c" 
#    "${PROJECT_SOURCE_DIR}/applications/hs/command_line_parser.c" 
    "${PROJECT_SOURCE_DIR}/applications/hs/graphics.c" 
)
target_include_directories(hs PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(hs PUBLIC ftw_general m pthread X11)


#===

#熊俊傑@jlselogin4:~/VACUUMMS$ ls applications/ddx/
#cg.gfg               ddx.c                fcc_generator.c      Makefile             
#crystal_generator.c  .ddx.c.swp           fcc.gfg              pddx.c               
#dddx.c               ddx.h                io_setup.c           pddx.h               
#dddx.h               fcc.cav              io_setup.h           trace_ddx.c          

#eg        ftw_samples  grafv      launcher  matrix  numa    utrw
#csd             essence   greetings  lj        mfp     pmd     sdhs
#clep     examples  genels       hs         mlp     pmdgpu  size

install(TARGETS rattle ddx ljx center mfp pddx hs 
        DESTINATION bin)

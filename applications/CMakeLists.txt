cmake_minimum_required(VERSION 3.10)

# set the project name
#project(VACUUMMS)
#project(utils)
#project(avg)

option(USE_MYJUNK "Use junk provided" ON)
option(BUILD_SHARED_LIBS "shared libs " ON)
option(BUILD_STATIC_LIBS "static libs " OFF)

#add_executable(rattle "${PROJECT_SOURCE_DIR}/applications/rattle.c")
#target_include_directories(rattle PUBLIC "${PROJECT_SOURCE_DIR}/include")

add_executable(ddx "${PROJECT_SOURCE_DIR}/applications/ddx/ddx.c" "${PROJECT_SOURCE_DIR}/applications/ddx/io_setup.c" )
target_include_directories(ddx PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(ddx PUBLIC ftw_general m pthread)


#add_executable(ljx "${PROJECT_SOURCE_DIR}/applications/ljx.c")
#target_include_directories(ljx PUBLIC "${PROJECT_SOURCE_DIR}/include")

#add_executable(center "${PROJECT_SOURCE_DIR}/applications/center.c")
#target_include_directories(center PUBLIC "${PROJECT_SOURCE_DIR}/include")

#add_executable(fv "${PROJECT_SOURCE_DIR}/applications/fv.c")
#target_include_directories(fv PUBLIC "${PROJECT_SOURCE_DIR}/include")

#add_executable(mfp "${PROJECT_SOURCE_DIR}/applications/mfp.c")
#target_include_directories(mfp PUBLIC "${PROJECT_SOURCE_DIR}/include")




#熊俊傑@jlselogin4:~/VACUUMMS$ ls applications/ddx/
#cg.gfg               ddx.c                fcc_generator.c      Makefile             
#crystal_generator.c  .ddx.c.swp           fcc.gfg              pddx.c               
#dddx.c               ddx.h                io_setup.c           pddx.h               
#dddx.h               fcc.cav              io_setup.h           trace_ddx.c          




#eg        ftw_samples  grafv      launcher  matrix  numa    utrw
#csd             essence   greetings  lj        mfp     pmd     sdhs
#clep     examples  genels       hs         mlp     pmdgpu  size




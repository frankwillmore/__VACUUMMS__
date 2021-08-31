cmake_minimum_required(VERSION 3.10)

# set the project name
#project(VACUUMMS)
#project(utils)
#project(avg)

option(USE_MYJUNK "Use junk provided" ON)
option(BUILD_SHARED_LIBS "shared libs " ON)
option(BUILD_STATIC_LIBS "static libs " OFF)

add_executable(rattle "${PROJECT_SOURCE_DIR}/applications/rattle.c")
target_include_directories(rattle PUBLIC "${PROJECT_SOURCE_DIR}/include")

add_executable(ddx "${PROJECT_SOURCE_DIR}/applications/ddx.c")
target_include_directories(ddx PUBLIC "${PROJECT_SOURCE_DIR}/include")

add_executable(ljx "${PROJECT_SOURCE_DIR}/applications/ljx.c")
target_include_directories(ljx PUBLIC "${PROJECT_SOURCE_DIR}/include")

add_executable(center "${PROJECT_SOURCE_DIR}/applications/center.c")
target_include_directories(center PUBLIC "${PROJECT_SOURCE_DIR}/include")

add_executable(fv "${PROJECT_SOURCE_DIR}/applications/fv.c")
target_include_directories(fv PUBLIC "${PROJECT_SOURCE_DIR}/include")

add_executable(mfp "${PROJECT_SOURCE_DIR}/applications/mfp.c")
target_include_directories(mfp PUBLIC "${PROJECT_SOURCE_DIR}/include")

#add_executable( "${PROJECT_SOURCE_DIR}/applications/.c")
#target_include_directories( PUBLIC "${PROJECT_SOURCE_DIR}/include")

#add_executable( "${PROJECT_SOURCE_DIR}/applications/.c")
#target_include_directories( PUBLIC "${PROJECT_SOURCE_DIR}/include")



#eg        ftw_samples  grafv      launcher  matrix  numa    utrw
#csd             essence   greetings  lj        mfp     pmd     sdhs
#clep     examples  genels       hs         mlp     pmdgpu  size




cmake_minimum_required(VERSION 3.10)

option(BUILD_SHARED_LIBS "shared libs " ON)
option(BUILD_STATIC_LIBS "static libs " OFF)

add_executable(rattle "${PROJECT_SOURCE_DIR}/applications/rattle/rattle.c")
target_include_directories(rattle PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(rattle PUBLIC ftw_general m pthread)

add_executable(ljx 
    "${PROJECT_SOURCE_DIR}/applications/ljx/ljx_main.c"
    "${PROJECT_SOURCE_DIR}/applications/ljx/command_line_parser.c"
    "${PROJECT_SOURCE_DIR}/applications/ljx/io_setup.c"
    "${PROJECT_SOURCE_DIR}/applications/ljx/energy.c"
    "${PROJECT_SOURCE_DIR}/applications/ljx/graphics.c"
)
target_include_directories(ljx PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(ljx PUBLIC ftw_general m pthread X11)


add_executable(hs 
    "${PROJECT_SOURCE_DIR}/applications/hs/hs_main.c"
    "${PROJECT_SOURCE_DIR}/applications/hs/io_setup.c"
    "${PROJECT_SOURCE_DIR}/applications/hs/graphics.c"
)
target_include_directories(hs PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(hs PUBLIC ftw_general m pthread X11)

add_executable(center "${PROJECT_SOURCE_DIR}/applications/center/center.c")
target_include_directories(center PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(center PUBLIC ftw_general m pthread)

if(BUILD_CUDA_COMPONENTS)
#    This is a non-working code, headers were lost
#    cuda_add_executable(fv "${PROJECT_SOURCE_DIR}/applications/fv/fv.cu")
#    target_include_directories(fv PUBLIC "${PROJECT_SOURCE_DIR}/include")
#    target_link_libraries(fv PUBLIC ftw_general m pthread)
endif()

add_executable(mfp
    "${PROJECT_SOURCE_DIR}/applications/mfp/mfp.c"
    "${PROJECT_SOURCE_DIR}/applications/mfp/io_setup.c"
)
target_include_directories(mfp PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(mfp PUBLIC ftw_general m pthread)

add_executable(ddx "${PROJECT_SOURCE_DIR}/applications/ddx/ddx.c" "${PROJECT_SOURCE_DIR}/applications/ddx/io_setup.c" )
target_include_directories(ddx PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(ddx PUBLIC ftw_general m)

add_executable(pddx "${PROJECT_SOURCE_DIR}/applications/ddx/ddx.c" "${PROJECT_SOURCE_DIR}/applications/ddx/io_setup.c" )
target_include_directories(pddx PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pddx PUBLIC ftw_general m pthread)

# needs work
#/usr/bin/ld: CMakeFiles/2pc.dir/2pc/io_setup.c.o: in function `loadConfiguration':
#io_setup.c:(.text+0x1c): undefined reference to `number_of_molecules'
#add_executable(2pc 
#    "${PROJECT_SOURCE_DIR}/applications/2pc/2pc.c" 
#    "${PROJECT_SOURCE_DIR}/applications/2pc/io_setup.c" 
#)
#target_include_directories(2pc PUBLIC "${PROJECT_SOURCE_DIR}/include")
#target_link_libraries(2pc PUBLIC ftw_general m pthread)

add_executable(csd 
    "${PROJECT_SOURCE_DIR}/applications/csd/csd.c" 
    "${PROJECT_SOURCE_DIR}/applications/csd/command_line_parser.c" 
)
target_include_directories(csd PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(csd PUBLIC ftw_general m pthread)

add_executable(rog 
    "${PROJECT_SOURCE_DIR}/applications/rog/rog.c" 
)
target_include_directories(rog PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(rog PUBLIC ftw_general m pthread)

add_executable(cav2cluster 
    "${PROJECT_SOURCE_DIR}/applications/cav2cluster/cav2cluster.c" 
)
target_include_directories(cav2cluster PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(cav2cluster PUBLIC ftw_general m )

add_executable(sdhs 
    "${PROJECT_SOURCE_DIR}/applications/sdhs/sdhs_main.c" 
    "${PROJECT_SOURCE_DIR}/applications/sdhs/io_setup.c" 
    "${PROJECT_SOURCE_DIR}/applications/sdhs/graphics.c" 
)
target_include_directories(sdhs PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(sdhs PUBLIC ftw_general m X11 )

add_executable(end2end 
    "${PROJECT_SOURCE_DIR}/applications/end2end/end2end.c" 
)
target_include_directories(end2end PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(end2end PUBLIC ftw_general m pthread)

# Cuda codes needing work (prolly just with CUDA): pmd pmdgpu fv/essence
# Unknown status: eg size
# sv - looks to have surface/volume utils, but needs another look

install(TARGETS 
        rattle 
        ljx 
        center 
        mfp 
        ddx 
        pddx 
        hs 
#        2pc         - missing symbol, see above note
        csd    
        rog  
        cav2cluster
        sdhs
        end2end
        DESTINATION bin)

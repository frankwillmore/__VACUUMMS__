cmake_minimum_required(VERSION 3.10)

option(BUILD_SHARED_LIBS "shared libs " ON)
option(BUILD_STATIC_LIBS "static libs " OFF)

# add the executable
add_executable(avg "${PROJECT_SOURCE_DIR}/utils/avg.c")
target_include_directories(avg PUBLIC "${PROJECT_SOURCE_DIR}/include")

add_executable(vis2pov "${PROJECT_SOURCE_DIR}/utils/vis2pov.c")
target_include_directories(vis2pov PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(vis2pov PUBLIC ftw_general m pthread)

add_executable(cav2vis "${PROJECT_SOURCE_DIR}/utils/cav2vis.c")
target_include_directories(cav2vis PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(cav2vis PUBLIC ftw_general m pthread)

add_executable(cfg2vis "${PROJECT_SOURCE_DIR}/utils/cfg2vis.c")
target_include_directories(cfg2vis PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(cfg2vis PUBLIC ftw_general m pthread)

add_executable(add "${PROJECT_SOURCE_DIR}/utils/add.c")
target_include_directories(add PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(add PUBLIC ftw_general m pthread)





add_executable(dst2hst "${PROJECT_SOURCE_DIR}/utils/dst2hst.c")
target_include_directories(dst2hst PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(dst2hst PUBLIC ftw_general m pthread)

add_executable(uniq "${PROJECT_SOURCE_DIR}/utils/uniq.c")
target_include_directories(uniq PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(uniq PUBLIC ftw_general m pthread)

add_executable(a2b "${PROJECT_SOURCE_DIR}/utils/a2b.c")
target_include_directories(a2b PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(a2b PUBLIC ftw_general m pthread)

add_executable(clustercat "${PROJECT_SOURCE_DIR}/utils/clustercat.c")
target_include_directories(clustercat PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(clustercat PUBLIC ftw_general m pthread)

add_executable(smooth "${PROJECT_SOURCE_DIR}/utils/smooth.c")
target_include_directories(smooth PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(smooth PUBLIC ftw_general m pthread)

add_executable(normalize "${PROJECT_SOURCE_DIR}/utils/normalize.c")
target_include_directories(normalize PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(normalize PUBLIC ftw_general m pthread)

add_executable(cfg2gfg "${PROJECT_SOURCE_DIR}/utils/cfg2gfg.c")
target_include_directories(cfg2gfg PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(cfg2gfg PUBLIC ftw_general m pthread)

add_executable(nsplit "${PROJECT_SOURCE_DIR}/utils/nsplit.c")
target_include_directories(nsplit PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(nsplit PUBLIC ftw_general m pthread)

add_executable(cram "${PROJECT_SOURCE_DIR}/utils/cram.c")
target_include_directories(cram PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(nsplit PUBLIC ftw_general m pthread)

# vacuumms executable simply returns the version number for the version in play
add_executable(vacuumms "${PROJECT_SOURCE_DIR}/utils/vacuumms.c")

##fvi2tiff loge log10 exp sum pow sq ck cram wdst2hst max min miss std sew povheader gfg2pov gfgc2pov fvi2pov cav2pov stack_tiffs expr_add expr_multiply dwf sqrt stream2slice truncate stream_multiply replicate_gfg replicate_cav sgfg2fvi

# vis2pov cav2vis cfg2vis add dst2hst uniq avg a2b clustercat smooth normalize cfg2gfg nsplit fvi2tiff loge log10 exp sum pow sq ck cram wdst2hst max min miss std sew povheader gfg2pov gfgc2pov fvi2pov cav2pov stack_tiffs expr_add expr_multiply dwf sqrt stream2slice truncate stream_multiply replicate_gfg replicate_cav sgfg2fvi


#install(TARGETS avg vis2pov cav2vis cfg2vis add dst2hst uniq a2b clustercat smooth normalize cfg2gfg nsplit
install(TARGETS avg vis2pov cav2vis cfg2vis add dst2hst 
        uniq a2b clustercat smooth normalize cfg2gfg nsplit
        vacuumms
        DESTINATION bin)
 

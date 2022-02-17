cmake_minimum_required(VERSION 3.10)

option(BUILD_SHARED_LIBS "shared libs " ON)
option(BUILD_STATIC_LIBS "static libs " OFF)

# add the executable
add_executable(avg "${PROJECT_SOURCE_DIR}/utils/avg.c")
target_include_directories(avg PUBLIC "${PROJECT_SOURCE_DIR}/include")

add_executable(vis2pov "${PROJECT_SOURCE_DIR}/utils/vis2pov.c")
target_include_directories(vis2pov PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(vis2pov PUBLIC ftw_general m )

add_executable(cav2vis "${PROJECT_SOURCE_DIR}/utils/cav2vis.c")
target_include_directories(cav2vis PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(cav2vis PUBLIC ftw_general m )

add_executable(cfg2vis "${PROJECT_SOURCE_DIR}/utils/cfg2vis.c")
target_include_directories(cfg2vis PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(cfg2vis PUBLIC ftw_general m )

add_executable(add "${PROJECT_SOURCE_DIR}/utils/add.c")
target_include_directories(add PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(add PUBLIC ftw_general m )

add_executable(dst2hst "${PROJECT_SOURCE_DIR}/utils/dst2hst.c")
target_include_directories(dst2hst PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(dst2hst PUBLIC ftw_general m )

add_executable(uniq "${PROJECT_SOURCE_DIR}/utils/uniq.c")
target_include_directories(uniq PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(uniq PUBLIC ftw_general m )

add_executable(a2b "${PROJECT_SOURCE_DIR}/utils/a2b.c")
target_include_directories(a2b PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(a2b PUBLIC ftw_general m )

add_executable(clustercat "${PROJECT_SOURCE_DIR}/utils/clustercat.c")
target_include_directories(clustercat PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(clustercat PUBLIC ftw_general m )

add_executable(smooth "${PROJECT_SOURCE_DIR}/utils/smooth.c")
target_include_directories(smooth PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(smooth PUBLIC ftw_general m )

add_executable(normalize "${PROJECT_SOURCE_DIR}/utils/normalize.c")
target_include_directories(normalize PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(normalize PUBLIC ftw_general m )

add_executable(cfg2gfg "${PROJECT_SOURCE_DIR}/utils/cfg2gfg.c")
target_include_directories(cfg2gfg PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(cfg2gfg PUBLIC ftw_general m )

add_executable(nsplit "${PROJECT_SOURCE_DIR}/utils/nsplit.c")
target_include_directories(nsplit PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(nsplit PUBLIC ftw_general m )

add_executable(cram "${PROJECT_SOURCE_DIR}/utils/cram.c")
target_include_directories(cram PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(cram PUBLIC ftw_general m )

add_executable(loge "${PROJECT_SOURCE_DIR}/utils/loge.c")
target_include_directories(loge PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(loge PUBLIC ftw_general m )

add_executable(log10 "${PROJECT_SOURCE_DIR}/utils/log10.c")
target_include_directories(log10 PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(log10 PUBLIC ftw_general m )

add_executable(exp "${PROJECT_SOURCE_DIR}/utils/exp.c")
target_include_directories(exp PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(exp PUBLIC ftw_general m )

add_executable(pow "${PROJECT_SOURCE_DIR}/utils/pow.c")
target_include_directories(pow PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pow PUBLIC ftw_general m )

add_executable(sum "${PROJECT_SOURCE_DIR}/utils/sum.c")
target_include_directories(sum PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(sum PUBLIC ftw_general m )

add_executable(sq "${PROJECT_SOURCE_DIR}/utils/sq.c")
target_include_directories(sq PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(sq PUBLIC ftw_general m )

add_executable(recip "${PROJECT_SOURCE_DIR}/utils/recip.c")
target_include_directories(recip PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(recip PUBLIC ftw_general m )

add_executable(ck "${PROJECT_SOURCE_DIR}/utils/ck.c")
target_include_directories(ck PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(ck PUBLIC ftw_general m pthread)

add_executable(wdst2hst "${PROJECT_SOURCE_DIR}/utils/wdst2hst.c")
target_include_directories(wdst2hst PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(wdst2hst PUBLIC ftw_general m pthread)

add_executable(max "${PROJECT_SOURCE_DIR}/utils/max.c")
target_include_directories(max PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(max PUBLIC ftw_general m pthread)

add_executable(min "${PROJECT_SOURCE_DIR}/utils/min.c")
target_include_directories(min PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(min PUBLIC ftw_general m pthread)

add_executable(miss "${PROJECT_SOURCE_DIR}/utils/miss.c")
target_include_directories(miss PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(miss PUBLIC ftw_general m pthread)

add_executable(std "${PROJECT_SOURCE_DIR}/utils/std.c")
target_include_directories(std PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(std PUBLIC ftw_general m pthread)

add_executable(sew "${PROJECT_SOURCE_DIR}/utils/sew.c")
target_include_directories(sew PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(sew PUBLIC ftw_general m pthread)

add_executable(povheader "${PROJECT_SOURCE_DIR}/utils/povheader.c")
target_include_directories(povheader PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(povheader PUBLIC ftw_general m pthread)

add_executable(gfg2pov "${PROJECT_SOURCE_DIR}/utils/gfg2pov.c")
target_include_directories(gfg2pov PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(gfg2pov PUBLIC ftw_general ftw_pov m pthread)

add_executable(gfgc2pov "${PROJECT_SOURCE_DIR}/utils/gfgc2pov.c")
target_include_directories(gfgc2pov PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(gfgc2pov PUBLIC ftw_general ftw_pov m pthread)

add_executable(fvi2pov "${PROJECT_SOURCE_DIR}/utils/fvi2pov.c")
target_include_directories(fvi2pov PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(fvi2pov PUBLIC ftw_general ftw_pov m pthread)

add_executable(cav2pov "${PROJECT_SOURCE_DIR}/utils/cav2pov.c")
target_include_directories(cav2pov PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(cav2pov PUBLIC ftw_general ftw_pov m pthread)

# TIFF
find_package(TIFF)

add_executable(stack_tiffs "${PROJECT_SOURCE_DIR}/utils/stack_tiffs.c")
target_include_directories(stack_tiffs PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_include_directories(stack_tiffs PUBLIC "${TIFF_INCLUDE_DIRS}")
target_link_libraries(stack_tiffs PUBLIC ftw_general m ftw_tiff tiff)

add_executable(readtiff "${PROJECT_SOURCE_DIR}/utils/readtiff.c")
target_include_directories(readtiff PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_include_directories(readtiff PUBLIC "${TIFF_INCLUDE_DIRS}")
target_link_libraries(readtiff PUBLIC ftw_general m ftw_tiff tiff)
# /TIFF

add_executable(expr_add "${PROJECT_SOURCE_DIR}/utils/expr_add.c")
target_include_directories(expr_add PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(expr_add PUBLIC ftw_general m )

add_executable(expr_multiply "${PROJECT_SOURCE_DIR}/utils/expr_multiply.c")
target_include_directories(expr_multiply PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(expr_multiply PUBLIC ftw_general m )

add_executable(dwf "${PROJECT_SOURCE_DIR}/utils/dwf.c")
target_include_directories(dwf PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(dwf PUBLIC ftw_general m )

add_executable(sqrt "${PROJECT_SOURCE_DIR}/utils/sqrt.c")
target_include_directories(sqrt PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(sqrt PUBLIC ftw_general m )

add_executable(stream2slice "${PROJECT_SOURCE_DIR}/utils/stream2slice.c")
target_include_directories(stream2slice PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(stream2slice PUBLIC ftw_general m pthread)

add_executable(truncate "${PROJECT_SOURCE_DIR}/utils/truncate.c")
target_include_directories(truncate PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(truncate PUBLIC ftw_general m pthread)

add_executable(stream_multiply "${PROJECT_SOURCE_DIR}/utils/stream_multiply.c")
target_include_directories(stream_multiply PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(stream_multiply PUBLIC ftw_general m pthread)

add_executable(replicate_gfg "${PROJECT_SOURCE_DIR}/utils/replicate_gfg.c")
target_include_directories(replicate_gfg PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(replicate_gfg PUBLIC ftw_general m pthread)

add_executable(replicate_cav "${PROJECT_SOURCE_DIR}/utils/replicate_cav.c")
target_include_directories(replicate_cav PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(replicate_cav PUBLIC ftw_general m pthread)

add_executable(sgfg2fvi "${PROJECT_SOURCE_DIR}/utils/sgfg2fvi.c")
target_include_directories(sgfg2fvi PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(sgfg2fvi PUBLIC ftw_general m pthread)

add_executable(gfg2fvi "${PROJECT_SOURCE_DIR}/utils/gfg2fvi.c")
target_include_directories(gfg2fvi PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(gfg2fvi PUBLIC ftw_general m pthread)

#add_executable(fvi2tiff "${PROJECT_SOURCE_DIR}/utils/fvi2tiff.c")
#target_include_directories(fvi2tiff PUBLIC "${PROJECT_SOURCE_DIR}/include")
#target_link_libraries(fvi2tiff PUBLIC ftw_general m pthread)

# vacuumms executable simply returns the version number for the version in play
add_executable(vacuumms "${PROJECT_SOURCE_DIR}/utils/vacuumms.c")

# These are all of the utils in the original distribution
# vis2pov cav2vis cfg2vis add dst2hst uniq avg a2b clustercat smooth normalize cfg2gfg nsplit fvi2tiff loge log10 exp sum pow sq ck cram wdst2hst max min miss std sew povheader gfg2pov gfgc2pov fvi2pov cav2pov stack_tiffs expr_add expr_multiply dwf sqrt stream2slice truncate stream_multiply replicate_gfg replicate_cav sgfg2fvi

install(TARGETS 
        avg 
        vis2pov 
        cav2vis 
        cfg2vis 
        add 
        dst2hst 
        uniq 
        a2b 
        clustercat 
        smooth 
        normalize 
        cfg2gfg 
        nsplit
        cram
        loge
        log10
        exp
        sum
        sq
        ck
        wdst2hst 
        max
        min
        miss
        std
        sew
        pow
        povheader
        gfg2pov
        gfgc2pov
        fvi2pov
        cav2pov
#        stack_tiffs
        expr_add
        expr_multiply
        dwf
        sqrt
        stream2slice
        truncate
#        fvi2tiff 
        stream_multiply 
        replicate_gfg 
        replicate_cav 
        sgfg2fvi
        gfg2fvi
        vacuumms
        DESTINATION bin)

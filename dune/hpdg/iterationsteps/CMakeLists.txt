add_subdirectory(mg)
install(FILES
dualdgmultigridstep.hh
dynamicblockgs.hh
DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/hpdg/iterationsteps)

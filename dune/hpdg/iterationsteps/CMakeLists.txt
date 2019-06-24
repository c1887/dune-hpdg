add_subdirectory(mg)
  coarsegrid.hh
  dualdgmultigridstep.hh
  dynamicblockgs.hh
  l1projected.hh
  l1smoother.hh
  parmgsetup.hh
  partnnmgstep.hh
install(FILES
DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/hpdg/iterationsteps)

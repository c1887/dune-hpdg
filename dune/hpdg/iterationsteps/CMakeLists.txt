add_subdirectory(mg)
install(FILES
  coarsegrid.hh
  dualdgmultigridstep.hh
  dynamicblockgs.hh
  l1projected.hh
  l1smoother.hh
  parmgsetup.hh
  partnnmgstep.hh
DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/hpdg/iterationsteps)

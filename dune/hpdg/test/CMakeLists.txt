install(FILES
  testobjects.hh
  utilities.hh
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/hpdg/test)

dune_add_test(
  SOURCES test_blockwiseoperations.cc)
dune_add_test(
  SOURCES test_arraytotuple.cc)
dune_add_test(
  SOURCES test_makedgtransfertuple.cc)
dune_add_test(
  SOURCES test_dgmultigridstep.cc)
dune_add_test(
  SOURCES test_geometricdgmultigrid.cc)
dune_add_test(
  SOURCES test_dynamicbasis.cc)
dune_add_test(
  SOURCES test_matrixwindow.cc)
dune_add_test(
  SOURCES test_dynamicbcrs.cc)
dune_add_test(
  SOURCES test_dynamicblockgs.cc)
dune_add_test(
  SOURCES test_gausskronrodrule.cc)

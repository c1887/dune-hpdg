install(FILES
  testobjects.hh
  utilities.hh
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/hpdg/test)

dune_add_test(
  SOURCES test_blockwiseoperations.cc)
dune_add_test(
  SOURCES test_arraytotuple.cc)
dune_add_test(
  SOURCES test_caches.cc)
dune_add_test(
  SOURCES test_commandlinept.cc)
dune_add_test(
  SOURCES test_discretizationerror.cc)
dune_add_test(
  SOURCES test_dynamicbasis.cc)
dune_add_test(
  SOURCES test_lumpedmass.cc)
dune_add_test(
  SOURCES test_matrixwindow.cc)
dune_add_test(
  SOURCES test_dynamicbcrs.cc)
dune_add_test(
  SOURCES test_dynamicblockgs.cc)
dune_add_test(
  SOURCES test_gausskronrodrule.cc)
dune_add_test(
  SOURCES test_ipdg_assembler.cc)
dune_add_test(
  SOURCES test_integertostring.cc)
dune_add_test(
  SOURCES testestimatorutility.cc)
dune_add_test(
  SOURCES test_persistentgvdatatransfer.cc)
dune_add_test(
  SOURCES test_solversetup.cc)

add_subdirectory(test)
add_subdirectory(localoperators)
install_files(
  operator.hh
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/hpdg/matrix-free)

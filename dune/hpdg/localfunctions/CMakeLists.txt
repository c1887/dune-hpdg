add_subdirectory(lagrange)

install(FILES
  qkdglegendre.hh
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/hpdg/localfunctions)


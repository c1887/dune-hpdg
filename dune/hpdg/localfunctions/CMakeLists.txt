add_subdirectory(lagrange)

install(FILES
  assemblycache.hh
  qkdglegendre.hh
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/hpdg/localfunctions)


add_subdirectory(qkgausslobatto)

install(FILES
  qkgauslobatto.hh
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/hpdg/localfunctions/lagrange)

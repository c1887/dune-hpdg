add_subdirectory(qkgausslobatto)

install(FILES
  qkdynamicordercache.hh
  qkgauslobatto.hh
  qkglfactory.hh
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/hpdg/localfunctions/lagrange)

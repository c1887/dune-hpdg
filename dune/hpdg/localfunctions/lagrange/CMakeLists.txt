add_subdirectory(qkgausslobatto)

install(FILES
  qkcache.hh
  qkdynamicordercache.hh
  qkgauskronrod.hh
  qkgauslegendre.hh
  qkgauslobatto.hh
  qkglfactory.hh
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/hpdg/localfunctions/lagrange)

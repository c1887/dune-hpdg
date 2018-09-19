add_subdirectory("localassemblers")
install(FILES
  dgtocgnonconformingtransferassembler.hh
  dgtocgtransferassembler.hh
  dgtodggridtransferassembler.hh
  dgtransferoperatorassembler.hh
  gausslobattoipdgassembler.hh
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/hpdg/assemblers)

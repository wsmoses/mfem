# ejwork shim: shim-xla2 with libRaise_best4sync.so (tmp/best4 + Enzyme-JAX #3641: cudaDeviceSynchronize lowered to reactantXLASync)
if(NOT TARGET ClangEnzymeFlags)
  add_library(ClangEnzymeFlags INTERFACE IMPORTED)
  set_target_properties(ClangEnzymeFlags PROPERTIES
    INTERFACE_COMPILE_OPTIONS
      "-fplugin=/home/wmoses/ejwork/so/libClangReactant_b08e.so;SHELL:-mllvm -raising-plugin-path;SHELL:-mllvm /home/wmoses/ejwork/so/libRaise_best4sync.so;SHELL:-mllvm -reactant-backend;SHELL:-mllvm xla-gpu;SHELL:-mllvm -reactant-dataflow;SHELL:-mllvm -reactant-device-lib;SHELL:-mllvm /usr/local/cuda-12.9/nvvm/libdevice/libdevice.10.bc")
  set_target_properties(ClangEnzymeFlags PROPERTIES INTERFACE_LINK_LIBRARIES "/home/wmoses/ejwork/rt/paper18/libReactantExtra_stable.so;-Wl,-rpath,/home/wmoses/ejwork/rt/paper18")
endif()
set(Enzyme_FOUND TRUE)
set(Enzyme_LIBRARIES ClangEnzymeFlags)

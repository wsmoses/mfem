# ejwork three-way comparison shim: Reactant plugin, cuda backend, raising library of the tmp/best build
if(NOT TARGET ClangEnzymeFlags)
  add_library(ClangEnzymeFlags INTERFACE IMPORTED)
  set_target_properties(ClangEnzymeFlags PROPERTIES
    INTERFACE_COMPILE_OPTIONS
      "-fplugin=/home/wmoses/ejwork/so/libClangReactant_best.so;SHELL:-mllvm -raising-plugin-path;SHELL:-mllvm /home/wmoses/ejwork/so/libRaise_best.so;SHELL:-mllvm -reactant-backend;SHELL:-mllvm cuda;SHELL:-mllvm -reactant-dataflow;SHELL:-mllvm -reactant-device-lib;SHELL:-mllvm /usr/local/cuda-12.9/nvvm/libdevice/libdevice.10.bc")
endif()
set(Enzyme_FOUND TRUE)
set(Enzyme_LIBRARIES ClangEnzymeFlags)

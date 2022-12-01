# Copyright (c) 2022-present, Trail of Bits, Inc.

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name ConversionToCIR)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix ConversionToCIR)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix ConversionToCIR)
add_public_tablegen_target(VASTConversionToCIRPassIncGen)
add_mlir_doc(Passes ConversionToCIRPasses ./ -gen-pass-doc)

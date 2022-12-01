// Copyright (c) 2022-present, Trail of Bits, Inc.

#ifndef VAST_CONVERSION_TOCIR_PASSES_TD
#define VAST_CONVERSION_TOCIR_PASSES_TD

include "mlir/Pass/PassBase.td"

def HLToCIR : Pass<"vast-hl-to-cir", "mlir::ModuleOp"> {
  let summary = "VAST HL to CIR Dialect conversion";
  let description = [{
    Converts high level VAST operations to CIR dialect.
  }];

  let constructor = "vast::createHLToCIRPass()";
  let dependentDialects = ["vast::hl::HighLevelDialect", "mlir::cir::CIRDialect"];
}

#endif // VAST_CONVERSION_TOCIR_PASSES_TD

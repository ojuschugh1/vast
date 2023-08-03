// Copyright (c) 2022-present, Trail of Bits, Inc.

#ifndef VAST_ANALYSIS_PASSES_TD
#define VAST_ANALYSIS_PASSES_TD

include "mlir/Pass/PassBase.td"

def TaintsPropagation : Pass<"vast-taints-propagation"> {
  let summary = "Taints Propagation";
  let description = [{ TODO }];
  let constructor = "vast::dfa::createTaintsPropagationPass()";
}

#endif // VAST_ANALYSIS_PASSES_TD

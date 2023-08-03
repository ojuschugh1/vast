// Copyright (c) 2023, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>
VAST_UNRELAX_WARNINGS

namespace vast::dfa {
    std::unique_ptr< mlir::Pass > createTaintsPropagationPass();

    // Generate the code for registering passes.
    #define GEN_PASS_REGISTRATION
    #include "vast/Analysis/Passes.h.inc"

} // namespace vast::dfa

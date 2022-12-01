// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>
VAST_UNRELAX_WARNINGS

#include <clang/CIR/Dialect/IR/CIRDialect.h>

#include <vast/Dialect/HighLevel/HighLevelDialect.hpp>
#include <vast/Dialect/HighLevel/HighLevelOps.hpp>

#include <memory>

namespace vast
{
    std::unique_ptr< mlir::Pass > createHLToCIRPass();

    // Generate the code for registering passes.
    #define GEN_PASS_REGISTRATION
    #include "vast/Conversion/ToCIR/Passes.h.inc"

} // namespace vast

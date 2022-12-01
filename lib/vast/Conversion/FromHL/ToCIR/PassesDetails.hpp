// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
VAST_UNRELAX_WARNINGS

#include <clang/CIR/Dialect/IR/CIRDialect.h>

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/LowLevel/LowLevelDialect.hpp"

namespace vast
{
    // Generate the classes which represent the passes
    #define GEN_PASS_CLASSES
    #include "vast/Conversion/ToCIR/Passes.h.inc"

} // namespace vast

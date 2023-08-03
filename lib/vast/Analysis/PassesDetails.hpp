// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
VAST_UNRELAX_WARNINGS

#include "vast/Analysis/Passes.hpp"

namespace vast
{
    // Generate the classes which represent the passes
    #define GEN_PASS_CLASSES
    #include "vast/Analysis/Passes.h.inc"

} // namespace vast

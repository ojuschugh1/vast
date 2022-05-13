// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast-c/Dialect/HighLevel.h"
#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Registration.h>
#include <mlir/CAPI/Support.h>

using namespace vast;
using namespace vast::hl;

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(HL, hl, HighLevelDialect)

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Attribute API.
//===----------------------------------------------------------------------===//

// Copyright (c) 2022-present, Trail of Bits, Inc.

#ifndef VACT_C_DIALECT_HIGH_LEVEL_H
#define VACT_C_DIALECT_HIGH_LEVEL_H

#include <mlir-c/IR.h>
#include <mlir-c/Registration.h>

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(HL, hl);

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Attribute API.
//===----------------------------------------------------------------------===//


#ifdef __cplusplus
}
#endif

#endif // VACT_C_DIALECT_HIGH_LEVEL_H


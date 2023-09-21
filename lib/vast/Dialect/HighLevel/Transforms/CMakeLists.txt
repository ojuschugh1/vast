# Copyright (c) 2022-present, Trail of Bits, Inc.

add_vast_conversion_library(HighLevelTransforms
  ExportFnInfo.cpp
  HLLowerTypes.cpp
  LLVMDump.cpp
  DCE.cpp
  Desugar.cpp
  LowerTypeDefs.cpp
  SpliceTrailingScopes.cpp
)


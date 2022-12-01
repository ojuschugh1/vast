# Copyright (c) 2022-present, Trail of Bits, Inc.

add_mlir_conversion_library(VASTHLToCIR
    ToCIR.cpp

    DEPENDS
        VASTConversionToCIRPassIncGen

    LINK_LIBS PUBLIC
        MLIRHighLevel
        MLIRCIR
)

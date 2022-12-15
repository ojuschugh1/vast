// Copyright (c) 2022, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/Types.h>
#include <mlir/Transforms/DialectConversion.h>
VAST_UNRELAX_WARNINGS

#include <vast/Util/Maybe.hpp>

namespace vast {

    using optional_wrapper< mlir_type >;
    using maybe_wrapper = Maybe< mlir_type >;
    using maybe_type = llvm::Optional< mlir_type >;

    struct type_converter : mlir::TypeConverter
    {
        using base = mlir::TypeConverter;
        using base::base;

        template< typename ...conversions >
        void add_conversions(conversions && ...convs) {
            (addConversion(std::forward< conversions >(convs)), ...);
        }
    };

} // namespace vast

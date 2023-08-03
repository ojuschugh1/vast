// Copyright (c) 2023, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
VAST_UNRELAX_WARNINGS

#include <algorithm>
#include <set>

#include "vast/Dialect/Meta/MetaAttributes.hpp"
#include "vast/Dialect/Meta/MetaDialect.hpp"
#include "vast/Util/Common.hpp"

namespace vast::dfa {

    struct taints_value {
        using taints = std::set< meta::identifier_t >;

        taints_value() = default;

        taints_value(const taints &set) : value(set) {}

        taints_value(taints &&set) : value(std::move(set)) {}

        taints_value(mlir::DictionaryAttr attr) {
            if (auto id = attr.get(meta::identifier_name)) {
                value.insert(attr.cast< meta::IdentifierAttr >().getValue());
            }
        }

        void print(llvm::raw_ostream &os) const;

        static auto join(const taints_value &lhs, const taints_value &rhs) -> taints_value {
            taints result;
            std::ranges::set_union(lhs.value, rhs.value, std::inserter(result, result.begin()));
            return result;
        }

        bool operator==(const taints_value &other) const = default;

      private:
        taints value;
    };

    template< typename T >
    using lattice = mlir::dataflow::Lattice< T >;

    using dataflow_solver = mlir::DataFlowSolver;

    struct taints_lattice : lattice< taints_value > {
        using lattice< taints_value >::Lattice;

        void onUpdate(dataflow_solver *solver) const override;
    };

    template< typename lat >
    using sparse_analysis = mlir::dataflow::SparseDataFlowAnalysis< lat >;

    struct taints_analysis : sparse_analysis< taints_lattice > {
        using base = sparse_analysis< taints_lattice >;
        using base::base;

        using operands_taints = mlir::ArrayRef< const taints_lattice * >;
        using result_taints   = mlir::ArrayRef< taints_lattice * >;

        void visitOperation(operation op, operands_taints ops, result_taints res) override;

        void setToEntryState(taints_lattice *lattice) override;
    };

} // namespace vast::dfa

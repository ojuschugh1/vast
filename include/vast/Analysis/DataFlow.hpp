// Copyright (c) 2023, Trail of Bits, Inc.

#pragma once

#include <algorithm>
#include <set>

#include "vast/Dialect/Meta/MetaAttributes.hpp"
#include "vast/Dialect/Meta/MetaDialect.hpp"
#include "vast/Util/Common.hpp"

namespace vast::dfa {

    static constexpr std::string_view identifier_name = "taints";

    struct any_taint {};

    using taints_set = std::set< meta::identifier_t >;

    using taint_type = std::variant< taints_set, any_taint >;

    bool operator==(const taint_type &lhs, const taint_type &rhs) {
        if (std::holds_alternative< any_taint >(lhs) &&
            std::holds_alternative< any_taint >(rhs))
        {
            return true;
        }

        const auto *lhs_set = std::get_if< taints_set >(&lhs);
        const auto *rhs_set = std::get_if< taints_set >(&rhs);
        if (lhs_set && rhs_set) {
            return *lhs_set == *rhs_set;
        }
        return false;
    }

    struct taint_lattice_value {
        taint_lattice_value() = default;

        taint_lattice_value(any_taint value) : taints(value) {}

        taint_lattice_value(const taints_set &set) : taints(set) {}

        taint_lattice_value(taints_set &&set) : taints(std::move(set)) {}

        taint_lattice_value(mlir::DictionaryAttr attr) {
            if (auto id = attr.get(meta::identifier_name)) {
                taints = taints_set{ attr.cast< meta::IdentifierAttr >().getValue() };
            }
        }

        static taint_lattice_value top() { return { any_taint() }; }

        static taint_lattice_value bottom() { return { taints_set() }; }

        bool is_top() const { return std::holds_alternative< any_taint >(taints); }

        bool is_bottom() const {
            if (auto set = std::get_if< taints_set >(&taints)) {
                return set->empty();
            }
            return false;
        }

        static auto getPessimisticValueState(mcontext_t *) -> taint_lattice_value {
            return top();
        }

        static auto getPessimisticValueState(mlir_value value) -> taint_lattice_value {
            if (auto parent = value.getDefiningOp()) {
                if (auto attr = parent->getAttrOfType< mlir::DictionaryAttr >("taints")) {
                    return taint_lattice_value(attr);
                }
            }
            return top();
        }

        static auto join(const taint_lattice_value &lhs, const taint_lattice_value &rhs)
            -> taint_lattice_value
        {
            if (lhs.is_top() || rhs.is_top()) {
                return top();
            }

            taints_set result;
            const auto &lhs_set = std::get< taints_set >(lhs.taints);
            const auto &rhs_set = std::get< taints_set >(rhs.taints);
            std::ranges::set_union(lhs_set, rhs_set, std::inserter(result, result.begin()));
            return result;
        }

        bool operator==(const taint_lattice_value &other) const {
            return taints == other.taints;
        }

      private:
        taint_type taints = taints_set();
    };

} // namespace vast::dfa

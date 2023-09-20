// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
VAST_UNRELAX_WARNINGS

#include "vast/Conversion/Common/Passes.hpp"
#include "vast/Conversion/Common/Patterns.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/DialectConversion.hpp"
#include "vast/Util/TypeUtils.hpp"
#include "vast/Util/Functions.hpp"

#include "vast/Conversion/Common/Rewriter.hpp"

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

#include "PassesDetails.hpp"

namespace vast::hl {

    namespace pattern {
        using type_map = std::map< mlir_type, mlir_type >;

        struct type_converter
            : tc::base_type_converter
            , tc::mixins< type_converter >
        {
            vast_module mod;

            type_converter(mcontext_t &mctx, vast_module mod)
                : tc::base_type_converter(), mod(mod)
            {
                addConversion([&](mlir_type t) { return convert(t); });
                addConversion([&](mlir::SubElementTypeInterface t) { return convert(t); });
            }

            maybe_types_t do_conversion(mlir_type type) {
                types_t out;
                if (mlir::succeeded(this->convertTypes(type, out))) {
                    return { std::move(out) };
                }
                return {};
            }

            mlir_type desugar_step(mlir_type type) {
                if (auto t = mlir::dyn_cast< hl::ElaboratedType >(type))
                    return t.getElementType();
                return type;
            }

            mlir_type desugar(mlir_type type) {
                auto prev = type;
                do {
                    std::swap(prev, type);
                    type = desugar_step(prev);
                } while (prev != type);
                return type;
            }

            mlir_type convert(mlir_type type) { return desugar(type); }

            mlir_type convert(mlir::SubElementTypeInterface with_subelements) {
                return with_subelements.replaceSubElements([&] (mlir_type type) {
                    return desugar(type);
                });
            }
        };

        struct desugar : generic_conversion_pattern
        {
            using base = generic_conversion_pattern;
            using base::base;

            type_converter &tc;

            desugar(type_converter &tc, mcontext_t &mctx) : base(tc, mctx), tc(tc) {}

            template< typename attrs_list >
            maybe_attr_t high_level_typed_attr_conversion(mlir::Attribute attr) const {
                using attr_t = typename attrs_list::head;
                using rest_t = typename attrs_list::tail;

                if (auto typed = mlir::dyn_cast< attr_t >(attr)) {
                    return Maybe(typed.getType())
                        .and_then([&] (auto type) {
                            return getTypeConverter()->convertType(type);
                        })
                        .and_then([&] (auto type) {
                            return attr_t::get(type, typed.getValue());
                        })
                        .template take_wrapped< maybe_attr_t >();
                }

                if constexpr (attrs_list::size != 1) {
                    return high_level_typed_attr_conversion< rest_t >(attr);
                } else {
                    return std::nullopt;
                }
            }

            auto convert_high_level_typed_attr() const {
                return [&] (mlir::Attribute attr) {
                    return high_level_typed_attr_conversion< high_level_typed_attrs >(attr);
                };
            }

            logical_result matchAndRewrite(
                mlir::Operation *op, mlir::ArrayRef< mlir::Value > ops,
                conversion_rewriter &rewriter
            ) const override {
                // Special case for functions, it may be that we can unify it with
                // the generic one.
                if (auto fn = mlir::dyn_cast< hl::FuncOp >(op)) {
                    return rewrite(fn, ops, rewriter);
                }

                auto rtys = tc.convert_types_to_types(op->getResultTypes());
                VAST_PATTERN_CHECK(rtys, "Type conversion failed in op {0}", *op);

                auto do_change = [&]() {
                    for (std::size_t i = 0; i < rtys->size(); ++i) {
                        op->getResult(i).setType((*rtys)[i]);
                    }

                    if (op->getNumRegions() != 0) {
                        fixup_entry_block(op->getRegion(0));
                    }

                    // TODO unify with high level type conversion
                    mlir::AttrTypeReplacer replacer;
                    replacer.addReplacement(tc::convert_type_attr(tc));
                    replacer.addReplacement(convert_high_level_typed_attr());
                    replacer.recursivelyReplaceElementsIn(op, true /* replace attrs */);
                };

                rewriter.updateRootInPlace(op, do_change);

                return mlir::success();
            }

            logical_result rewrite(
                hl::FuncOp fn, mlir::ArrayRef< mlir::Value > ops,
                conversion_rewriter &rewriter
            ) const {
                auto trg = tc.convert_type_to_type(fn.getFunctionType());
                VAST_PATTERN_CHECK(trg, "Failed type conversion of, {0}", fn);

                rewriter.updateRootInPlace(fn, [&]() {
                    fn.setType(*trg);
                    if (fn->getNumRegions() != 0) {
                        fixup_entry_block(fn.getBody());
                    }
                });

                return mlir::success();
            }

            void fixup_entry_block(mlir::Region &region) const {
                if (region.empty()) {
                    return;
                }

                for (auto arg : region.front().getArguments()) {
                    auto trg = tc.convert_type_to_type(arg.getType());
                    VAST_PATTERN_CHECK(trg, "Type conversion failed: {0}", arg);
                    arg.setType(*trg);
                }
            }
        };

    } // namespace pattern

    bool has_desugared_type(operation op) {
        return has_type_somewhere< hl::ElaboratedType >(op);
    }

    struct Desugar : ModuleConversionPassMixin< Desugar, DesugarBase >
    {
        using base = ModuleConversionPassMixin< Desugar, DesugarBase >;

        static auto create_conversion_target(mcontext_t &mctx) {
            mlir::ConversionTarget trg(mctx);

            trg.markUnknownOpDynamicallyLegal([](operation op) {
                return !has_desugared_type(op);
            });

            return trg;
        }

        void runOnOperation() override
        {
            auto &mctx = getContext();
            auto target = create_conversion_target(mctx);
            vast_module op = getOperation();

            rewrite_pattern_set patterns(&mctx);

            auto tc = pattern::type_converter(mctx, op);
            patterns.template add< pattern::desugar >(tc, mctx);

            if (mlir::failed(mlir::applyPartialConversion(op, target, std::move(patterns)))) {
                return signalPassFailure();
            }
        }
    };

} // namespace vast::hl

std::unique_ptr< mlir::Pass > vast::hl::createDesugarPass() {
    return std::make_unique< vast::hl::Desugar >();
}

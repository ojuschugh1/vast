// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Conversion/ToCIR/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
VAST_UNRELAX_WARNINGS

#include <clang/CIR/Dialect/IR/CIRDialect.h>

#include "vast/Conversion/Common/Passes.hpp"
#include "vast/Conversion/Common/Patterns.hpp"
#include "vast/Conversion/Common/TypeConverter.hpp"

#include "vast/Util/Maybe.hpp"
#include "vast/Util/TypeList.hpp"
#include "vast/Util/DialectConversion.hpp"

#include "PassesDetails.hpp"

namespace vast {

    //
    // function conversion
    //
    struct func_pattern : operation_conversion_pattern< hl::FuncOp > {
        using base = operation_conversion_pattern< hl::FuncOp >;

        using base::base;
        using adaptor_t = typename hl::FuncOp::Adaptor;

        logical_result matchAndRewrite(hl::FuncOp op, adaptor_t adaptor, conversion_rewriter &rewriter) const override {
            // TODO deal with attributes and type conversions
            // TODO convert or reuse cir linkage in FuncOp
            // TODO adapt clang codegen
            auto linkage = mlir::cir::GlobalLinkageKind::ExternalLinkage;
            auto fn = rewriter.create< mlir::cir::FuncOp >(
                op.getLoc(), op.getName(), op.getFunctionType(), linkage
            );

            rewriter.inlineRegionBefore(op.getBody(), fn.getBody(), fn.end());
            rewriter.eraseOp(op);

            return logical_result::success();
        }

        static void legalize(conversion_target &target) {
            target.addLegalOp< mlir::cir::FuncOp >();
            target.addIllegalOp< hl::FuncOp >();
        }
    };

    using func_conversions = util::type_list< func_pattern >;

    //
    // var conversion
    //

    template< typename source >
    struct variable_conversion_pattern : operation_conversion_pattern< source > {

        using base = operation_conversion_pattern< source >;

        using base::base;
        using adaptor_t = typename source::Adaptor;

        // mlir_type to_pointer_type(mlir_type type) {
        //     return
        // }

        logical_result matchAndRewrite(source op, adaptor_t adaptor, conversion_rewriter &rewriter) const override {
            auto var_type = op.getType();
            auto local_var_ptr_type = op.getType();

            // TODO set alignment
            auto align = base::i64(0);

            rewriter.replaceOpWithNewOp< mlir::cir::AllocaOp >(op,
                local_var_ptr_type, var_type, op.getName(), align
            );

            // TODO deal with initializer
            return logical_result::success();
        }
    };

    using var_conversions = util::type_list<
        variable_conversion_pattern< hl::VarDeclOp >
    >;

    //
    // cast operations
    //
    // template< typename source, mlir::cir::CastKind kind >
    // struct cast_pattern : operation_conversion_pattern< source > {
    //     using base = operation_conversion_pattern< source >;

    //     using base::base;
    //     using adaptor_t = typename source::Adaptor;

    //     logical_result matchAndRewrite(source op, adaptor_t adaptor, conversion_rewriter &rewriter) const override {
    //         rewriter.replaceOpWithNewOp< mlir::cir::CastOp >(op,
    //             op.getType(), kind, adaptor.getLhs(), adaptor.getRhs()
    //         );
    //         return logical_result::success();
    //     }

    //     static void legalize(conversion_target &target) {
    //         target.addLegalOp< mlir::cir::CastOp >();
    //         // target.addIllegalOp<
    //         //     hl::AddIOp, hl::AddFOp,
    //         //     hl::SubIOp, hl::SubFOp,
    //         //     hl::MulIOp, hl::MulFOp,
    //         //     hl::DivSOp, hl::DivUOp, hl::DivFOp,
    //         //     hl::RemSOp, hl::RemUOp,  hl::RemUOp,
    //         //     hl::BinXorOp, hl::BinOrOp, hl::BinAndOp,
    //         //     hl::BinShlOp, hl::BinShrOp
    //         // >();
    //     }
    // };

    // using cast_conversions = util::type_list<
    //     cast_pattern< hl::ImplicitCastOp, mlir::cir::CastOp >
    // >;

    //
    // binary operations
    //
    template< typename source, mlir::cir::BinOpKind kind >
    struct binary_pattern : operation_conversion_pattern< source > {
        using base = operation_conversion_pattern< source >;

        using base::base;
        using adaptor_t = typename source::Adaptor;

        logical_result matchAndRewrite(source op, adaptor_t adaptor, conversion_rewriter &rewriter) const override {
            rewriter.replaceOpWithNewOp< mlir::cir::BinOp >(op,
                op.getType(), kind, adaptor.getLhs(), adaptor.getRhs()
            );
            return logical_result::success();
        }

        static void legalize(conversion_target &target) {
            target.addLegalOp< mlir::cir::BinOp >();
            target.addIllegalOp<
                hl::AddIOp, hl::AddFOp,
                hl::SubIOp, hl::SubFOp,
                hl::MulIOp, hl::MulFOp,
                hl::DivSOp, hl::DivUOp, hl::DivFOp,
                hl::RemSOp, hl::RemUOp,  hl::RemUOp,
                hl::BinXorOp, hl::BinOrOp, hl::BinAndOp,
                hl::BinShlOp, hl::BinShrOp
            >();
        }
    };

    using arithmetic_conversions = util::type_list<
        binary_pattern< hl::AddIOp, mlir::cir::BinOpKind::Add >,
        binary_pattern< hl::AddFOp, mlir::cir::BinOpKind::Add >,
        binary_pattern< hl::SubIOp, mlir::cir::BinOpKind::Sub >,
        binary_pattern< hl::SubFOp, mlir::cir::BinOpKind::Sub >,
        binary_pattern< hl::MulIOp, mlir::cir::BinOpKind::Mul >,
        binary_pattern< hl::MulFOp, mlir::cir::BinOpKind::Mul >,
        binary_pattern< hl::DivSOp, mlir::cir::BinOpKind::Div >,
        binary_pattern< hl::DivUOp, mlir::cir::BinOpKind::Div >,
        binary_pattern< hl::DivFOp, mlir::cir::BinOpKind::Div >,
        binary_pattern< hl::RemSOp, mlir::cir::BinOpKind::Rem >,
        binary_pattern< hl::RemUOp, mlir::cir::BinOpKind::Rem >,
        binary_pattern< hl::RemUOp, mlir::cir::BinOpKind::Rem >
    >;

    using binary_conversions = util::type_list<
        binary_pattern< hl::BinXorOp, mlir::cir::BinOpKind::Xor >,
        binary_pattern< hl::BinOrOp,  mlir::cir::BinOpKind::Or  >,
        binary_pattern< hl::BinAndOp, mlir::cir::BinOpKind::And >
    >;

    using shift_conversions = util::type_list<
        binary_pattern< hl::BinShlOp, mlir::cir::BinOpKind::Shl >,
        binary_pattern< hl::BinShrOp, mlir::cir::BinOpKind::Shr >
    >;

    struct HLToCIRPass : ModuleConversionPassMixin< HLToCIRPass, HLToCIRBase > {

        using base = ModuleConversionPassMixin< HLToCIRPass, HLToCIRBase >;

        using base::getContext;
        using base::getOperation;
        using base::getAnalysis;

        //
        // conversion target
        //
        static conversion_target create_conversion_target(MContext &context) {
            conversion_target target(context);
            target.addLegalDialect< mlir::cir::CIRDialect >();
            return target;
        }

        //
        // type conversions
        //

        static inline auto keep_if_conversion = [] (auto pred) -> optional_type {
            return [] (auto ty) {
                // TODO why duplicit?
                return maybe_type(ty).keep_if(pred).take_wrapped< optional_type >();
            };
        };

        static inline auto keep_non_hl_type = keep_if_conversion(
            [] (auto ty) { return !isHighLevelType(t); }
        );

        static inline auto lvalue_type_conversion = [] (hl::LValueType ty) -> maybe_type {
            llvm::errs() << "convert type\n";
            return maybe_type();
        };

        type_converter make_type_converter() {
            type_converter tc;
            tc.add_conversions(
                keep_non_hl_type,
                lvalue_type_conversion
            );
            return tc;
        }

        //
        // conversion setup
        //
        void populate_conversions(rewrite_pattern_set &patterns) {
            type_converter tc = make_type_converter();

            base::populate_conversions<
                /* function conversions */
                func_conversions,
                /* decl operations */
                var_conversions,
                /* cast operations */
                /* cast_conversions, */
                /* binary conversions */
                arithmetic_conversions,
                binary_conversions,
                shift_conversions
            >(patterns, tc);
        }
    };

    std::unique_ptr< mlir::Pass > createHLToCIRPass() {
        return std::make_unique< HLToCIRPass >();
    }

} // namespace vast

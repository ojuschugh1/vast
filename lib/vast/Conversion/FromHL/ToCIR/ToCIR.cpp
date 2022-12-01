#include "vast/Conversion/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
VAST_UNRELAX_WARNINGS

#include <clang/CIR/Dialect/IR/CIRDialect.h>

#include "vast/Conversion/Common/Passes.hpp"
#include "vast/Conversion/Common/Patterns.hpp"

#include "vast/Util/DialectConversion.hpp"

#include "../../PassesDetails.hpp"

namespace vast {

    template< typename source, mlir::cir::BinOpKind kind >
    struct arith_pattern : operation_rewrite_pattern< source > {
        using base = operation_rewrite_pattern< source >;

        using base::base;

        logical_result matchAndRewrite(source op, adaptor_t adaptor, conversion_rewriter &rewriter) const override {
            rewriter.replaceOpWithNewOp< mlir::cir::BinOp >(op,
                op.getType(), kind, adaptor.getLhs(), adaptor.getRhs()
            );
            return logical_result::success();
        }

        static void legalize(conversion_target &target) {
            target.addLegalOp< mlir::cir::BinOp >();
        }
    };

    //
    // binary operations
    //

    using arithmetic_conversions = util::type_list<
        arith_pattern< hl::AddIOp, mlir::cir::BinOpKind::Add >,
        arith_pattern< hl::AddFOp, mlir::cir::BinOpKind::Add >,
        arith_pattern< hl::SubIOp, mlir::cir::BinOpKind::Sub >,
        arith_pattern< hl::SubFOp, mlir::cir::BinOpKind::Sub >,
        arith_pattern< hl::MulIOp, mlir::cir::BinOpKind::Mul >,
        arith_pattern< hl::MulFOp, mlir::cir::BinOpKind::Mul >,
        arith_pattern< hl::DivSOp, mlir::cir::BinOpKind::Div >,
        arith_pattern< hl::DivUOp, mlir::cir::BinOpKind::Div >,
        arith_pattern< hl::DivFOp, mlir::cir::BinOpKind::Div >,
        arith_pattern< hl::RemSOp, mlir::cir::BinOpKind::Rem >,
        arith_pattern< hl::RemUOp, mlir::cir::BinOpKind::Rem >,
        arith_pattern< hl::RemUOp, mlir::cir::BinOpKind::Rem >
    >;

    using binary_conversions = util::type_list<
        arith_pattern< hl::BinXorOp, mlir::cir::BinOpKind::Xor >,
        arith_pattern< hl::BinOrOp,  mlir::cir::BinOpKind::Or  >,
        arith_pattern< hl::BinAndOp, mlir::cir::BinOpKind::And >
    >;

    using shift_conversions = util::type_list<
        arith_pattern< hl::BinShlOp, mlir::cir::BinOpKind::Shl >,
        arith_pattern< hl::BinShrOp, mlir::cir::BinOpKind::Shr >
    >;

    struct HLToCIRPass : ModuleConversionPassMixin< HLToCIRPass, HLToCIRBase > {

        using base = ModuleConversionPassMixin< HLToCIRPass, HLToCIRBase >;

        static conversion_target create_conversion_target(MContext &context) {
            conversion_target target(context);
            target.addLegalDialect< mlir::cir::CIRDialect >();
            return target;
        }

        static void populate_conversions(rewrite_pattern_set &patterns) {
            base::populate_conversions<
                arithmetic_conversions,
                binary_conversions,
                shift_conversions
            >(patterns);
        }
    };

    std::unique_ptr< mlir::Pass > createHLToCIRPass() {
        return std::make_unique< HLToCIRPass >();
    }

} // namespace vast

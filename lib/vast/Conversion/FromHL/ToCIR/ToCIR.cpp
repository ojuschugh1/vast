#include "vast/Conversion/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/DialectConversion.hpp"
#include "../../PassesDetails.hpp"

namespace vast {

    struct HLToCIRPass : HLToCIRBase< HLToCIRPass > {
        void runOnOperation() override {

        }
    };

    std::unique_ptr< mlir::Pass > createHLToCIRPass() {
        return std::make_unique< HLToCIRPass >();
    }

} // namespace vast

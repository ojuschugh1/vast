#include "vast/Analysis/DataFlow.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Analysis/DataFlowFramework.h>
VAST_UNRELAX_WARNINGS

#include "PassesDetails.hpp"
#include "vast/Analysis/Passes.hpp"

namespace vast::dfa {

    void taints_value::print(llvm::raw_ostream &os) const {
        if (value.empty()) {
            os << "<UNINITIALIZED>";
            return;
        }

        os << "taint value";
    }

    void taints_lattice::onUpdate(dataflow_solver *solver) const {}

    void taints_analysis::visitOperation(
        operation op, operands_taints operands, result_taints results
    ) {
        llvm::dbgs() << "Visiting operation: " << *op << "\n";
    }

    void taints_analysis::setToEntryState(taints_lattice *lattice) {
        propagateIfChanged(lattice, lattice->join(taints_value()));
    }

    struct TaintsPropagation : public TaintsPropagationBase< TaintsPropagation > {
        void runOnOperation() override {
            auto op = getOperation();

            dataflow_solver solver;
            solver.load< taints_analysis >();
            if (failed(solver.initializeAndRun(op))) {
                return signalPassFailure();
            }
            // rewrite(solver, op->getContext(), op->getRegions());
        }
    };

    std::unique_ptr< mlir::Pass > createTaintsPropagationPass() {
        return std::make_unique< TaintsPropagation >();
    }

} // namespace vast::dfa

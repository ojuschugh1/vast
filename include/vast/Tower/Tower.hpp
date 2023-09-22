// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <llvm/IR/Module.h>

#include <mlir/Transforms/LocationSnapshot.h>
VAST_UNRELAX_WARNINGS

#include <filesystem>

namespace vast::tw {

    struct tower {
        tower(mcontext_t &mctx) : mctx(mctx) {}

        //
        // named snapshots in the tower
        //
        llvm::StringMap< vast_module > snaps;

        using module_storage_t = std::vector< owning_module_ref >;

        mcontext_t &mctx;
        module_storage_t layers;

        std::filesystem::path dir;

        //
        // For LLVM Layer
        //
        llvm::LLVMContext llvm_context;
        std::unique_ptr< llvm::Module > llvm;

        auto last_module() -> vast_module { return layers.back().get(); }

        void make_snapshot(string_ref name) {
            mlir::OpPrintingFlags flags;
            flags.enableDebugInfo();

            auto path = (dir / name.str()).replace_extension(".mlir");
            mlir::PassManager pm(&mctx);
            pm.addPass(mlir::createLocationSnapshotPass(flags, path.string(), name));
            if (mlir::failed(pm.run(last_module()))) {
                VAST_UNREACHABLE("error: snapshot pass failed");
            }

            snaps[name] = last_module();
        }

        void foundation(owning_module_ref mod) {
            layers.push_back(std::move(mod));
            make_snapshot("source");
        }

        void raise(string_ref pass_name, string_ref tag) {
            mlir::PassManager pm(&mctx);
            if (mlir::failed(mlir::parsePassPipeline(pass_name, pm))) {
                VAST_UNREACHABLE("error: failed to parse pipeline");
            }

            if (mlir::failed(pm.run(last_module()))) {
                VAST_UNREACHABLE("error: pass {} failed", pass_name);
            }

            make_snapshot(tag);
        }

        void raise(string_ref pass_name) {
            raise(pass_name, pass_name);
        }

        bool has_snapshot(string_ref name) const {
            return snaps.count(name);
        }

        void initialize(std::filesystem::path dst) {
            dir = dst;
            if (std::filesystem::exists(dir)) {
                std::filesystem::remove_all(dir);
            }
            std::filesystem::create_directory(dir);

        }
    };

} // namespace vast::tw

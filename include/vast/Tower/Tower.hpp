// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Pass/PassManager.h>
#include <llvm/IR/Module.h>
VAST_UNRELAX_WARNINGS

namespace vast::tw {

    using llvm_module = llvm::Module *;

    struct default_loc_rewriter_t
    {
        static auto insert(mlir::Operation *op) -> void;
        static auto remove(mlir::Operation *op) -> void;
        static auto prev(mlir::Operation *op) -> mlir::Operation *;
    };

    using pass_ptr_t = std::unique_ptr< mlir::Pass >;

    template< typename loc_rewriter_t >
    struct tower
    {
        using loc_rewriter = loc_rewriter_t;
        using module_storage_t = std::vector< owning_module_ref >;

        mcontext_t *ctx;
        module_storage_t mods;
        std::optional< llvm_module > llvm;

        struct handle_t
        {
            std::size_t id;
            vast_module mod;
        };

        static auto get(mcontext_t &ctx, owning_module_ref mod) {
            tower t{ .ctx = &ctx };
            t.mods.push_back(std::move(mod));
            return t;
        }

        auto apply(handle_t handle, mlir::PassManager &pm) -> handle_t {
            handle.mod.walk(loc_rewriter::insert);

            mods.emplace_back(mlir::cast< vast_module >(handle.mod->clone()));

            auto id  = mods.size() - 1;
            auto mod = mods.back().get();

            if (mlir::failed(pm.run(mod))) {
                VAST_UNREACHABLE("error: some pass in apply() failed");
            }

            handle.mod.walk(loc_rewriter::remove);

            return { id, mod };
        }

        auto apply(handle_t handle, pass_ptr_t pass) -> handle_t {
            mlir::PassManager pm(ctx);
            pm.addPass(std::move(pass));
            return apply(handle, pm);
        }

        auto top() -> handle_t { return { mods.size(), mods.back().get() }; }

        auto last_module() -> vast_module { return mods.back().get(); }
    };

    using default_tower = tower< default_loc_rewriter_t >;

} // namespace vast::tw

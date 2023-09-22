// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Tower/Tower.hpp"
#include "vast/repl/common.hpp"
#include "vast/repl/command_base.hpp"
#include "vast/repl/pipeline.hpp"

#include <llvm/ADT/StringMap.h>

namespace vast::repl {

    struct state_t {
        explicit state_t(mcontext_t &ctx) : ctx(ctx), tower(ctx) {}

        //
        // perform exit in next step
        //
        bool exit = false;

        //
        // c/c++ source file to compile
        //
        std::optional< std::string > source;

        //
        // mlir module and context
        //
        mcontext_t &ctx;
        tw::default_tower tower;

        //
        // named snapshots into tower
        //
        llvm::StringMap< tw::default_tower::handle_t > snaps;

        //
        // sticked commands performed after each step
        //
        std::vector< command_ptr > sticked;

        //
        // named pipelines
        //
        llvm::StringMap< pipeline > pipelines;

        //
        // verbosity flags
        //
        bool verbose_pipeline = true;

        llvm::LLVMContext llvm_context;
    };

} // namespace vast::repl

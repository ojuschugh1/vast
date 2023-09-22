// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Tower/Tower.hpp"
#include "vast/repl/common.hpp"
#include "vast/repl/command_base.hpp"
#include "vast/repl/pipeline.hpp"

#undef CR1
#undef CR2

VAST_RELAX_WARNINGS
#include <llvm/IR/Module.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/LLVMContext.h>

#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/Analysis/DependenceAnalysis.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/AssumptionCache.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Transforms/Utils/ScalarEvolutionExpander.h>

#include <llvm/ADT/StringMap.h>
VAST_UNRELAX_WARNINGS

#include <filesystem>

namespace vast::repl {

    //
    // LLVM Analysis Info
    //
    struct function_analysis_info {
        explicit function_analysis_info(llvm::Function *fn)
            : dt(*fn)
            , tlii()
            , tli(tlii)
            , ac(*fn)
            , aa(tli)
            , li(dt)
            , se(*fn, tli, ac, dt, li)
            , di(fn, &aa, &se, &li)
        {}

        llvm::DominatorTree dt;
        llvm::TargetLibraryInfoImpl tlii;
        llvm::TargetLibraryInfo tli;
        llvm::AssumptionCache ac;
        llvm::AliasAnalysis aa;
        llvm::LoopInfo li;
        llvm::ScalarEvolution se;
        llvm::DependenceInfo di;
    };

    struct state_t {
        explicit state_t(mcontext_t &ctx) : ctx(ctx), tower(ctx) {}

        //
        // perform exit in next step
        //
        bool exit = false;

        //
        // c/c++ source file to compile
        //
        std::filesystem::path path;

        //
        // mlir module and context
        //
        mcontext_t &ctx;
        tw::tower tower;


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

        //
        // LLVM
        //
        llvm::LLVMContext llvm_context;
        std::map< llvm::Function*, function_analysis_info > function_info;
    };

} // namespace vast::repl

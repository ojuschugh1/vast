// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/repl/command.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/Target/LLVMIR/LLVMTranslationInterface.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/ModuleTranslation.h>

#include <llvm/IR/Module.h>
#include <llvm/IR/LLVMContext.h>
VAST_UNRELAX_WARNINGS

#include "vast/Conversion/Passes.hpp"
#include "vast/Tower/Tower.hpp"
#include "vast/repl/common.hpp"
#include <optional>

namespace vast::repl {
namespace cmd {

    void check_source(const state_t &state) {
        if (!state.source.has_value()) {
            VAST_UNREACHABLE("error: missing source");
        }
    }

    const std::string &get_source(const state_t &state) {
        check_source(state);
        return state.source.value();
    }

    void check_and_emit_module(state_t &state) {
        if (!state.snaps.count("source")) {
            state.tower.mods.push_back(
                codegen::emit_module(get_source(state), &state.ctx)
            );
            state.snaps["source"] = state.tower.top();
        }
    }

    //
    // exit command
    //
    void exit::run(state_t &state) const {
        state.exit = true;
    }

    //
    // help command
    //
    void help::run(state_t&) const {
        VAST_UNIMPLEMENTED;
    };

    //
    // load command
    //
    void load::run(state_t &state) const {
        auto source  = get_param< source_param >(params);
        state.source = codegen::get_source(source.path);
    };

    //
    // show command
    //
    void show_source(const state_t &state) {
        llvm::outs() << get_source(state) << "\n";
    }

    void show_ast(const state_t &state) {
        auto unit = codegen::ast_from_source(get_source(state));
        unit->getASTContext().getTranslationUnitDecl()->dump(llvm::outs());
        llvm::outs() << "\n";
    }

    void show_module(state_t &state) {
        check_and_emit_module(state);
        llvm::outs() << state.tower.last_module() << "\n";
    }

    void show_symbols(state_t &state) {
        check_and_emit_module(state);

        util::symbols(state.tower.last_module(), [&] (auto symbol) {
            llvm::outs() << util::show_symbol_value(symbol) << "\n";
        });
    }

    void show_snaps(state_t &state) {
        if (state.snaps.empty()) {
            llvm::outs() << "error: no tower snapshots\n";
        }

        for (const auto &[name, _] : state.snaps) {
            llvm::outs() << name << "\n";
        }
    }

    void show_pipelines(state_t &state) {
        if (state.pipelines.empty()) {
            llvm::outs() << "no pipelines\n";
        }

        for (const auto &[name, _] : state.pipelines) {
            llvm::outs() << name << "\n";
        }
    }

    void show_llvm(state_t &state) {
        if (!state.tower.llvm) {
            llvm::outs() << "no llvm module\n";
        }
        llvm::outs() << *(state.tower.llvm);
    }

    void show::run(state_t &state) const {
        auto what = get_param< kind_param >(params);
        switch (what) {
            case show_kind::source:  return show_source(state);
            case show_kind::ast:     return show_ast(state);
            case show_kind::module:  return show_module(state);
            case show_kind::symbols: return show_symbols(state);
            case show_kind::snaps:   return show_snaps(state);
            case show_kind::pipelines: return show_pipelines(state);
            case show_kind::llvm: return show_llvm(state);
        }
    };

    //
    // meta command
    //
    void meta::add(state_t &state) const {
        using ::vast::meta::add_identifier;

        auto name_param = get_param< symbol_param >(params);
        util::symbols(state.tower.last_module(), [&] (auto symbol) {
            if (util::symbol_name(symbol) == name_param.value) {
                auto id = get_param< identifier_param >(params);
                add_identifier(symbol, id.value);
                llvm::outs() << symbol << "\n";
            }
        });
    }

    void meta::get(state_t &state) const {
        using ::vast::meta::get_with_identifier;
        auto id = get_param< identifier_param >(params);
        for (auto op : get_with_identifier(state.tower.last_module(), id.value)) {
            llvm::outs() << *op << "\n";
        }
    }

    void meta::run(state_t &state) const {
        check_and_emit_module(state);

        auto action  = get_param< action_param >(params);
        switch (action) {
            case meta_action::add: add(state); break;
            case meta_action::get: get(state); break;
        }
    }

    //
    // raise command
    //
    void emit_llvm(state_t &state) {
        auto &tower = state.tower;
        auto op = tower.last_module();
        // If the old data layout with high level types is left in the module,
        // some parsing functionality inside the `mlir::translateModuleToLLVMIR`
        // will fail and no conversion translation happens, even in case these
        // entries are not used at all.
        // auto old_dl = op->getAttr(mlir::DLTIDialect::kDataLayoutAttrName);
        op->setAttr(
            mlir::DLTIDialect::kDataLayoutAttrName, mlir::DataLayoutSpecAttr::get(&state.ctx, {})
        );

        tower.llvm = mlir::translateModuleToLLVMIR(op, tower.llvm_context);

        // Restore the data layout in case this module is getting re-used later.
        // op->setAttr(mlir::DLTIDialect::kDataLayoutAttrName, old_dl);

        // mlir::ExecutionEngine::setupTargetTriple(tower.llvm.get());
    }


    void run_passes(state_t &state, std::ranges::range auto passes) {
        check_and_emit_module(state);
        mlir::PassManager pm(&state.ctx);
        auto last = state.tower.top();
        for (const auto &pass : passes) {
            if (state.verbose_pipeline) {
                llvm::errs() << "[vast] running:  " << pass << "\n";
            }

            if (pass == "emit-llvm") {
                emit_llvm(state);
                continue;
            }

            std::string pass_name = llvm::Twine("vast-" + pass).str();
            if (mlir::failed(mlir::parsePassPipeline(pass_name, pm))) {
                return;
            }
            last = state.tower.apply(last, pm);

            if (state.verbose_pipeline) {
                llvm::errs() << "[vast] snapshot: " << pass << "\n";
            }
            state.snaps[pass] = last;
        }
    }

    void raise::run(state_t &state) const {
        check_and_emit_module(state);

        std::string pipeline = get_param< pipeline_param >(params).value;
        llvm::SmallVector< llvm::StringRef, 2 > passes;
        llvm::StringRef(pipeline).split(passes, ',');

        run_passes(state, passes);
    }

    //
    // sticky command
    //
    void sticky::run(state_t &state) const {
        auto cmd = get_param< command_param >(params);
        add_sticky_command(cmd.value, state);
    }

    void add_sticky_command(string_ref cmd, state_t &state) {
        auto tokens = parse_tokens(cmd);
        state.sticked.push_back(parse_command(tokens));
    }

    //
    // snap command
    //
    void snap::run(state_t &state) const {
        auto name = get_param< name_param >(params);
        state.snaps[name.value] = state.tower.top();
    }

    //
    // make command
    //
    void make::run(state_t &state) const {
        auto name = get_param< pipeline_param >(params);

        if (!state.pipelines.count(name.value)) {
            llvm::errs() << "error: unknown pipeline " << name.value << "\n";
            return;
        }

        auto pl = state.pipelines[name.value];
        run_passes(state, pl.passes);
    }

} // namespace cmd

    command_ptr parse_command(std::span< command_token > tokens) {
        return match< cmd::command_list >(tokens);
    }

} // namespace vast::repl

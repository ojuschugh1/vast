// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/repl/command.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/Target/LLVMIR/LLVMTranslationInterface.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/ModuleTranslation.h>

#include <llvm/IR/Module.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/LLVMContext.h>
VAST_UNRELAX_WARNINGS

#include "vast/Conversion/Passes.hpp"
#include "vast/Tower/Tower.hpp"
#include "vast/repl/common.hpp"
#include <optional>

namespace vast::repl {
namespace cmd {

    void check_and_emit_module(state_t &state) {
        if (!state.tower.has_snapshot("source")) {
            state.tower.foundation(
                codegen::emit_module(state.path, &state.ctx)
            );
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
        state.path = source.path;
        state.tower.initialize(
            source.path.replace_extension("").string() + ".tower"
        );
    };

    //
    // show command
    //
    void show_source(const state_t &state) {
        llvm::outs() << codegen::get_source(state.path) << "\n";
    }

    void show_ast(const state_t &state) {
        auto unit = codegen::ast_from_source(state.path);
        unit->getASTContext().getTranslationUnitDecl()->dump(llvm::outs());
        llvm::outs() << "\n";
    }

    void show_module(state_t &state, bool locations) {
        check_and_emit_module(state);
        mlir::OpPrintingFlags flags;
        if (locations) {
            flags.enableDebugInfo();
        }
        state.tower.last_module().print(llvm::outs(), flags);
    }

    void show_symbols(state_t &state) {
        check_and_emit_module(state);

        util::symbols(state.tower.last_module(), [&] (auto symbol) {
            llvm::outs() << util::show_symbol_value(symbol) << "\n";
        });
    }

    void show_snapshots(state_t &state) {
        if (state.tower.snaps.empty()) {
            llvm::outs() << "error: no tower snapshots\n";
        }

        for (const auto &[name, _] : state.tower.snaps) {
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
            return;
        }
        llvm::outs() << *(state.tower.llvm);
    }

    void show::run(state_t &state) const {
        auto what = get_param< kind_param >(params);
        auto locs = get_param< locations_param >(params);
        switch (what) {
            case show_kind::source:  return show_source(state);
            case show_kind::ast:     return show_ast(state);
            case show_kind::module:  return show_module(state, locs.set);
            case show_kind::symbols: return show_symbols(state);
            case show_kind::snapshots:   return show_snapshots(state);
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
        auto op = tower.last_module().clone();
        op->setAttr(
            mlir::DLTIDialect::kDataLayoutAttrName, mlir::DataLayoutSpecAttr::get(&state.ctx, {})
        );

        tower.llvm = mlir::translateModuleToLLVMIR(op, tower.llvm_context);
    }

    void run_passes(state_t &state, std::ranges::range auto passes) {
        check_and_emit_module(state);
        for (const auto &pass : passes) {
            if (state.verbose_pipeline) {
                llvm::errs() << "[vast] running:  " << pass << "\n";
            }

            if (pass == "emit-llvm") {
                emit_llvm(state);
                continue;
            }

            std::string pass_name = llvm::Twine("vast-" + pass).str();
            state.tower.raise(pass_name, pass);

            if (state.verbose_pipeline) {
                llvm::errs() << "[vast] snapshot: " << pass << "\n";
            }
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

    void analyze_llvm(state_t &state) {
        if (!state.tower.llvm) {
            llvm::outs() << "no llvm module\n";
            return;
        }

        for (auto &fn : *state.tower.llvm) {
            if (fn.empty()) {
                continue;
            }

            llvm::errs() << "[vast] analyzing " << fn.getName() << "...\n";
            state.function_info.emplace(std::piecewise_construct,
              std::forward_as_tuple(&fn),
              std::forward_as_tuple(function_analysis_info(&fn))
            );

            llvm::errs() << "[vast] built dominator tree\n";
            llvm::errs() << "[vast] built target library info\n";
            llvm::errs() << "[vast] built assumption cache\n";
            llvm::errs() << "[vast] built alias analysis\n";
            llvm::errs() << "[vast] built loop info\n";
            llvm::errs() << "[vast] built scalar evolution\n";
            llvm::errs() << "[vast] built dependence info\n";
        }
    }

    void analyze::run(state_t &state) const {
        auto name = get_param< target_param >(params);
        if (name.value == "llvm") {
            analyze_llvm(state);
        } else {
            llvm::errs() << "unknown analysis target: " << name.value << "\n";
        }
    }

    void inspect::run(state_t &state) const {
        auto layer_name = get_param< layer_param >(params);
        auto location = get_param< location_param >(params);

        // layer.mod.walk([] (operation op) {
        //     llvm::errs() << op->getLoc() << "\n";
        // });
    }

} // namespace cmd

    command_ptr parse_command(std::span< command_token > tokens) {
        return match< cmd::command_list >(tokens);
    }

} // namespace vast::repl

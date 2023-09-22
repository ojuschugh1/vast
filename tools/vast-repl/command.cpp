// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/repl/command.hpp"

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
            const auto &source = get_source(state);
            auto mod           = codegen::emit_module(source, &state.ctx);
            state.tower        = tw::default_tower::get(state.ctx, std::move(mod));
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

    void show::run(state_t &state) const {
        auto what = get_param< kind_param >(params);
        switch (what) {
            case show_kind::source:  return show_source(state);
            case show_kind::ast:     return show_ast(state);
            case show_kind::module:  return show_module(state);
            case show_kind::symbols: return show_symbols(state);
            case show_kind::snaps:   return show_snaps(state);
            case show_kind::pipelines: return show_pipelines(state);
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

    void run_passes(state_t &state, std::ranges::range auto passes) {
        check_and_emit_module(state);
        mlir::PassManager pm(&state.ctx);
        auto last = state.tower.top();
        for (const auto &pass : passes) {
            if (state.verbose_pipeline) {
                llvm::errs() << "[vast] running:  " << pass << "\n";
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

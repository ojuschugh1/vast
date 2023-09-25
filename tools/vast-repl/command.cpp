// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/repl/command.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/Target/LLVMIR/LLVMTranslationInterface.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

#include <llvm/IR/Module.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/InstIterator.h>
VAST_UNRELAX_WARNINGS

#include "vast/Conversion/Passes.hpp"
#include "vast/Tower/Tower.hpp"
#include "vast/repl/common.hpp"

#include <vast/Dialect/HighLevel/HighLevelUtils.hpp>

#include <optional>
#include <unordered_map>
#include <unordered_set>

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

    namespace LLVM = mlir::LLVM;

    struct get_mapping
    {
        // func to mapping of its things
        std::map< std::string, tw::tower::mlir_to_llvm > function_mapping;
        std::map< operation, llvm::GlobalVariable * > global_vars;
        std::map< operation, llvm::Function * > global_functions;

        inline static const std::unordered_set< std::string > m_skip = { "llvm.mlir.constant" };

        inline static const std::unordered_set< std::string > allowed_miss = {
            "llvm.bitcast", "llvm.mlir.addressof"
        };

        inline static const std::unordered_set< std::string > l_skip = {};

        inline static const std::unordered_map< std::string, std::string > translations = {
            {        "llvm.alloca",        "alloca"},
            {       "llvm.bitcast",       "bitcast"},
            {         "llvm.store",         "store"},
            {          "llvm.load",          "load"},
            {        "llvm.return",           "ret"},
            {          "llvm.call",          "call"},
            { "llvm.getelementptr", "getelementptr"},

            {"llvm.mlir.addressof",              ""}
        };

        std::tuple< mlir::LLVM::LLVMFuncOp, llvm::Function * >
        get_fn(auto name, auto m_mod, auto l_mod) {
            auto l_fn = l_mod->getFunction(name);
            for (auto m_fn : hl::top_level_ops< mlir::LLVM::LLVMFuncOp >(m_mod)) {
                if (m_fn.getName() == name) {
                    return { m_fn, l_fn };
                }
            }

            return {};
        }

        auto key(mlir::Block::iterator it) -> std::string {
            return it->getName().getStringRef().str();
        }

        auto key(llvm::BasicBlock::iterator it) -> std::string { return it->getOpcodeName(); }

        bool skip(mlir::Block::iterator it) { return m_skip.count(key(it)); }

        bool skip(llvm::BasicBlock::iterator it) { return l_skip.count(key(it)); }

        bool match(auto m_it, auto l_it) {
            auto m = translations.find(key(m_it));
            VAST_CHECK(
                m != translations.end(), "Missing translation {0}: {1}", key(m_it), *m_it
            );
            return m->second == key(l_it);
        }

        auto annotate_functions(mlir::ModuleOp op, llvm::Module *l_mod) {
            auto [m_func, l_func] = get_fn("main", op, l_mod);
            VAST_ASSERT(m_func && l_func);
            global_functions.emplace(m_func, l_func);

            auto &current = function_mapping["main"];

            auto m_it                        = m_func.getRegion().begin()->begin();
            auto m_end                       = m_func.getRegion().begin()->end();
            llvm::BasicBlock::iterator l_it  = l_func->begin()->begin();
            llvm::BasicBlock::iterator l_end = l_func->begin()->end();

            while (m_it != m_end) {
                if (skip(m_it)) {
                    ++m_it;
                    continue;
                }
                VAST_ASSERT(l_it != l_end);


                if (!match(m_it, l_it)) {
                    if (skip(l_it)) {
                        ++l_it;
                        continue;
                    }
                    if (allowed_miss.count(key(m_it))) {
                        ++m_it;
                        continue;
                    }
                    VAST_CHECK(false, "Cannot progress on {0}!", key(m_it));
                }

                current.emplace(&*m_it, &*l_it);
                ++l_it;
                ++m_it;
            }
        }

        auto annotate_gvs(mlir::ModuleOp m_mod, llvm::Module *l_mod) {
            for (auto m_var : hl::top_level_ops< mlir::LLVM::GlobalOp >(m_mod)) {
                auto l_var = l_mod->getGlobalVariable(m_var.getName());
                global_vars.emplace(m_var, l_var);
            }
        }

        void get(mlir::ModuleOp m_mod, llvm::Module *l_mod) {
            annotate_functions(m_mod, l_mod);
            annotate_gvs(m_mod, l_mod);
        }
    };

    void emit_llvm(state_t &state) {
        auto &tower = state.tower;
        auto op = tower.last_module().clone();
        tower.make_snapshot("llvm");
        op->setAttr(
            mlir::DLTIDialect::kDataLayoutAttrName, mlir::DataLayoutSpecAttr::get(&state.ctx, {})
        );

        auto mod = mlir::translateModuleToLLVMIR(op, tower.llvm_context, "LLVMDialectModule");
        tower.llvm = std::move(mod);

        get_mapping mapping;
        // use old module snapshotted in tower
        mapping.get(tower.last_module(), tower.llvm.get());
        // TODO FIXME
        tower.value_mapping = mapping.function_mapping["main"];
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

    void symbols_with_name(auto module, string_ref name, auto &&yield) {
        util::symbols(module, [&] (auto symbol) {
            if (util::symbol_name(symbol) == name) {
                yield(symbol);
            }
        });
    }

    std::optional< mlir::Location > get_location_in_layer(mlir::FusedLoc fused, string_ref layer) {
        for (auto loc : fused.getLocations()) {
            if (auto named = llvm::dyn_cast< mlir::NameLoc >(loc)) {
                if (named.getName() == layer) {
                    return named;
                }
            }
        }

        return std::nullopt;
    }

    void project(
        const auto &orig, auto &layer, auto &&yield, string_ref from_layer = "source"
    ) {
        auto get_source_loc = [&] (auto op) {
            return get_location_in_layer(
                mlir::cast< mlir::FusedLoc >(op->getLoc()), from_layer
            );
        };

        auto origin_location = get_source_loc(orig).value();
        layer.walk([&] (operation op) {
            if (auto loc = get_source_loc(op); loc && origin_location == loc.value()) {
                yield(op);
            }
        });
    }

    void project_symbol(
        state_t &state, const auto &symbol_name, const auto &layer_name, auto &&yield,
        string_ref from_layer = "source"
    ) {
        symbols_with_name(state.tower.snaps[from_layer], symbol_name, [&] (auto symbol) {
            auto layer = state.tower.snaps[layer_name];
            project(symbol, layer, yield, from_layer);
        });
    }

    void inspect::run(state_t &state) const {
        auto layer_name = get_param< layer_param >(params);
        auto symbol_name = get_param< symbol_param >(params);

        if (layer_name.value == "llvm") {
            project_symbol(state, symbol_name.value, layer_name.value, [&] (auto op) {
                if (state.tower.value_mapping.count(op)) {
                    state.tower.value_mapping[op]->print(llvm::outs());
                    llvm::outs() << "\n";
                }
            });
        } else {
            project_symbol(state, symbol_name.value, layer_name.value, [] (auto op) {
                op->dump();
            });
        }
    }

    void alias::run(state_t &state) const {
        auto first_name  = get_param< first_param >(params);
        auto second_name = get_param< second_param >(params);

        auto in_llvm = [&] (auto symbol) {
            std::vector< llvm::Instruction * > ops;
            project_symbol(state, symbol, "irs-to-llvm", [&] (auto op) {
                if (state.tower.value_mapping.count(op)) {
                    ops.push_back(state.tower.value_mapping[op]);
                }
            });
            return ops;
        };

        auto a = in_llvm(first_name.value);
        auto b = in_llvm(second_name.value);

        auto main = state.tower.llvm->getFunction("main");
        auto info = function_analysis_info(main);

        auto alias_result = [&] () -> std::optional< llvm::AliasResult > {
            for (auto ai : a) {
                for (auto bi : b) {
                    if (auto res = info.aa.alias(ai, bi); static_cast< llvm::AliasResult::Kind >(res) != llvm::AliasResult::NoAlias) {
                        return res;
                    }
                }
            }

            return std::nullopt;
        } ();

        if (alias_result.has_value()) {
            llvm::outs() << alias_result.value() << "\n";
        } else {
            llvm::outs() << "no alias\n";
        }
    }

    void snapshot::run(state_t &state) const {
        auto snapshot_name = get_param< name_param >(params);
        auto locations = get_param< locations_param >(params);

        mlir::OpPrintingFlags flags;
        if (locations.set) {
            flags.enableDebugInfo();
        }
        state.tower.snaps[snapshot_name.value].print(llvm::outs(), flags);
    }

} // namespace cmd

    command_ptr parse_command(std::span< command_token > tokens) {
        return match< cmd::command_list >(tokens);
    }

} // namespace vast::repl

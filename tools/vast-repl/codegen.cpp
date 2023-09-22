// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/repl/codegen.hpp"
#include "vast/repl/common.hpp"

#include "vast/repl/state.hpp"

#include "vast/CodeGen/CodeGen.hpp"

#include <fstream>

namespace vast::repl::codegen {

    std::string slurp(std::ifstream& in) {
        std::ostringstream sstr;
        sstr << in.rdbuf();
        return sstr.str();
    }

    std::string get_source(std::filesystem::path source) {
        std::ifstream in(source);
        return slurp(in);
    }


    std::unique_ptr< clang::ASTUnit > ast_from_source(std::filesystem::path file_name) {
        auto source = get_source(file_name);
        return clang::tooling::buildASTFromCodeWithArgs(source, { "-xc" }, file_name.string());
    }

    owning_module_ref emit_module(std::filesystem::path file_name, mcontext_t *mctx) {
        auto unit = codegen::ast_from_source(file_name);
        auto &actx = unit->getASTContext();

        auto make_mlir_location = [&] (auto loc) {
            auto file = loc.getFileEntry() ? loc.getFileEntry()->getName() : "unknown";
            auto line = loc.getLineNumber();
            auto col  = loc.getColumnNumber();
            return mlir::FileLineColLoc::get(mctx, file, line, col);
        };

        auto loc = clang::FullSourceLoc(unit->getStartOfMainFileID(), actx.getSourceManager());

        vast::cg::CodeGenContext cgctx(*mctx, actx, make_mlir_location(loc));
        vast::cg::DefaultCodeGen codegen(cgctx);
        codegen.emit_module(unit.get());
        return std::move(cgctx.mod);
    }

} // namespace vast::repl::codegen

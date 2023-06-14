// Copyright (c) 2023, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include "llvm/ADT/ScopedHashTable.h"

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

#include "vast/Translation/Mangler.hpp"

#include <gap/core/generator.hpp>

#include <functional>
#include <queue>

namespace vast::cg
{
    template< typename From, typename To >
    struct scoped_table : llvm::ScopedHashTable< From, To >
    {
        using value_type = To;

        using base = llvm::ScopedHashTable< From, To >;
        using base::base;

        using base::count;
        using base::insert;

        logical_result declare(const From &from, const To &to) {
            //llvm::errs() << to << "\n";
            if (count(from)) {
                return mlir::failure();
            }

            insert(from, to);
            return mlir::success();
        }
    };

    struct scope_context {
        using action_t = std::function< void() >;

        ~scope_context() {
            for (const auto &action : deferred()) {
                action();
            }

            VAST_ASSERT(deferred_codegen_actions.empty());
        }

        gap::generator< action_t > deferred() {
            while (!deferred_codegen_actions.empty()) {
                co_yield deferred_codegen_actions.front();
                deferred_codegen_actions.pop();
            }
        }

        void defer(action_t action) {
            deferred_codegen_actions.emplace(std::move(action));
        }

        std::queue< action_t > deferred_codegen_actions;
    };

    template< typename From, typename Symbol >
    using table_scope = llvm::ScopedHashTableScope< From, Symbol >;

    using enum_constants_scope = table_scope< const clang::EnumConstantDecl *, hl::EnumConstantOp >;
    using enum_decls_scope     = table_scope< const clang::EnumDecl *, hl::EnumDeclOp >;
    using functions_scope      = table_scope< mangled_name_ref, hl::FuncOp >;
    using labels_scope         = table_scope< const clang::LabelDecl*, hl::LabelDeclOp >;
    using type_defs_scope      = table_scope< const clang::TypedefDecl *, hl::TypeDefOp >;
    using type_decls_scope     = table_scope< const clang::TypeDecl *, hl::TypeDeclOp >;
    using variables_scope      = table_scope< const clang::VarDecl *, Value >;

    struct CodeGenContext;

    // Refers to block scope §6.2.1 of C standard
    //
    // If the declarator or type specifier that declares the identifier appears
    // inside a block or within the list of parameter declarations in a function
    // definition, the identifier has block scope, which terminates at the end
    // of the associated block.
    struct block_scope : scope_context {
        enum_decls_scope enumdecls;
        type_decls_scope typedecls;
        type_defs_scope typedefs;
        variables_scope vars;

        block_scope() = delete;
        block_scope(CodeGenContext *);
    };


    // refers to function scope §6.2.1 of C standard
    struct function_scope : block_scope {
        labels_scope labels;

        function_scope() = delete;
        function_scope(CodeGenContext *);
    };

    // Refers to function prototype scope §6.2.1 of C standard
    //
    // If the declarator or type specifier that declares the identifier appears
    // within the list of parameter declarations in a function prototype (not
    // part of a function definition), the identifier has function prototype
    // scope, which terminates at the end of the function declarator
    struct prototype_scope : scope_context {
        enum_decls_scope enumdecls;
        type_decls_scope typedecls;
        variables_scope vars;

        prototype_scope() = delete;
        prototype_scope(CodeGenContext *);
    };

    // Refers to file scope §6.2.1 of C standard
    //
    // If the declarator or type specifier that declares the identifier appears
    // outside of any block or list of parameters, the identifier has file
    // scope, which terminates at the end of the translation unit.
    struct module_scope : scope_context {
        enum_decls_scope enumdecls;
        functions_scope functions;
        type_decls_scope typedecls;
        type_defs_scope typedefs;
        variables_scope vars;

        module_scope() = delete;
        module_scope(CodeGenContext *);
    };

    // Scope of member names for structures and unions

    struct members_scope : scope_context {
        enum_constants_scope enumconsts;
    };

} // namespace vast::cg

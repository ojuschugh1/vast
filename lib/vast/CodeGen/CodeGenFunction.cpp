// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/CodeGen/CodeGenDriver.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/GlobalDecl.h>
#include <clang/Basic/TargetInfo.h>
VAST_UNRELAX_WARNINGS

// FIXME: get rid of dependency from upper layer
#include "vast/CodeGen/TypeInfo.hpp"

namespace vast::cg
{
    bool codegen_driver::has_this_return(clang::GlobalDecl decl) const {
        return cxx_abi->has_this_return(decl);
    }

    bool codegen_driver::has_most_derived_return(clang::GlobalDecl decl) const {
        return cxx_abi->has_most_derived_return(decl);
    }

    function_arg_list codegen_driver::build_function_arg_list(clang::GlobalDecl decl) {
        const auto function_decl = clang::cast< clang::FunctionDecl >(decl.getDecl());
        // auto rty = function_decl->getReturnType();

        function_arg_list args;

        const auto *method = clang::dyn_cast< clang::CXXMethodDecl >(function_decl);
        if (method && method->isInstance()) {
            if (has_this_return(decl)) {
                VAST_UNIMPLEMENTED;
            }
            else if (has_most_derived_return(decl)) {
                VAST_UNIMPLEMENTED;
            }

            VAST_UNIMPLEMENTED;
            // get_cxx_abi.buildThisParam(*this, args);
        }

        // The base version of an inheriting constructor whose constructed base is a
        // virtual base is not passed any arguments (because it doesn't actually
        // call the inherited constructor).
        bool passed_params = [&] {
            if (const auto *ctor = clang::dyn_cast< clang::CXXConstructorDecl >(function_decl)) {
                if (auto inherited = ctor->getInheritedConstructor()) {
                    VAST_UNIMPLEMENTED_MSG("build_function_arg_list: inherited ctor");
                    // return getTypes().inheritingCtorHasParams(inherited, decl.getCtorType());
                }
            }
            return true;
        } ();

        if (passed_params) {
            for (auto *param : function_decl->parameters()) {
                args.push_back(param);
                if (param->hasAttr< clang::PassObjectSizeAttr >()) {
                    VAST_UNIMPLEMENTED_MSG("build_function_arg_list: PassObjectSizeAttr function param");
                }
            }
        }

        if (method) {
            if (clang::isa< clang::CXXConstructorDecl >(method) || clang::isa< clang::CXXDestructorDecl>(method)) {
                VAST_UNIMPLEMENTED;
                // get_cxx_abi().addImplicitStructorParams(*this, rty, args);
            }
        }

        return args;
    }

    bool codegen_driver::may_drop_function_return(qual_type rty) const {
        // We can't just disard the return value for a record type with a complex
        // destructor or a non-trivially copyable type.
        if (const auto *recorrd_type = rty.getCanonicalType()->getAs< clang::RecordType >()) {
            VAST_UNIMPLEMENTED;
        }

        return rty.isTriviallyCopyableType(acontext());
    }

    void codegen_driver::deal_with_missing_return(hl::FuncOp fn, const clang::FunctionDecl *decl) {
        auto rty = decl->getReturnType();

        bool shoud_emit_unreachable = (
            opts.codegen.StrictReturn || may_drop_function_return(rty)
        );

        // if (SanOpts.has(SanitizerKind::Return)) {
        //     VAST_UNIMPLEMENTED;
        // }

        if (rty->isVoidType()) {
            codegen.emit_implicit_void_return(fn, decl);
        } else if (decl->hasImplicitReturnZero()) {
            codegen.emit_implicit_return_zero(fn, decl);
        } else if (shoud_emit_unreachable) {
            // C++11 [stmt.return]p2:
            //   Flowing off the end of a function [...] results in undefined behavior
            //   in a value-returning function.
            // C11 6.9.1p12:
            //   If the '}' that terminates a function is reached, and the value of the
            //   function call is used by the caller, the behavior is undefined.

            // TODO: skip if SawAsmBlock
            if (opts.codegen.OptimizationLevel == 0) {
                codegen.emit_trap(fn, decl);
            } else {
                codegen.emit_unreachable(fn, decl);
            }
        } else {
            VAST_UNIMPLEMENTED_MSG("unknown missing return case");
        }
    }

    operation get_last_effective_operation(auto &block) {
        if (block.empty()) {
            return {};
        }
        auto last = &block.back();
        if (auto scope = mlir::dyn_cast< core::ScopeOp >(last)) {
            return get_last_effective_operation(scope.getBody().back());
        }

        return last;
    }

    hl::FuncOp codegen_driver::emit_function_epilogue(hl::FuncOp fn, clang::GlobalDecl decl) {
        auto function_decl = clang::cast< clang::FunctionDecl >( decl.getDecl() );

        auto &last_block = fn.getBody().back();
        auto missing_return = [&] (auto &block) {
            if (codegen.has_insertion_block()) {
                if (auto op = get_last_effective_operation(block)) {
                    return !op->template hasTrait< core::return_trait >();
                }
                return true;
            }

            return false;
        };

        if (missing_return(last_block)) {
            deal_with_missing_return(fn, function_decl);
        }


        // Emit the standard function epilogue.
        // TODO: finishFunction(BodyRange.getEnd());

        // If we haven't marked the function nothrow through other means, do a quick
        // pass now to see if we can.
        // TODO: if (!CurFn->doesNotThrow()) TryMarkNoThrow(CurFn);

        return fn;
    }

    // This function implements the logic from CodeGenFunction::GenerateCode
    hl::FuncOp codegen_driver::build_function_body(
        hl::FuncOp fn, clang::GlobalDecl decl, const function_info_t &fty_info
    ) {
        auto args = build_function_arg_list(decl);
        fn = codegen.emit_function_prologue(
            fn, decl, fty_info, args, opts
        );

        if (mlir::failed(fn.verifyBody())) {
            return nullptr;
        }

        return emit_function_epilogue(fn, decl);
    }

} // namespace vast::cg

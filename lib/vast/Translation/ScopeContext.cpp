#include "vast/Translation/ScopeContext.hpp"
#include "vast/Translation/CodeGenContext.hpp"

namespace vast::cg{
    block_scope::block_scope(CodeGenContext *ctx) : enumdecls(ctx->enumdecls),
                                                    typedecls(ctx->typedecls),
                                                    typedefs(ctx->typedefs),
                                                    vars(ctx->vars)
                                                    {}

    function_scope::function_scope(CodeGenContext *ctx) : block_scope(ctx),
                                                          labels(ctx->labels)
                                                          {}

    prototype_scope::prototype_scope(CodeGenContext *ctx) : enumdecls(ctx->enumdecls),
                                                            typedecls(ctx->typedecls),
                                                            vars(ctx->vars)
                                                            {}

    module_scope::module_scope(CodeGenContext *ctx) : enumdecls(ctx->enumdecls),
                                                      functions(ctx->funcdecls),
                                                      typedecls(ctx->typedecls),
                                                      typedefs(ctx->typedefs),
                                                      vars(ctx->vars)
                                                      {}
} //namespace vast::cg

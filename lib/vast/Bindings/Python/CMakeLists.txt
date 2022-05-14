#
# Set up Python binding tools
#

include(AddMLIRPython)

#
# Declare native Python extension
#

declare_mlir_python_sources(VASTBindingsPythonExtension)

# declare_mlir_python_extension(VASTBindingsPythonExtension.Core
#   MODULE_NAME _vast
#   ADD_TO_PARENT VASTBindingsPythonExtension
#   SOURCES
#     VASTModule.cpp
#     ESIModule.cpp
#     HWModule.cpp
#     MSFTModule.cpp
#   EMBED_CAPI_LINK_LIBS
#     VASTCAPIComb
#     VASTCAPIESI
#     VASTCAPIMSFT
#     VASTCAPIHW
#     VASTCAPISeq
#     VASTCAPISV
#     VASTCAPIExportVerilog
#   PRIVATE_LINK_LIBS
#     ${_depends}
# )

# ################################################################################
# # Declare Python sources
# ################################################################################

# declare_mlir_python_sources(VASTBindingsPythonSources
#   ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
#   SOURCES
#     vast/__init__.py
#     vast/esi.py
#     vast/support.py
#     vast/dialects/_ods_common.py)

# ################################################################################
# # Declare dialect-specific bindings.
# ################################################################################

# # Ensure the build directory for generated Python files exists. Ninja is able to
# # generate this, but make does not and the build fails.
# file(MAKE_DIRECTORY ${VAST_BINARY_DIR}/lib/Bindings/Python/vast/dialects)

# declare_mlir_python_sources(VASTBindingsPythonSources.Dialects
#   ADD_TO_PARENT VASTBindingsPythonSources)

# declare_mlir_dialect_python_bindings(
#   ADD_TO_PARENT VASTBindingsPythonSources.Dialects
#   ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
#   TD_FILE vast/dialects/CombOps.td
#   SOURCES
#     vast/dialects/comb.py
#     vast/dialects/_comb_ops_ext.py
#   DIALECT_NAME comb)

# declare_mlir_dialect_python_bindings(
#   ADD_TO_PARENT VASTBindingsPythonSources.Dialects
#   ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
#   TD_FILE vast/dialects/ESIOps.td
#   SOURCES vast/dialects/esi.py
#   DIALECT_NAME esi)

# declare_mlir_dialect_python_bindings(
#   ADD_TO_PARENT VASTBindingsPythonSources.Dialects
#   ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
#   TD_FILE vast/dialects/HWOps.td
#   SOURCES
#     vast/dialects/hw.py
#     vast/dialects/_hw_ops_ext.py
#   DIALECT_NAME hw)

# declare_mlir_dialect_python_bindings(
#   ADD_TO_PARENT VASTBindingsPythonSources.Dialects
#   ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
#   TD_FILE vast/dialects/MSFTOps.td
#   SOURCES
#     vast/dialects/msft.py
#     vast/dialects/_msft_ops_ext.py
#   DIALECT_NAME msft)

# declare_mlir_dialect_python_bindings(
#   ADD_TO_PARENT VASTBindingsPythonSources.Dialects
#   ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
#   TD_FILE vast/dialects/SeqOps.td
#   SOURCES
#     vast/dialects/seq.py
#     vast/dialects/_seq_ops_ext.py
#   DIALECT_NAME seq)

# declare_mlir_dialect_python_bindings(
#   ADD_TO_PARENT VASTBindingsPythonSources.Dialects
#   ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
#   TD_FILE vast/dialects/SVOps.td
#   SOURCES
#     vast/dialects/sv.py
#     vast/dialects/_sv_ops_ext.py
#   DIALECT_NAME sv)

# ################################################################################
# # Build composite binaries
# ################################################################################

# # Bundle our own, self-contained CAPI library with all of our deps.
# add_mlir_python_common_capi_library(VASTBindingsPythonCAPI
#   INSTALL_COMPONENT VASTBindingsPythonModules
#   INSTALL_DESTINATION python_packages/vast_core/mlir/_mlir_libs
#   # NOTE: When the MLIR API is relocated under vast, this would change to
#   # .../vast/_mlir_libs
#   OUTPUT_DIRECTORY "${VAST_PYTHON_PACKAGES_DIR}/vast_core/mlir/_mlir_libs"
#   RELATIVE_INSTALL_ROOT "../../../.."
#   DECLARED_SOURCES
#     MLIRPythonSources.Core
#     VASTBindingsPythonSources
#     VASTBindingsPythonExtension
# )

# # Bundle the MLIR python sources into our package.
# # The MLIR API is position independent, so we explicitly output it to the mlir/
# # folder as a temporary measure. It will eventually migrate under the vast/
# # folder and be accessible under the unified "import vast..." namespace.
# add_mlir_python_modules(VASTMLIRPythonModules
#   ROOT_PREFIX "${VAST_PYTHON_PACKAGES_DIR}/vast_core/mlir"
#   INSTALL_PREFIX "python_packages/vast_core/mlir"
#   DECLARED_SOURCES
#     MLIRPythonSources.Core
#     # We need the vast extensions co-located with the MLIR extensions. When
#     # the namespace is unified, this moves to the below.
#     VASTBindingsPythonExtension
#   COMMON_CAPI_LINK_LIBS
#     VASTBindingsPythonCAPI
# )

# # Bundle the VAST python sources into our package.
# add_mlir_python_modules(VASTPythonModules
#   ROOT_PREFIX "${VAST_PYTHON_PACKAGES_DIR}/vast_core"
#   INSTALL_PREFIX "python_packages/vast_core"
#   DECLARED_SOURCES
#     VASTBindingsPythonSources
#   COMMON_CAPI_LINK_LIBS
#     VASTBindingsPythonCAPI
# )

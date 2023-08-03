# Copyright (c) 2023, Trail of Bits, Inc.

add_library(vast_analysis
    DataFlow.cpp
)

target_link_libraries(vast_analysis PRIVATE vast::settings)

add_dependencies(vast_analysis VASTAnalysisPassIncGen)
add_library(vast::analysis ALIAS vast_analysis)

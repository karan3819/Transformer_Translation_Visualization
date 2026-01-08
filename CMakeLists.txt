cmake_minimum_required(VERSION 3.15)
project(TransformerTranslationVisualization 
        VERSION 1.0.0
        DESCRIPTION "Transformer-based Neural Machine Translation with Attention Visualization"
        LANGUAGES CXX)

# ==============================================================================
# PROJECT CONFIGURATION
# ==============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -DNDEBUG")

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ==============================================================================
# SOURCE FILES
# ==============================================================================

set(HEADERS
    transformer_translation_viz.hpp
)

set(SOURCES
    main.cpp
)

# ==============================================================================
# BUILD TARGETS
# ==============================================================================

# Main executable
add_executable(transformer_translator ${SOURCES} ${HEADERS})

# Set output directory
set_target_properties(transformer_translator PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ==============================================================================
# COMPILER OPTIONS
# ==============================================================================

target_compile_options(transformer_translator PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
)

# ==============================================================================
# DEPENDENCIES
# ==============================================================================

# No external dependencies required for core functionality
# Standard library includes only

# ==============================================================================
# INSTALLATION
# ==============================================================================

install(TARGETS transformer_translator
        RUNTIME DESTINATION bin)

install(FILES ${HEADERS}
        DESTINATION include)

# ==============================================================================
# TESTING
# ==============================================================================

enable_testing()

# Test 1: Basic execution
add_test(NAME BasicExecution
         COMMAND transformer_translator "hello world"
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# ==============================================================================
# DOCUMENTATION
# ==============================================================================

# Optional: Add Doxygen documentation generation
find_package(Doxygen)
if(DOXYGEN_FOUND)
    set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/docs)
    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE README.md)
    
    doxygen_add_docs(docs
                     transformer_translation_viz.hpp
                     main.cpp
                     README.md
                     COMMENT "Generating documentation")
endif()

# ==============================================================================
# PRINT CONFIGURATION
# ==============================================================================

message(STATUS "=================================================")
message(STATUS "Transformer Translation Visualization (C++)")
message(STATUS "=================================================")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Installation Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "=================================================")

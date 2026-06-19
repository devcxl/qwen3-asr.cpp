# Build the Python extension module (_core) using pybind11

# ---- pybind11 detection ---------------------------------------------------
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    # Fallback: try the installed pybind11 via Python
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        ERROR_QUIET
    )
    if(PYBIND11_CMAKE_DIR)
        list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
        find_package(pybind11 CONFIG REQUIRED)
    else()
        message(FATAL_ERROR "pybind11 not found. Install with: pip install pybind11")
    endif()
endif()

message(STATUS "pybind11 found: ${pybind11_DIR}")

# ---- Shared library: _core (Python extension) -----------------------------
pybind11_add_module(_core
    python/bindings.cpp
)
target_include_directories(_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(_core PRIVATE
    qwen3_asr
    forced_aligner
)
set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/qwen3_asr"
)

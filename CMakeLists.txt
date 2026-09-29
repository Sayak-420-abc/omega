cmake_minimum_required(VERSION 3.18)
project(highperf_ai_ml_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_ONNXRUNTIME "Build with ONNX Runtime backend" ON)
option(BUILD_LIBTORCH    "Build with LibTorch backend (TorchScript)" OFF)
option(BUILD_REST_API    "Build REST API server (httplib)" ON)
option(FORCE_CPU_ORT     "Force linking against CPU ONNX Runtime even if GPU lib exists" OFF)

# Put binaries in build/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
if (CMAKE_CONFIGURATION_TYPES)
  foreach(cfg IN LISTS CMAKE_CONFIGURATION_TYPES)
    string(TOUPPER "${cfg}" cfgU)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfgU} ${CMAKE_BINARY_DIR}/bin)
  endforeach()
endif()

include(FetchContent)

# cxxopts
FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
  GIT_TAG        v3.2.0
)
FetchContent_MakeAvailable(cxxopts)

# cpp-httplib
FetchContent_Declare(
  httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG        v0.15.3
)
FetchContent_MakeAvailable(httplib)

# stb
FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG        master
)
FetchContent_MakeAvailable(stb)

# --- ONNX Runtime (expects prebuilt under third_party/onnxruntime via fetch script) ---
if(BUILD_ONNXRUNTIME)
  set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime" CACHE PATH "Path to onnxruntime root")

  if (EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h")
    add_library(onnxruntime SHARED IMPORTED)
    if (WIN32)
      set_target_properties(onnxruntime PROPERTIES
        IMPORTED_IMPLIB "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_DIR}/include")
    elseif(APPLE)
      # macOS builds are CPU in our script
      set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_DIR}/include")
    else()
      # Linux: prefer GPU lib if available and not forced to CPU
      if (NOT FORCE_CPU_ORT AND EXISTS "${ONNXRUNTIME_DIR}/lib/libonnxruntime_gpu.so")
        message(STATUS "Linking ONNX Runtime GPU")
        set_target_properties(onnxruntime PROPERTIES
          IMPORTED_LOCATION "${ONNXRUNTIME_DIR}/lib/libonnxruntime_gpu.so"
          INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_DIR}/include")
        add_compile_definitions(ORT_HAS_CUDA=1)
      else()
        message(STATUS "Linking ONNX Runtime CPU")
        set_target_properties(onnxruntime PROPERTIES
          IMPORTED_LOCATION "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so"
          INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_DIR}/include")
      endif()
    endif()
    add_compile_definitions(HAS_ONNX_BACKEND=1)
  else()
    message(WARNING "ONNXRuntime not found at ${ONNXRUNTIME_DIR}. Run scripts/fetch_assets.sh or disable BUILD_ONNXRUNTIME.")
  endif()
endif()

# --- LibTorch optional ---
if(BUILD_LIBTORCH)
  set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/third_party/libtorch")
  find_package(Torch QUIET)
  if (Torch_FOUND)
    message(STATUS "Found LibTorch at ${Torch_DIR}")
    add_compile_definitions(HAS_TORCH_BACKEND=1)
  else()
    message(WARNING "LibTorch not found. Place it under third_party/libtorch or disable BUILD_LIBTORCH.")
  endif()
endif()

set(SRC
  src/main.cpp
  src/image_io.cpp
  src/backend_onnx.cpp
  src/backend_libtorch.cpp
  src/infer_api.cpp
)

add_executable(highperf_ai_ml_inference ${SRC})
set_target_properties(highperf_ai_ml_inference PROPERTIES OUTPUT_NAME "highperf-ai-ml-inference")
target_include_directories(highperf_ai_ml_inference PRIVATE ${CMAKE_SOURCE_DIR}/include ${stb_SOURCE_DIR})
target_link_libraries(highperf_ai_ml_inference PRIVATE cxxopts)

if(BUILD_REST_API)
  target_compile_definitions(highperf_ai_ml_inference PRIVATE HAS_REST_API=1)
  target_link_libraries(highperf_ai_ml_inference PRIVATE httplib)
endif()

if(BUILD_ONNXRUNTIME AND TARGET onnxruntime)
  target_link_libraries(highperf_ai_ml_inference PRIVATE onnxruntime)
endif()

if(BUILD_LIBTORCH AND Torch_FOUND)
  target_link_libraries(highperf_ai_ml_inference PRIVATE "${TORCH_LIBRARIES}")
  target_compile_options(highperf_ai_ml_inference PRIVATE ${TORCH_CXX_FLAGS})
endif()

# Copy ONNX Runtime DLL on Windows so tests can run
if (WIN32 AND BUILD_ONNXRUNTIME)
  add_custom_command(TARGET highperf_ai_ml_inference POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${ONNXRUNTIME_DIR}/bin/onnxruntime.dll"
      "$<TARGET_FILE_DIR:highperf_ai_ml_inference>/")
endif()

# --- Tests (CTest smoke) ---
include(CTest)
enable_testing()
add_test(
  NAME smoke_infer
  COMMAND $<TARGET_FILE:highperf_ai_ml_inference>
          --backend onnx
          --model ${CMAKE_SOURCE_DIR}/models/squeezenet1.1.onnx
          --input ${CMAKE_SOURCE_DIR}/assets/sample.jpg
          --topk 3
)

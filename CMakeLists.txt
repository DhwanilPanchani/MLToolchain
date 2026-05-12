cmake_minimum_required(VERSION 3.20)
project(MLToolchain CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------
# Stub target — demonstrates the build entry point.
# In production this would be replaced with:
#
#   find_package(onnxruntime REQUIRED)   # ONNX Runtime C++ SDK
#   find_package(QNN REQUIRED)           # Qualcomm QNN backend for INT8 on Edge SoCs
#
#   target_link_libraries(mltoolchain
#       PRIVATE onnxruntime::onnxruntime
#               QNN::HTP            # Hexagon Tensor Processor (HTP) backend
#   )
#
# The INT8 model exported by the Python pipeline would then be loaded with:
#   Ort::Session session(env, L"models/quantized/mobilenetv2_int8.onnx", session_options);
# ---------------------------------------------------------

add_executable(mltoolchain cpp/stub.cpp)

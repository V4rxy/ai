cmake_minimum_required(VERSION 3.24)
project(EngineX VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt6 6.5 REQUIRED COMPONENTS Widgets)
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio)
find_package(onnxruntime CONFIG QUIET)

add_executable(EngineX
    src/main.cpp
    src/MainWindow.cpp
    src/MainWindow.h
    src/CapturePanel.cpp
    src/CapturePanel.h
    src/ModelInspector.cpp
    src/ModelInspector.h
)

target_link_libraries(EngineX PRIVATE Qt6::Widgets ${OpenCV_LIBS})

if(TARGET onnxruntime::onnxruntime)
    target_link_libraries(EngineX PRIVATE onnxruntime::onnxruntime)
    target_compile_definitions(EngineX PRIVATE ENGINEX_HAS_ONNXRUNTIME=1)
elseif(TARGET onnxruntime)
    target_link_libraries(EngineX PRIVATE onnxruntime)
    target_compile_definitions(EngineX PRIVATE ENGINEX_HAS_ONNXRUNTIME=1)
else()
    message(STATUS "ONNX Runtime not found: model inspection will report unavailable.")
endif()

target_include_directories(EngineX PRIVATE src ${OpenCV_INCLUDE_DIRS})

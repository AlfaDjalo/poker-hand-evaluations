cmake_minimum_required(VERSION 3.12)
project(ompeval_cpp)

set(CMAKE_CXX_STANDARD 17)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --includes
                OUTPUT_VARIABLE PYBIND11_INCLUDES)
# better: use find_package(pybind11) if installed

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../OMPEval/include)
add_compile_options(-O3 -march=native)

pybind11_add_module(ompeval_cpp equity_wrapper.cpp)
target_include_directories(ompeval_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../OMPEval/include)
target_link_libraries(ompeval_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../OMPEval/libompeval.a) # if you compiled static lib

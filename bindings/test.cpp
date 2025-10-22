#include <pybind11/pybind11.h>
#include "EquityCalculator.h"

namespace py = pybind11;

PYBIND11_MODULE(ompeval, m) {
    m.doc() = "Python bindings for OMPEval";

    m.def("example_function", []() {
        return 42;
    });

    // Later you can wrap your classes:
    // py::class_<EquityCalculator>(m, "EquityCalculator")
    //     .def(py::init<>())
    //     .def("start", &EquityCalculator::start);
}

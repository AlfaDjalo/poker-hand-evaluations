#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "EquityCalculator.h"

namespace py = pybind11;

PYBIND11_MODULE(ompeval, m) {
    m.doc() = "Minimal OMPEval binding";
    
    // Expose CardRange so Python can construct hand ranges
    py::class_<omp::CardRange>(m, "CardRange")
    .def(py::init<const std::string &>());

    py::class_<omp::EquityCalculator::Results>(m, "Results")
        .def_readonly("players", &omp::EquityCalculator::Results::players)
        .def_readonly("equity", &omp::EquityCalculator::Results::equity)
        .def_readonly("wins", &omp::EquityCalculator::Results::wins)
        .def_readonly("ties", &omp::EquityCalculator::Results::ties)
        .def_readonly("hands", &omp::EquityCalculator::Results::hands)
        .def_readonly("finished", &omp::EquityCalculator::Results::finished);

    py::class_<omp::EquityCalculator>(m, "EquityCalculator")
        .def(py::init<>())
        .def("start", [](omp::EquityCalculator &self,
                        const std::vector<omp::CardRange>& handRanges,
                        uint64_t boardCards,
                        uint64_t deadCards,
                        bool enumerateAll,
                        double stdevTarget) {
            // Dummy callback that does nothing
            auto dummy_callback = [](const omp::EquityCalculator::Results&) {};
            // Provide non-zero max time and random seed
            return self.start(handRanges, boardCards, deadCards, enumerateAll, stdevTarget, dummy_callback, 1.0, 12345);
        },
            py::arg("handRanges") = std::vector<omp::CardRange>(),
            py::arg("boardCards") = 0,
            py::arg("deadCards") = 0,
            py::arg("enumerateAll") = false,
            py::arg("stdevTarget") = 5e-5)
        .def("get_results", &omp::EquityCalculator::getResults)

        .def("wait", &omp::EquityCalculator::wait);

}


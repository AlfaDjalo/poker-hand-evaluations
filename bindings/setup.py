from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "ompeval",
        [
            "ompeval_bindings.cpp",
            "../OMPEval/omp/CardRange.cpp",
            "../OMPEval/omp/CombinedRange.cpp",
            "../OMPEval/omp/EquityCalculator.cpp",
            "../OMPEval/omp/HandEvaluator.cpp",
        ],
        include_dirs=["../OMPEval/omp"],
        extra_compile_args=["/O2"],
    ),
    Pybind11Extension(
        "equity_wrapper",
        ["equity_wrapper.cpp",
        "omaha_equity.cpp",
        "../OMPEval/omp/CardRange.cpp",
        "../OMPEval/omp/CombinedRange.cpp",
        "../OMPEval/omp/EquityCalculator.cpp",
        "../OMPEval/omp/HandEvaluator.cpp"],
        include_dirs=["../OMPEval/omp"],
        extra_compile_args=["/O2"],
    ),
    Pybind11Extension(
        "holdem_wrapper",
        ["holdem_wrapper.cpp",
        "holdem_equity.cpp",
        "../OMPEval/omp/CardRange.cpp",
        "../OMPEval/omp/CombinedRange.cpp",
        "../OMPEval/omp/EquityCalculator.cpp",
        "../OMPEval/omp/HandEvaluator.cpp"],
        include_dirs=["../OMPEval/omp"],
        extra_compile_args=["/O2"],
    )
]

setup(
    name="poker_bindings",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
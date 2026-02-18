#define PY_SSIZE_T_CLEAN
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Eigen/Dense>

#include <variant>
#include <vector>
#include <complex>
#include <stdexcept>
#include <Python.h>
#include <type_traits>
#include <cstring>

#include <riccati/solver.hpp>
#include <riccati/evolve.hpp>
#include <riccati/stepsize.hpp>
#include <riccati/vectorizer.hpp>
#include <riccati/utils.hpp>

namespace py = pybind11;
using complex_t = std::complex<double>;

template <typename Vec>
py::array_t<typename Vec::value_type> vector_to_array_copy(const Vec& vec) {
    using value_t = typename Vec::value_type;
    const auto size = static_cast<std::size_t>(vec.size());
    py::array_t<value_t> arr(vec.size());
    if (size != 0) {
        std::memcpy(arr.mutable_data(), vec.data(), size * sizeof(value_t));
    }
    return arr;
}

// Callback wrappers using pybind11 functions
template <typename ReturnType>
struct CallbackWrapper {
    py::function func;

    CallbackWrapper() = default;
    explicit CallbackWrapper(py::function f) : func(std::move(f)) {}

    ReturnType operator()(double x) const {
        py::gil_scoped_acquire gil;
        py::object res = func(x);
        if constexpr (std::is_same_v<ReturnType, complex_t>) {
            return res.cast<complex_t>();
        } else {
            return res.cast<ReturnType>();
        }
    }

    template <typename EigenVec>
    Eigen::Matrix<ReturnType, -1, 1> operator()(const EigenVec& x) const {
        py::gil_scoped_acquire gil;
        py::object res = func(x);
        const Eigen::Index expected = static_cast<Eigen::Index>(x.size());

        auto broadcast = [&](ReturnType value) {
            Eigen::Matrix<ReturnType, -1, 1> out(expected);
            out.setConstant(value);
            return out;
        };

        if (py::isinstance<py::array>(res)) {
            py::array arr = py::cast<py::array>(res);
            if (arr.ndim() == 0) {
                return broadcast(py::cast<ReturnType>(res));
            }
            py::array_t<ReturnType, py::array::c_style | py::array::forcecast> casted(arr);
            const Eigen::Index size = static_cast<Eigen::Index>(casted.size());
            if (size == expected) {
                Eigen::Matrix<ReturnType, -1, 1> out(expected);
                std::memcpy(out.data(), casted.data(), static_cast<std::size_t>(expected) * sizeof(ReturnType));
                return out;
            }
            if (size == 1) {
                return broadcast(*casted.data());
            }
            throw std::runtime_error("Callback returned array with incompatible shape");
        }

        return broadcast(res.cast<ReturnType>());
    }
};

// Scalar-only callback wrapper for non-vectorized Python functions.
// Only handles scalar double input; used with riccati::Vectorizer to
// automatically decompose vector inputs into element-by-element calls.
template <typename ReturnType>
struct ScalarCallbackWrapper {
    py::function func;

    ScalarCallbackWrapper() = default;
    explicit ScalarCallbackWrapper(py::function f) : func(std::move(f)) {}

    ReturnType operator()(double x) const {
        py::gil_scoped_acquire gil;
        py::object res = func(x);
        if constexpr (std::is_same_v<ReturnType, complex_t>) {
            return res.cast<complex_t>();
        } else {
            return res.cast<ReturnType>();
        }
    }
};

using SolverDD = riccati::SolverInfo<CallbackWrapper<double>, CallbackWrapper<double>, double, int64_t,
    riccati::arena_allocator<double, riccati::arena_alloc>, riccati::EmptyLogger, double, double>;
using SolverCD = riccati::SolverInfo<CallbackWrapper<complex_t>, CallbackWrapper<double>, double, int64_t,
    riccati::arena_allocator<double, riccati::arena_alloc>, riccati::EmptyLogger, complex_t, double>;
using SolverDC = riccati::SolverInfo<CallbackWrapper<double>, CallbackWrapper<complex_t>, double, int64_t,
    riccati::arena_allocator<double, riccati::arena_alloc>, riccati::EmptyLogger, double, complex_t>;

using SolverCC = riccati::SolverInfo<CallbackWrapper<complex_t>, CallbackWrapper<complex_t>, double, int64_t,
    riccati::arena_allocator<double, riccati::arena_alloc>, riccati::EmptyLogger, complex_t, complex_t>;

// Vectorized solver types: wrap scalar-only callbacks in riccati::Vectorizer
using VecSolverDD = riccati::SolverInfo<
    riccati::Vectorizer<ScalarCallbackWrapper<double>>,
    riccati::Vectorizer<ScalarCallbackWrapper<double>>,
    double, int64_t, riccati::arena_allocator<double, riccati::arena_alloc>,
    riccati::EmptyLogger, double, double>;
using VecSolverCD = riccati::SolverInfo<
    riccati::Vectorizer<ScalarCallbackWrapper<complex_t>>,
    riccati::Vectorizer<ScalarCallbackWrapper<double>>,
    double, int64_t, riccati::arena_allocator<double, riccati::arena_alloc>,
    riccati::EmptyLogger, complex_t, double>;
using VecSolverDC = riccati::SolverInfo<
    riccati::Vectorizer<ScalarCallbackWrapper<double>>,
    riccati::Vectorizer<ScalarCallbackWrapper<complex_t>>,
    double, int64_t, riccati::arena_allocator<double, riccati::arena_alloc>,
    riccati::EmptyLogger, double, complex_t>;
using VecSolverCC = riccati::SolverInfo<
    riccati::Vectorizer<ScalarCallbackWrapper<complex_t>>,
    riccati::Vectorizer<ScalarCallbackWrapper<complex_t>>,
    double, int64_t, riccati::arena_allocator<double, riccati::arena_alloc>,
    riccati::EmptyLogger, complex_t, complex_t>;

using SolverVariant = std::variant<SolverDD, SolverCD, SolverDC, SolverCC,
                                   VecSolverDD, VecSolverCD, VecSolverDC, VecSolverCC>;

enum class CallbackType { Double = 0, Complex = 1 };

static CallbackType detect_type(py::function func) {
    py::gil_scoped_acquire gil;
    py::object res = func(0.5);
    PyObject* raw = res.ptr();
    if (PyComplex_Check(raw)) {
        return CallbackType::Complex;
    }
    try {
        auto dt = py::dtype::from_args(res);
        const char kind = dt.kind();
        if (kind == 'c') {
            return CallbackType::Complex;
        }
    } catch (const std::exception&) {
        // Fall through to default real if dtype introspection fails.
    }
    return CallbackType::Double;
}

static SolverVariant make_solver_variant(py::function omega_fun, py::function gamma_fun,
                                         int nini, int nmax, int n, int p) {
    auto otype = detect_type(omega_fun);
    auto gtype = detect_type(gamma_fun);
    switch (otype) {
      case CallbackType::Double:
        switch (gtype) {
          case CallbackType::Double:
              return SolverVariant(std::in_place_type<SolverDD>,
                                   CallbackWrapper<double>(omega_fun),
                                   CallbackWrapper<double>(gamma_fun),
                                   nini, nmax, n, p);
          case CallbackType::Complex:
              return SolverVariant(std::in_place_type<SolverDC>,
                                   CallbackWrapper<double>(omega_fun),
                                   CallbackWrapper<complex_t>(gamma_fun),
                                   nini, nmax, n, p);
        }
      case CallbackType::Complex:
          switch (gtype) {
            case CallbackType::Double:
                return SolverVariant(std::in_place_type<SolverCD>,
                                     CallbackWrapper<complex_t>(omega_fun),
                                     CallbackWrapper<double>(gamma_fun),
                                     nini, nmax, n, p);
            case CallbackType::Complex:
                return SolverVariant(std::in_place_type<SolverCC>,
                                     CallbackWrapper<complex_t>(omega_fun),
                                     CallbackWrapper<complex_t>(gamma_fun),
                                     nini, nmax, n, p);
          }
    }
    // TODO: Nicer error message
    throw std::domain_error("Unsupported callback types");
}

static SolverVariant make_vectorized_solver_variant(py::function omega_fun, py::function gamma_fun,
                                                     int nini, int nmax, int n, int p) {
    auto otype = detect_type(omega_fun);
    auto gtype = detect_type(gamma_fun);
    switch (otype) {
      case CallbackType::Double:
        switch (gtype) {
          case CallbackType::Double:
              return SolverVariant(std::in_place_type<VecSolverDD>,
                                   riccati::vectorize(ScalarCallbackWrapper<double>(omega_fun)),
                                   riccati::vectorize(ScalarCallbackWrapper<double>(gamma_fun)),
                                   nini, nmax, n, p);
          case CallbackType::Complex:
              return SolverVariant(std::in_place_type<VecSolverDC>,
                                   riccati::vectorize(ScalarCallbackWrapper<double>(omega_fun)),
                                   riccati::vectorize(ScalarCallbackWrapper<complex_t>(gamma_fun)),
                                   nini, nmax, n, p);
        }
      case CallbackType::Complex:
          switch (gtype) {
            case CallbackType::Double:
                return SolverVariant(std::in_place_type<VecSolverCD>,
                                     riccati::vectorize(ScalarCallbackWrapper<complex_t>(omega_fun)),
                                     riccati::vectorize(ScalarCallbackWrapper<double>(gamma_fun)),
                                     nini, nmax, n, p);
            case CallbackType::Complex:
                return SolverVariant(std::in_place_type<VecSolverCC>,
                                     riccati::vectorize(ScalarCallbackWrapper<complex_t>(omega_fun)),
                                     riccati::vectorize(ScalarCallbackWrapper<complex_t>(gamma_fun)),
                                     nini, nmax, n, p);
          }
    }
    throw std::domain_error("Unsupported callback types");
}

class RiccatiSolver {
public:
    RiccatiSolver(py::function omega_fun, py::function gamma_fun,
                  int nini, int nmax, int n, int p, bool vectorized = true)
        : solver_(vectorized
                  ? make_solver_variant(std::move(omega_fun), std::move(gamma_fun),
                                        nini, nmax, n, p)
                  : make_vectorized_solver_variant(std::move(omega_fun), std::move(gamma_fun),
                                                    nini, nmax, n, p)) {}

    py::tuple solve(double xi, double xf, complex_t yi, complex_t dyi,
                    double eps, double epsilon_h, double init_stepsize,
                    py::object t_eval, bool hard_stop) {
        py::gil_scoped_release release;

        auto run = [&](auto& solver) -> py::tuple {
            using solver_t = std::decay_t<decltype(solver)>;
            using complex_scalar = typename solver_t::complex_t;

            Eigen::VectorXd t_eval_vec;
            bool use_teval = false;
            if (!t_eval.is_none()) {
                py::gil_scoped_acquire gil;
                auto arr = py::cast<py::array_t<double>>(t_eval);
                py::buffer_info info = arr.request();
                t_eval_vec = Eigen::Map<Eigen::VectorXd>(static_cast<double*>(info.ptr),
                                                         static_cast<Eigen::Index>(info.shape[0]));
                use_teval = true;
            }

            complex_scalar y_init = static_cast<complex_scalar>(yi);
            complex_scalar dy_init = static_cast<complex_scalar>(dyi);

            auto result = use_teval
                ? riccati::evolve(solver, xi, xf, y_init, dy_init, eps, epsilon_h, init_stepsize, t_eval_vec, hard_stop)
                : riccati::evolve(solver, xi, xf, y_init, dy_init, eps, epsilon_h, init_stepsize,
                                  Eigen::Matrix<double, 0, 0>{}, hard_stop);

            auto& [xs, ys, dys, successes, phases, steptypes, yeval, dyeval, dense_start] = result;

            // Build numpy arrays (copying vectors)
            py::gil_scoped_acquire gil;
            auto t_arr = vector_to_array_copy(xs);
            auto y_arr = vector_to_array_copy(ys);
            auto ydot_arr = vector_to_array_copy(dys);
            auto success_arr = vector_to_array_copy(successes);
            auto phase_arr = vector_to_array_copy(phases);
            auto steptype_arr = vector_to_array_copy(steptypes);

            auto yeval_arr = vector_to_array_copy(yeval);
            auto dyeval_arr = vector_to_array_copy(dyeval);

            return py::make_tuple(t_arr, y_arr, ydot_arr, success_arr,
                                  phase_arr, steptype_arr, yeval_arr,
                                  dyeval_arr);
        };

        return std::visit(run, solver_);
    }

private:
    SolverVariant solver_;
};

PYBIND11_MODULE(_riccati, m) {
    m.doc() = "Riccati solver bindings (pybind11)";

    py::class_<RiccatiSolver>(m, "RiccatiSolver")
        .def(py::init<py::function, py::function, int, int, int, int, bool>(),
             py::arg("omega_fun"), py::arg("gamma_fun"),
             py::arg("nini"), py::arg("nmax"), py::arg("n"), py::arg("p"),
             py::arg("vectorized") = true)
        .def("solve", &RiccatiSolver::solve,
             py::arg("xi"), py::arg("xf"), py::arg("yi"), py::arg("dyi"),
             py::arg("eps"), py::arg("epsilon_h"), py::arg("init_stepsize"),
             py::arg("t_eval") = py::none(), py::arg("hard_stop") = false);

    m.def("_dummy_riccati", [](py::object x) { return x; });

    m.def("_riccati_solve_default",
          [](double xi, double xf, std::complex<double> y0_0, std::complex<double> y0_1,
             py::function omega_fun, py::function gamma_fun,
             double eps, double epsilon_h, double init_stepsize,
             py::object t_eval, bool hard_stop,
             int nini, int nmax, int n, int p, bool vectorized) {
              RiccatiSolver solver(std::move(omega_fun), std::move(gamma_fun), nini, nmax, n, p, vectorized);
              return solver.solve(xi, xf, y0_0, y0_1, eps, epsilon_h, init_stepsize, t_eval, hard_stop);
          },
          py::arg("xi"), py::arg("xf"), py::arg("y0_0"), py::arg("y0_1"),
          py::arg("omega_fun"), py::arg("gamma_fun"),
          py::arg("eps"), py::arg("epsilon_h"), py::arg("init_stepsize"),
          py::arg("t_eval") = py::none(), py::arg("hard_stop") = false,
          py::arg("nini") = 16, py::arg("nmax") = 32, py::arg("n") = 32, py::arg("p") = 32,
          py::arg("vectorized") = true);

    m.def("choose_osc_stepsize",
          [](py::function omega_fun, py::function gamma_fun,
             double x0, double h, double epsilon_h,
             int nini, int nmax, int n, int p) {
              auto solver = make_solver_variant(std::move(omega_fun), std::move(gamma_fun),
                                                nini, nmax, n, p);
              auto run = [&](auto& s) -> double {
                  auto tup = riccati::choose_osc_stepsize(s, x0, h, epsilon_h);
                  return std::get<0>(tup);
              };
              return std::visit(run, solver);
          },
          py::arg("omega_fun"), py::arg("gamma_fun"),
          py::arg("x0"), py::arg("h"), py::arg("epsilon_h"),
          py::arg("nini") = 16, py::arg("nmax") = 32, py::arg("n") = 32, py::arg("p") = 32);

    m.def("choose_nonosc_stepsize",
          [](py::function omega_fun, py::function gamma_fun,
             double x0, double h, double epsilon_h,
             int nini, int nmax, int n, int p) {
              auto solver = make_solver_variant(std::move(omega_fun), std::move(gamma_fun),
                                                nini, nmax, n, p);
              auto run = [&](auto& s) -> double {
                  return riccati::choose_nonosc_stepsize(s, x0, h, epsilon_h);
              };
              return std::visit(run, solver);
          },
          py::arg("omega_fun"), py::arg("gamma_fun"),
          py::arg("x0"), py::arg("h"), py::arg("epsilon_h"),
          py::arg("nini") = 16, py::arg("nmax") = 32, py::arg("n") = 32, py::arg("p") = 32);
}

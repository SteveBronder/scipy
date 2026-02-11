cmake = import('cmake')
opt_var = cmake.subproject_options()

opt_var.add_cmake_defines(
  {'EIGEN_BUILD_DOC': false,
   'EIGEN_BUILD_BTL': false,
   'EIGEN_BUILD_BLAS': false,
   'EIGEN_BUILD_LAPACK': false,
   'EIGEN_BUILD_TESTING': false,
   'EIGEN_BUILD_PKGCONFIG': false,
   'EIGEN_LEAVE_TEST_IN_ALL_TARGET': false})

# Configure Eigen as a CMake subproject (this triggers wrap download if needed)
eigen_sp = cmake.subproject('eigen', options: opt_var)

message('CMake targets:\n - ' + '\n - '.join(eigen_sp.target_list()))
# Eigen’s exported target is usually Eigen3::Eigen
eigen3_dep = eigen_sp.dependency('eigen', include_type: 'system')

riccaticpp = subproject('riccaticpp')
riccaticpp_dep = riccaticpp.get_variable('riccaticpp_dep')

eigen_deps = [np_dep, pybind11_dep, eigen3_dep, riccaticpp_dep]

py3.extension_module(
  '_riccati',
  ['_riccati_pybind.cpp'],
  dependencies: eigen_deps,
  link_args: version_link_args,
  install: true,
  subdir: 'scipy/integrate/_ivp',
  cpp_args: ['-std=c++17'],
)

py3.install_sources([
    '__init__.py',
    '_osc.py',
    'base.py',
    'bdf.py',
    'common.py',
    'dop853_coefficients.py',
    'ivp.py',
    'lsoda.py',
    'radau.py',
    'rk.py'
  ],
  subdir: 'scipy/integrate/_ivp'
)

subdir('tests')

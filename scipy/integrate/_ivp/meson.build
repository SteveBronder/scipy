# Riccati pybind11 extension
riccati_inc = include_directories('../include')

eigen3_dep = dependency('eigen3', required: false, fallback: ['eigen', 'eigen3'])
if eigen3_dep.found()
  eigen_deps = [np_dep, pybind11_dep, eigen3_dep]
  eigen_incdirs = [riccati_inc]
else
  eigen_local = include_directories('../../../subprojects/eigen-3.4.0', is_system: true)
  eigen_deps = [np_dep, pybind11_dep]
  eigen_incdirs = [riccati_inc, eigen_local]
endif

py3.extension_module(
  '_riccati',
  ['_riccati_pybind.cpp'],
  dependencies: eigen_deps,
  include_directories: eigen_incdirs,
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

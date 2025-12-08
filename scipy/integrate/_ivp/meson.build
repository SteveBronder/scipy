# Get Eigen3 dependency
eigen3_dep = dependency('eigen3', required: true)

# Riccati C++ wrapper sources
riccati_wrapper_sources = [
  '../src/riccati_wrapper.cpp',
]

# Include directories for riccati headers
riccati_inc = include_directories('../../include')

py3.extension_module(
  '_riccati',
  [lib_cython_gen.process('_riccati.pyx'), riccati_wrapper_sources],
  dependencies: [np_dep, ccallback_dep, eigen3_dep],
  include_directories: riccati_inc,
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

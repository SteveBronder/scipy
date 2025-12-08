# Phase 2: Cython Extension Integration for _riccati

This memory documents the successful integration of the `_riccati` Cython extension module into SciPy's build system (Phase 2 of the riccati migration). Future phases will build on this foundation.

## Files Created in Phase 2

1. **scipy/integrate/_ivp/_riccati.pyx**
   - Location: Inside the `_ivp` subdirectory (not at top-level `scipy/integrate/`)
   - Contains: Dummy function `cpdef int _dummy_riccati(int x): return x`
   - Important: Keep pure Cython for Phase 2; no C++ headers, no Eigen yet
   - Module docstring documents phase and purpose

2. **scipy/integrate/_ivp/tests/test_riccati_dummy.py**
   - Test class: `TestRiccatiDummy` 
   - Tests: `test_import()` verifies module loads; `test_dummy_riccati_function()` checks behavior
   - Import pattern: `from scipy.integrate._ivp import _riccati`
   - Uses pytest class structure matching existing SciPy test patterns

## Build System Integration

### Extension Module Definition
- **Location**: `scipy/integrate/_ivp/meson.build` (NOT the top-level integrate/meson.build)
- **Pattern**: Added BEFORE `py3.install_sources()` block:
  ```meson
  py3.extension_module(
    '_riccati',
    [lib_cython_gen.process('_riccati.pyx')],
    dependencies: [np_dep],
    link_args: version_link_args,
    install: true,
    subdir: 'scipy/integrate/_ivp',
  )
  ```
- **Key details**:
  - Module name is just `'_riccati'` (not full path)
  - Use `lib_cython_gen.process()` for .pyx files (lib_cython_gen defined in scipy/_lib/meson.build)
  - `subdir: 'scipy/integrate/_ivp'` ensures install to correct location
  - `version_link_args` required for symbol visibility consistency
  - `np_dep` available from parent scope; add other deps (e.g., `eigen_dep`) in Phase 3

### Test File Installation
- **Location**: `scipy/integrate/_ivp/tests/meson.build`
- **Pattern**: Add to existing `py3.install_sources()` list:
  ```meson
  py3.install_sources([
      '__init__.py',
      'test_ivp.py',
      'test_riccati_dummy.py',  # <-- Added
      'test_rk.py'
    ],
    subdir: 'scipy/integrate/_ivp/tests',
    install_tag: 'tests'
  )
  ```
- **Critical**: Test files MUST be added to meson.build or pytest won't find them in build-install directory

## Python Integration

### Import Pattern in _osc.py
- **Location**: `scipy/integrate/_ivp/_osc.py`
- **Import line**: `from . import _riccati as _ric` (added after other imports)
- **Usage**: Called inside `solve_ivp_osc()` before `NotImplementedError`:
  ```python
  # Call dummy extension to verify it's callable
  _ = _ric._dummy_riccati(1)
  ```
- **Purpose**: Ensures extension loads during normal solve_ivp_osc execution; validates build

## Build and Test Commands

### Building
- Command: `pixi run build`
- Internally runs: `spin build` → meson compile → meson install
- Output: Extension installs to `build-install/usr/lib/python3.14/site-packages/scipy/integrate/_ivp/_riccati.cpython-314-darwin.so`
- Build time: ~5-10 seconds for incremental builds with just _riccati changes

### Testing
- Specific test: `pixi run test scipy/integrate/_ivp/tests/test_riccati_dummy.py -v`
- All _ivp tests: `pixi run test scipy/integrate/_ivp/ -v`
- Results from Phase 2: 67 passed (65 existing + 2 new riccati_dummy tests)
- Tests run from: `build-install/usr/lib/python3.14/site-packages` with PYTHONPATH set

### Direct Testing (for debugging)
```bash
cd /Users/sbronder/opensource/scipy/build-install/usr/lib/python3.14/site-packages
PYTHONPATH=/Users/sbronder/opensource/scipy/build-install/usr/lib/python3.14/site-packages \
  /Users/sbronder/opensource/scipy/.pixi/envs/test/bin/python3.14 -c \
  "from scipy.integrate._ivp import _riccati; print(_riccati._dummy_riccati(5))"
```

## Key Learnings for Phase 3

### Adding C++ and Eigen
1. **Include directories**: Will need to add riccati headers under `scipy/integrate/include/riccati/`
2. **Dependencies**: Add `eigen_dep` to extension module dependencies (already available in SciPy's meson)
3. **Language**: May need to specify `override_options: ['cython_language=cpp']` for C++ compilation
4. **Headers**: Can `cimport` C++ classes/functions via Cython's C++ support once riccati headers are in place

### Extension Module Naming
- Python import path: `scipy.integrate._ivp._riccati`
- Shared library name: `_riccati.cpython-314-darwin.so` (platform-specific)
- Module installed to: `scipy/integrate/_ivp/` subdirectory
- This matches SciPy's pattern for subpackage extensions (like other _ivp modules)

### Common Gotchas
1. **Forgetting test file in meson.build**: Tests exist but pytest can't find them
2. **Wrong subdir path**: Extension builds but installs to wrong location
3. **Missing version_link_args**: Extension builds but symbol visibility issues may arise
4. **Build cache**: Sometimes need to delete `build/` directory for clean rebuild after meson.build changes

### Testing Integration
- All existing _ivp tests must continue to pass
- New tests follow pytest + class structure (`TestClassName` with `test_method_name`)
- Import statements use relative imports from within _ivp package
- Tests run via pixi in isolated build-install environment (not source tree)

## Phase 3 Preparation Checklist
- [ ] Add riccati C++ headers to `scipy/integrate/include/riccati/`
- [ ] Update `_riccati.pyx` to use C++ and cimport riccati headers
- [ ] Add `eigen_dep` to extension module dependencies
- [ ] Implement actual riccati solver functions (replacing dummy)
- [ ] Port riccaticpp tests to use new `_riccati` module
- [ ] Wire up `solve_ivp_osc` to call real solver instead of raising NotImplementedError

## Status
Phase 2 complete and verified:
- ✅ Extension builds successfully
- ✅ Module imports correctly from Python
- ✅ Dummy function callable and returns expected values
- ✅ All existing tests pass (67 tests in _ivp)
- ✅ SOP checklist fully marked complete

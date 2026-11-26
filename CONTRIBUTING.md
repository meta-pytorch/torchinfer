# Contributing to torchinfer
We want to make contributing to this project as easy and transparent as
possible.
## Pull Requests
We actively welcome your pull requests.
1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes. From the repository root, create and activate
   a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   With NVIDIA:

   ```bash
   python3 -m pip install \
     --extra-index-url https://download.pytorch.org/whl/cu130 \
     '.[nvidia,test]'
   python3 -m pytest
   ```

   With AMD MI300X:

   ```bash
   python3 -m pip install \
     --extra-index-url https://download.pytorch.org/whl/rocm7.1 \
     '.[amd,test]'
   python3 -m pytest
   ```

   Pytest imports the checkout through the repository's `pythonpath` setting.
   pip installs the package non-editably by default, so rerun the matching
   `pip install` command after changing package code before running examples or
   otherwise validating the installed package.

   Edit only the CUDA `kernel.h` and `kernel.cpp` templates. Their `hip/`
   copies are generated; preserve backend-specific behavior with `USE_ROCM`.
   Regenerate and check them from the repository root:

   ```bash
   python -m aot_tensor.compile.triton.gen_hip_templates --write \
     --templates-dir aot_tensor/compile/triton/templates
   python -m aot_tensor.compile.triton.gen_hip_templates --check \
     --templates-dir aot_tensor/compile/triton/templates
   ```

   For reproducible standalone generation, use Torch 2.14.0 in a separate
   development environment; it does not change the AMD runtime dependency.
   The generator uses Torch's hipify engine. CI compares the generated copies
   byte-for-byte; review mapping changes when updating the generator engine.
5. Make sure your Python code lints. From the repository root, install
   [Lintrunner](https://github.com/pytorch/pytorch/wiki/lintrunner) and the
   adapter required by `.lintrunner.toml`:

   ```bash
   pip install lintrunner lintrunner-adapters
   lintrunner init
   ```

   Use a `pip` that installs into the same Python environment as `python` on
   your `PATH`; the Lintrunner configuration invokes `python` to run the adapter.
   Lint locally changed files, or check a specific Python file:

   ```bash
   lintrunner
   lintrunner aot_tensor/api/loading.py
   ```

   Format changed files with `lintrunner f`. Its `HIPTEMPLATES` formatter also
   regenerates the HIP copies after a CUDA template or generator edit:

   ```bash
   lintrunner format --take=HIPTEMPLATES aot_tensor/compile/triton/templates/kernel.h
   ```

   A warning about a missing `.lintrunner.private.toml` is harmless. Lintrunner
   checks Python formatting and generated HIP freshness; it does not run tests.
6. If you haven't already, complete the Contributor License Agreement ("CLA").
## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.
Complete your CLA here: <https://code.facebook.com/cla>
## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.
Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.
## License
By contributing to torchinfer, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.

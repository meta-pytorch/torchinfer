# Contributing to torchinfer
We want to make contributing to this project as easy and transparent as
possible.
## Pull Requests
We actively welcome your pull requests.
1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
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

   Format locally changed Python files with `lintrunner f`. To check every
   Python file, run `lintrunner --all-files`.

   A warning about a missing `.lintrunner.private.toml` is harmless. This
   configuration checks `.py` and `.pyi` files only; it does not format C++ or
   run tests.
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

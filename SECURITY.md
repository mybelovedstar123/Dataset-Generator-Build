# Security policy

## Supported versions

Only the latest commit on `main` is supported. Fixes are not backported.

## Reporting a vulnerability

Please **do not open a public issue**. Report it privately instead, through
[GitHub private vulnerability reporting](https://github.com/mybelovedstar123/Dataset-Generator-Build/security/advisories/new)
(**Security** tab, then **Report a vulnerability**).

Please include:

- what is affected (file and function, or the config key), and how;
- the steps or a minimal input to reproduce it;
- the impact you expect, such as code execution, unbounded memory or CPU use, or
  silent output corruption.

You can expect an acknowledgement within 7 days. After a fix is released, the
report will be disclosed through a GitHub security advisory.

## Scope

In scope: `generator.py`, the handling of `config.json`, and the pinned
dependencies in `requirements.txt`. Vulnerabilities in NumPy, SciPy, Numba or
scikit-learn themselves should be reported to those projects.

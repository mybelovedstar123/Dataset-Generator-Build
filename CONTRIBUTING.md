# Contributing

Thanks for taking the time to improve the generator. This file covers the rules
that keep its output reproducible, and how to check a change before opening a
pull request.

## Setting up

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Importing `generator.py` also needs the harness's `cascade` package, or a stub
module that provides `cascade.interface.DataGenerator`.

## Reproducibility rules

Training runs depend on a given seed producing the same rows every time. A change
must not alter rows that existing seeds already produce, unless that is the stated
purpose of the change.

- **Draw new randomness from a new stream.** Key it with a new tag or stage id in
  the `SeedSequence` key. Never take extra draws from an existing stream, because
  that shifts every draw after it.
- **Add behaviour as a layer.** New row sources wrap the existing `Generator` and
  replace only the rows their own stream selects, the way the partner, teach and
  fork layers do. Don't edit the layers underneath.
- **Gate new options in `config.json`.** A new option's default must reproduce the
  current behaviour. Validate it at construction, so a bad value fails with
  `ValueError` / `ConfigError` instead of failing mid-run.
- **Keep the output independent of thread count and completion order.** Workers
  write only their own slots; batches are emitted by index.

If a change does alter existing output on purpose, say so in the pull request
description and explain why.

## Checks

CI runs these on every pull request. Run them locally first:

```bash
python -m py_compile generator.py
pip install ruff==0.13.1
ruff check --no-cache --isolated --select E9,F63,F7 generator.py
for f in *.json; do python -m json.tool "$f" > /dev/null && echo "ok  $f"; done
python .github/scripts/check_requirements.py requirements.txt
pip install --dry-run --require-hashes -r requirements.txt
```

## Commits and pull requests

- One logical change per commit: one family, one fix, or one config option.
- Use the imperative mood in the subject line, at most 72 characters. Use the body
  to say *why*, and what it affects.
- Keep pull requests focused, and fill in the template, including the
  reproducibility checklist.
- To report a security issue, follow [SECURITY.md](SECURITY.md) instead of opening
  an issue.

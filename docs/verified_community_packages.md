# Verified Community Packages

Verified community packages are independently maintained integrations that the
Trained by Humans (TBH) organization recognizes and publishes through its
release controls. They extend the `ml-pipes` ecosystem without becoming part
of the shared framework or its default installation.

This policy applies to packages such as
[`ml-pipes-supervision`](https://github.com/trained-by-humans/ml-pipes-supervision)
and
[`ml-pipes-ultralytics`](https://github.com/trained-by-humans/ml-pipes-ultralytics).

## Scope and ownership

A verified community package:

- remains in its own repository and is published as its own PyPI distribution;
- owns a focused integration boundary and its own public `ml_pipes.<module>`
  surface;
- is opt-in: it must not be added to the umbrella package's default or `all`
  install profile; and
- may carry heavy, proprietary, or differently licensed runtimes only when
  their implications are clearly documented.

The community maintainers own day-to-day code, tests, documentation, and
upstream compatibility. TBH organization administrators own publication
authority, the trusted-publisher configuration, protected release paths, and
verified status.

## Support contract

Each package must document:

- supported Python versions and the tested `ml-pipes` framework versions;
- supported upstream dependency ranges and any known compatibility limits;
- optional runtime or licensing requirements; and
- the package's public import surface and installation instructions.

The minimum supported Python range for a verified package is Python 3.10
through 3.13 unless its package documentation declares a narrower, tested
range. Release candidates may exactly pin their tested framework candidate.
After a stable framework release, packages should use tested bounded ranges
rather than permanent exact framework pins.

## CI and distribution minimums

Every verified package must have CI that, at a minimum:

1. runs its relevant test suite on Python 3.10, 3.11, 3.12, and 3.13;
2. builds a wheel and source distribution;
3. validates distributions with `twine check`; and
4. installs the built wheel into a clean environment, runs `pip check`, and
   imports the package's public module.

The source distribution must contain only the source and metadata required to
build and install the package. Large examples, model weights, generated sites,
and other repository assets stay out of release artifacts unless they are
runtime requirements.

## Security and release authority

Verified packages must enable GitHub Private Vulnerability Reporting and
provide security-reporting instructions equivalent to the framework's
[Security Policy](https://github.com/trained-by-humans/ml-pipes/blob/main/SECURITY.md). Sensitive reports must not be directed to
public issues.

Only TBH organization administrators may:

- approve release-preparation changes;
- create or move release tags;
- change `pyproject.toml` release metadata or `.github/workflows/` release
  automation; and
- approve production PyPI deployments.

Repository rules must require that review path for release files and protect
`v*` release tags. Community contributors may continue to contribute through
normal pull requests without receiving PyPI credentials or publication
authority.

Packages use PyPI Trusted Publishing with GitHub Actions OIDC. Long-lived PyPI
API tokens are not permitted. Each package's `RELEASING.md` records its exact
TestPyPI and PyPI workflow and environment bindings.

## Release and deprecation

Each package release follows its package-specific release contract. Community
maintainers prepare releases; TBH release maintainers authorize publication
and retain control of the verified distribution path. Package-specific release
instructions live in the package repository; for example:

- [`ml-pipes-supervision` release instructions](https://github.com/trained-by-humans/ml-pipes-supervision/blob/main/RELEASING.md)
- [`ml-pipes-ultralytics` release instructions](https://github.com/trained-by-humans/ml-pipes-ultralytics/blob/main/RELEASING.md)

To deprecate a verified package, TBH and the community maintainers must publish
a release notice with the reason, migration guidance, the final supported
version, and an end-of-support date. The package documentation and PyPI
metadata must point users to that notice. Repository archival happens only
after the announced support window and security-reporting path are addressed.

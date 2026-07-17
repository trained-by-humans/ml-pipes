# Security Policy

## Supported Releases

Until the first public release, report security issues against the current
`main` branch or the latest release candidate branch that reproduces the
problem.

## Reporting A Vulnerability

Report suspected vulnerabilities through GitHub Private Vulnerability
Reporting. Do not open a public GitHub issue for a sensitive report.

Include the affected package name, the version or commit you tested, a minimal
reproduction, and any required environment details.

## In Scope

- vulnerabilities in `ml-pipes` package code, CLI behavior, validation,
  tracing, inspection, benchmarking, and release tooling
- packaging or workflow mistakes that could compromise published
  distributions
- unsafe default behavior shipped by the library itself

## Out Of Scope

- application code built on top of `ml-pipes`
- endpoint authentication, authorization, or rate limiting in user services
- infrastructure, secrets, or deployment configuration owned by downstream
  applications
- model quality issues, prompt quality, or third-party service outages unless
  `ml-pipes` introduces the vulnerability directly

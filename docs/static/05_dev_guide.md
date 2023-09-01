# Contribution Guide

## Introduction

The Macrosynergy package is an open-source project. We welcome contributions of all kinds,
from simple bug reports through to new features and documentation improvements.
This document outlines the process for contributing to the project.


## Raising an Issue

If you have found a bug or have a question about the package, please raise an
issue on the [GitHub issue tracker](https://github.com/macrosynergy/macrosynergy/issues/new/choose).

If you are reporting a bug, please include as much information as possible to help us
reproduce the issue. This includes environment details (e.g. operating system, Python version,
dependency versions (`pip freeze`)) error messages, and a minimal reproducible example.

## Contributing Code

Contributing code to the project is done through pull requests of the nature described below.

### New Features

A new feature is a substantial change to the package that adds new functionality.
New features can be discussed in a [Feature Request issue](https://github.com/macrosynergy/macrosynergy/issues/new?assignees=&labels=&projects=&template=feature_request.md&title=)
on the issues tracker.

### Bug Fixes

A bug fix is a change to the package that fixes a bug. The reasons and details of the bug
must be documented as a GitHub issue to provide context for the fix, and to create a record
of the bug and its resolution.

### Hotfixes

Hotfixes or emergency fixes are bug fixes that are required to fix a critical bug in a
released version of the package. Currently, hotfixes are not supported for community contributions and are limited to the package maintainers.

### Documentation Improvements

Documentation improvements are changes to the package documentation that improve the
clarity or completeness of the documentation.

## Pull Requests

## Pull Request Process

All pull requests (PR) must be made against the `develop` branch of the repository. All PRs require
at least one review from a maintainer before they can be merged.
The pull requests require status checks to pass before they can be merged.

## Status Checks

The following status checks are run on all pull requests:

- All unit tests must pass
- All integration tests must pass
- Code coverage must be at least 50%. For community contributions, this is checked by the maintainer.

## Title Conventions

Titles for pull requests should be of the form:

**\<Type>: Clear description with `code_formatted` correctly**

Where `Type` is one of:

- `Feature`: For new features
- `Bugfix`: For bug fixes
- `Hotfix`: For hotfixes
- `Docs`: For documentation improvements
- `Chore`: For changes to the build process, dependencies, CI, etc.

### Commit Conventions

Commits should be kept small and focused on a single change. The commit message should
describe the change in a clear and concise manner.

## Community Contributions

The following apply to contributors outside the Macrosynergy team:

- All contributions must be made under the terms of the [project license](https://github.com/macrosynergy/macrosynergy/blob/main/LICENSE) on the package repository.
- All contributions must be made through pull requests.
- Contributors may only contribute code that they have authored or have the rights to contribute.
- Pull requests can only be raised from a fork of the repository.
- All pull requests must be made against the `develop` branch of the repository.

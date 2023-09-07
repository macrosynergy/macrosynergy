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

## Feature Requests

A new feature is a substantial change to the package that adds new functionality.
New features can be discussed in a [Feature Request issue](https://github.com/macrosynergy/macrosynergy/issues/new?assignees=&labels=&projects=&template=feature_request.md&title=)
on the issues tracker.

## CI/CD Pipeline

The package has the 3 primary branches:

- `develop`: The development branch; used for development and testing of new features
- `test`: The test branch; used for testing and release candidates
- `main`: The main branch; used for stable releases

```
develop ← feature/<feature_name>
        ← bugfix/<bugfix_name>
        
test ← develop

main ← hotfix/<hotfix_name>
     ← test  

...
develop ← version++ ← main
```


### New Features

Adding a new feature or enhancement requires a new feature branch to be created
from the `develop` branch. The feature branch should be named

```
develop ← feature/<feature_name>
```

Once the feature is complete, a pull request should be raised against the `develop`
branch. See the [Pull Requests Section](#pull-requests) for more details.

### Bugfixes

Bugfixes also be made against the `develop` branch.

```
develop ← bugfix/<bugfix_name>
```

### Hotfixes

Hotfixes are reserved for critical bugfixes that need to be deployed immediately.
These are typically reserved for security issues, build process bugs, or issues
with business-critical functionality. Hotfixes should be made against the `main` branch. Once PR is merged, the `main` branch should be merged into the `develop` branch; and a new release should be created from the `main` branch.

**The hotfix must also contain a version increment to allow for the new release.**
```
main ← hotfix/<hotfix_name>
```


### CI/CD

For changes related to build processes, CI/CD, or other maintenance tasks:
```
chores/<chore_name>
```
Make sure to use "Chore" as the type of the pull request. (See [Pull Requests/Title Conventions](#title-conventions))
## Pull Requests

### Pull Request Process

All pull requests (PR) must be made against the `develop` branch of the repository. All PRs require
at least one review from a maintainer before they can be merged.
The pull requests require status checks to pass before they can be merged.

### Status Checks

The following status checks are run on all pull requests:

- All unit tests must pass
- All integration tests must pass
- Code coverage must be at least 50%. For community contributions, this is checked by the maintainer.

### Title Conventions

Titles for pull requests should be of the form:

**\<Type>: Clear description with `code_formatted` correctly**

Where **\<Type>** is one of:

- `Feature`: For new features
- `Bugfix`: For bug fixes
- `Hotfix`: For hotfixes
- `Chore`: Reserved for CI/CD and other maintenance tasks



## Community Contributions

For community contributions, the pull requests are reviewed by the package maintainers.
  

The following apply to contributors outside the Macrosynergy team:

1. All contributions must be made under the terms of the [project license](https://github.com/macrosynergy/macrosynergy/blob/main/LICENSE) on the package repository.
1. All contributions must be made through pull requests.
1. Contributors may only contribute code that they have authored or have the rights to contribute.
1. Pull requests can only be raised from a fork of the repository.
1. Contributors may not make any modifications to the unit tests, integration tests, dependencies, or any CI/CD configuration (e.g. GitHub Actions, Codecov, etc.)

1. Contributors may not modify any static files such as
the static documentation pages, the `README.md` file, and the `LICENSE` file.



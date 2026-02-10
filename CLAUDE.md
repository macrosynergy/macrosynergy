# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Macrosynergy is a Python package for quantamental (quantitative + fundamental) financial market research. It provides tools for analyzing JPMaQS (J.P. Morgan Macrosynergy Quantamental System) data and developing trading strategies.

## Core Data Structure

The package centers around **QuantamentalDataFrame**, a pandas DataFrame subclass with strict requirements:

- **Required columns**: `real_date` (datetime64), `cid` (cross-section identifier), `xcat` (extended category)
- **Optional columns**: `value`, `grading`, `eop_lag`, `mop_lag`, `last_updated`
- **cid**: Represents countries/regions as 3-letter codes (e.g., "USD", "EUR", "GBP")
- **xcat**: Extended categories representing data types (e.g., "FXXR_NSA" for FX returns, "EQXR_NSA" for equity returns)
- **ticker**: Combination of cid and xcat, formatted as "{cid}_{xcat}"

The QuantamentalDataFrame class is defined in `macrosynergy/management/types/qdf/` and uses categorical dtypes for performance optimization.

## Package Architecture

1. **download**: DataQuery API interface for downloading JPMaQS data (requires OAuth or certificate authentication)
2. **management**: Core utilities including data validation, simulation, and type definitions
   - `management.simulate`: Generate synthetic quantamental data for testing
   - `management.types`: Type definitions including QuantamentalDataFrame
   - `management.utils`: DataFrame operations and mathematical utilities
   - `management.validation`: Data validation functions
3. **panel**: Panel data calculations (historic volatility, z-scores, beta values, baskets)
4. **pnl**: Portfolio construction, risk management, and realistic PnL analysis
5. **signal**: Transform indicators to trading signals and analyze signal-return relationships
6. **learning**: Machine learning techniques for quantamental data
7. **visuals**: Visualization tools for quantamental data and analyses

## Common Development Commands

### Testing
```bash
# Run all unit tests (parallel execution with pytest-xdist)
pytest

# Run tests for a specific subpackage
pytest ./tests/unit/panel
pytest ./tests/unit/management
pytest ./tests/unit/download

# Run with coverage reporting
pytest --cov=macrosynergy --verbose

# Run tests with specific number of workers
pytest -n 4
```

### Linting
```bash
# Lint with flake8 (show critical errors only)
flake8 --count --select=E9,F63,F7,F82 --show-source --exclude=./docs/**,./.github/scripts/*,./build/** --statistics

# Full lint with complexity and line length checks
flake8 --count --exit-zero --max-complexity=10 --max-line-length=127 --exclude=./docs/**,./.github/scripts/*,./build/** --statistics
```

### Code Formatting
```bash
# Format code with black
black macrosynergy/
black tests/
```

### Documentation
```bash
# Build documentation
cd docs
python gen.py  # Generate API documentation
make html      # Build HTML documentation
```

### Installation
```bash
# Install package in development mode
pip install -e .

# Install with test dependencies
pip install -e ".[test]"

# Install all optional dependencies
pip install -e ".[all]"
```

## Git Workflow

- **Main development branch**: `develop` (not `main`)
- **Pull requests**: Create PRs against the `develop` branch
- **PR requirements**: PR titles must comply with quality control standards (checked by `.github/scripts/check_pr.py`)

## Key Constants and Conventions

Defined in `macrosynergy/management/constants.py`:

- **Frequency mappings**: D (daily), W (weekly), M (monthly), Q (quarterly), A (annual)
- **Annualization factors**: Used for converting metrics to annual basis (252 for daily, 52 for weekly, etc.)
- **Country groups**:
  - `cids_dmca`: DM currency areas (AUD, CAD, CHF, EUR, GBP, JPY, NOK, NZD, SEK, USD)
  - `cids_dm`: All developed markets
  - `cids_em`: All emerging markets (Latam, EMEA, EM Asia)

## Testing Structure

Tests are organized by subpackage in `tests/unit/`:
- `tests/unit/download/`: Download module tests
- `tests/unit/learning/`: Learning module tests
- `tests/unit/management/`: Management utilities tests
- `tests/unit/panel/`: Panel analysis tests
- `tests/unit/pnl/`: PnL calculation tests
- `tests/unit/signal/`: Signal analysis tests
- `tests/unit/visual/`: Visualization tests

The test suite uses:
- `pytest` with `pytest-xdist` for parallel execution
- `pytest-cov` for coverage reporting
- `parameterized` for parameterized tests
- Data simulation utilities from `tests/simulate.py`

## Code Patterns

### Creating Quantamental DataFrames

For testing or examples, use the simulation utilities:
```python
from macrosynergy.management.simulate import make_qdf

# Define cross-sections and categories
cids = ['AUD', 'GBP', 'USD']
xcats = ['FXXR_NSA', 'FXCRY_NSA']

# Simulate data
df = make_qdf(df_cids, df_xcats, back_ar=0.75)
```

### Working with QuantamentalDataFrame

```python
from macrosynergy.management.types import QuantamentalDataFrame

# Convert pandas DataFrame to QuantamentalDataFrame
qdf = QuantamentalDataFrame(df)

# QuantamentalDataFrames use categorical dtypes by default for performance
qdf.is_categorical()  # Returns True

# Convert between categorical and string types
qdf.to_string_type()  # Convert to string
qdf.to_categorical()  # Convert to categorical
```

## Important Notes

- Python 3.8+ required (Python 3.7 support deprecated)
- The package supports pandas 1.3.5 to <3.0.0
- Uses versioneer for version management (version defined in `setup.py`)
- Documentation follows "code-as-documentation" principle using Sphinx autodoc
- DataQuery API access requires either OAuth credentials or certificate/key pair

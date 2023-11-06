# `macrosynergy.learning`

## Description

The `macrosynergy.learning` subpackage contains functions and classes to assist the 
creation of machine learning solutions with macro quantamental data. 

Currently, the functionality is built around integrating the `macrosynergy` package and 
associated JPMaQS data with the popular `scikit-learn` library, which provides a simple
interface for fitting common statistical machine learning models, as well as feature
selection methods, cross-validation classes and performance metrics.

The panel format of our quantamental dataframes renders many of the common classes and 
functions in scikit-learn either impossible or difficult to use, either conceptually or
practically. We provide custom scikit-learn estimators, transformers, metrics, wrappers
and classes to build a bridge between tradable economics and machine learning. 
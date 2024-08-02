import sys
print("Testing the installation of the package...")
print(f"Python: {sys.version}")

import numpy as np
print(f"Numpy: {np.__version__}")
print(f"Numpy: {np.__path__}")

import pandas as pd
print(f"Pandas: {pd.__path__}")
import matplotlib.pyplot as plt
import seaborn as sns


# print(f"Sklearn: {sklearn.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Seaborn: {sns.__version__}")

import sys
import traceback

def import_package(package_name):
    try:
        __import__(package_name)
        print(f"Successfully imported {package_name}")
    except ImportError as e:
        print(f"Failed to import {package_name}: {e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=10, file=sys.stdout)
        find_issue_modules(package_name)

def find_issue_modules(package_name):
    # Attempt to import submodules one by one to identify where the error occurs
    package = sys.modules.get(package_name)
    if package is None:
        print(f"Package {package_name} is not available for inspection.")
        return

    for submodule in dir(package):
        full_submodule = f"{package_name}.{submodule}"
        try:
            __import__(full_submodule)
            print(f"Successfully imported {full_submodule}")
        except ImportError as e:
            print(f"Failed to import {full_submodule}: {e}")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback, limit=10, file=sys.stdout)


# Example usage
import_package('macrosynergy')
import macrosynergy
print(f"Macrosynergy: {macrosynergy.__version__}")
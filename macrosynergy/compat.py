from . import PYTHON_3_8_OR_LATER

if PYTHON_3_8_OR_LATER:
    RESAMPLE_NUMERIC_ONLY = {"numeric_only": True}
else:
    RESAMPLE_NUMERIC_ONLY = {}

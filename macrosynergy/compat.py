from . import PYTHON_3_8_OR_LATER

if PYTHON_3_8_OR_LATER:
    RESAMPLE_NUMERIC_ONLY = {"numeric_only": True}
    JOBLIB_RETURN_AS = {"return_as": "generator"}
    from sklearn.base import OneToOneFeatureMixin
else:
    RESAMPLE_NUMERIC_ONLY = {}
    JOBLIB_RETURN_AS = {}
    from sklearn.base import _OneToOneFeatureMixin as OneToOneFeatureMixin

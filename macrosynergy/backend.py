from types import ModuleType

CURRENT_BACKEND = "pandas"
import warnings


def get_current_backend() -> ModuleType:
    global CURRENT_BACKEND
    return CURRENT_BACKEND


def set_backend(backend_name: str = "pandas") -> None:
    global CURRENT_BACKEND
    if backend_name in [
        "pandas",
        "pd",
    ]:
        CURRENT_BACKEND = "pandas"
        return

    elif backend_name in [
        "modin.pandas",
        "modin",
        "modin.pd",
        "mpd",
    ]:
        CURRENT_BACKEND = "modin.pandas"
        return

    raise ValueError(
        f"Backend '{backend_name}' not supported. Please use 'pandas' or 'modin.pandas'."
    )

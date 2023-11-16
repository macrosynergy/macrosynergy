"""Macrosynergy Quant Research Package"""

try:
    from .version import git_revision as __git_revision__
    from .version import version as __version__
except ModuleNotFoundError:
    from setup import get_version_info

    FULLVERSION, GIT_REVISION = get_version_info()
    __version__ = FULLVERSION
    __git_revision__ = GIT_REVISION

from . import visuals, download, panel, pnl, management, signal, learning

__all__ = ["visuals", "download", "panel", "pnl", "management", "signal", "learning"]

__name__ = ["__version__"]


# allows the package version information to be accessed from the package

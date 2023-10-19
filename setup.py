"""Macrosynergy Quant Research Package"""
import os
import sys
import subprocess
import warnings


if sys.version_info[:2] < (3, 8):
    raise RuntimeError("Python version >= 3.8 required.")

MAJOR = 0
MINOR = 0
MICRO = 44
ISRELEASED = True
VERSION = "%d.%d.%d" % (MAJOR, MINOR, MICRO)

if sys.version_info >= (3, 12):
    # The first version not in the `Programming Language :: Python :: ...` classifiers above
    warnings.warn(
        f"Macrosynergy {VERSION} may not yet support Python "
        f"{sys.version_info.major}.{sys.version_info.minor}.",
        RuntimeWarning,
    )

if sys.version_info < (3, 8):
    warnings.warn(
        f"Python {sys.version_info.major}.{sys.version_info.minor} "
        "has reached end-of-life. The Macrosynergy package no longer supports this version. "
        "Please upgrade to Python 3.8 or later.",
        RuntimeWarning,
    )


# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ["SYSTEMROOT", "PATH", "HOME"]:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env["LANGUAGE"] = "C"
        env["LANG"] = "C"
        env["LC_ALL"] = "C"
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, env=env)
        return out

    try:
        out = _minimal_ext_cmd(["git", "rev-parse", "HEAD"])
        GIT_REVISION = out.strip().decode("ascii")
    except (subprocess.SubprocessError, OSError):
        GIT_REVISION = "Unknown"

    if not GIT_REVISION:
        # this shouldn't happen but apparently can (see gh-8512)
        GIT_REVISION = "Unknown"

    return GIT_REVISION


def get_version_info():
    # Adding the git rev number needs to be done inside write_version_py(),
    # otherwise the import of qstools.version messes up the build under Python 3.
    FULLVERSION = VERSION
    if os.path.exists(".git"):
        GIT_REVISION = git_version()
    elif os.path.exists("macrosynergy/version.py"):
        # must be a source distribution, use existing version file
        try:
            from macrosynergy.version import git_revision as GIT_REVISION
        except ImportError:
            raise ImportError(
                "Unable to import git_revision. Try removing "
                "macrosynergy/version.py and the build directory "
                "before building."
            )
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        import time

        time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
        FULLVERSION += f".dev0+{time_stamp}_{GIT_REVISION[:7]}"

    return FULLVERSION, GIT_REVISION


def write_version_py(filename="macrosynergy/version.py"):
    cnt = """\
# THIS FILE IS GENERATED FROM macrosynergy SETUP.PY
short_version: str = '%(version)s'
version: str = '%(version)s'
full_version: str = '%(full_version)s'
git_revision: str = '%(git_revision)s'
release: bool = %(isrelease)s
if not release:
    version = full_version
"""
    FULLVERSION, GIT_REVISION = get_version_info()

    a = open(filename, "w")
    try:
        a.write(
            cnt
            % {
                "version": VERSION,
                "full_version": FULLVERSION,
                "git_revision": GIT_REVISION,
                "isrelease": str(ISRELEASED),
            }
        )
    finally:
        a.close()


def setup_package():
    from setuptools import setup

    # Rewrite the version file every time
    write_version_py()

    metadata = dict(
        version=get_version_info()[0],
    )

    setup(**metadata)


if __name__ == "__main__":
    setup_package()

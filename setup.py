"""Macrosynergy Quant Research Package"""
import os
import sys
import subprocess
import warnings
from pathlib import Path

DOCLINES = (__doc__ or '').split("\n")

path = Path(__file__).parent
with open(os.path.join(path, "README.md"), "r") as f:
    readme = f.read()

if sys.version_info[:2] < (3, 6):
    raise RuntimeError("Python version >= 3.6 required.")

CLASSIFIERS = """\
Intended Audience :: Science/Research
Intended Audience :: Developers
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3 :: Only
Topic :: Software Development
Topic :: Scientific/Engineering
Typing :: Typed
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: MacOS
Development Status :: 4 - Beta
"""

MAJOR = 0
MINOR = 0
MICRO = 20
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

if sys.version_info >= (3, 10):
    # The first version not in the `Programming Language :: Python :: ...` classifiers above
    warnings.warn(
        f"Macrosynergy {VERSION} may not yet support Python "
        f"{sys.version_info.major}.{sys.version_info.minor}.",
        RuntimeWarning,
    )


# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, env=env)
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
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
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists('macrosynergy/version.py'):
        # must be a source distribution, use existing version file
        try:
            from qstools.version import git_revision as GIT_REVISION
        except ImportError:
            raise ImportError("Unable to import git_revision. Try removing "
                              "qstools/version.py and the build directory "
                              "before building.")
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        import time

        time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
        FULLVERSION += f'.dev0+{time_stamp}_{GIT_REVISION[:7]}'

    return FULLVERSION, GIT_REVISION


def write_version_py(filename='macrosynergy/version.py'):
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

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'git_revision': GIT_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()


with open(os.path.join(os.path.dirname(__file__), "requirements.txt")) as f:
    REQUIREMENTS = f.read()


def setup_package():
    from setuptools import setup, find_packages

    src_path = os.path.dirname(os.path.abspath(__file__))
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    # Rewrite the version file every time
    write_version_py()

    metadata = dict(
        name='macrosynergy',
        maintainer="Macrosynergy",
        maintainer_email="info@macrosynergy.com",
        description=DOCLINES[0],
        license="MIT License",
        long_description_content_type="text/markdown",
        long_description=readme,
        url="https://www.macrosynergy.com",
        author_email='info@macrosynergy.com',
        author='Macrosynergy Ltd',
        download_url="https://github.com/macrosynergy/macrosynergy",
        project_urls={
            "Bug Tracker": "https://github.com/macrosynergy/macrosynergy/issues",
            "Source Code": "https://github.com/macrosynergy/macrosynergy",
            "Documentation": "https://docs.macrosynergy.com"
        },
        classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
        platforms=["Windows", "Linux", "Mac OS-X"],
        test_suite='pytest',
        python_requires='>=3.6',
        install_requires=REQUIREMENTS.split("\n"),
        include_package_data=True,
        packages=find_packages(),
        version=get_version_info()[0],
    )
    # __copyright__ = 'Copyright 2020 Macrosynergy Ltd'

    try:
        setup(**metadata)
    finally:
        del sys.path[0]
        os.chdir(old_path)
    return


if __name__ == "__main__":
    setup_package()

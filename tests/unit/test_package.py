import subprocess
import sys
import unittest


class TestPackage(unittest.TestCase):
    def test_lazy_imports(self):
        # Check that when qstools imported correctly
        cmd = ("import macrosynergy as msy; "
               "import sys; ")

        cmd = sys.executable + ' -c "' + cmd + '"'
        p = subprocess.Popen(cmd, shell=True, close_fds=True)
        p.wait()
        rc = p.returncode
        assert rc == 0

    def test_docstring_optimization_compat(self):
        # Check that importing with stripped docstrings does not raise
        cmd = sys.executable + ' -OO -c "import macrosynergy as qs"'
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        out = p.communicate()
        rc = p.returncode
        assert rc == 0, out


if __name__ == '__main__':
    unittest.main()

import hashlib
from tests.perf.env import environment_fingerprint, fingerprint_hash, comparable


def test_fingerprint_has_required_keys():
    fp = environment_fingerprint()
    for key in [
        "cpu_brand", "cpu_arch", "cpu_count_logical", "os_system",
        "os_release", "python_version", "lib_versions", "hostname", "timestamp",
    ]:
        assert key in fp, f"missing key: {key}"
    assert {"numpy", "pandas", "statsmodels", "scipy", "pyarrow", "macrosynergy"} <= set(
        fp["lib_versions"]
    )


def test_fingerprint_hash_is_stable_and_short():
    fp = environment_fingerprint()
    h1 = fingerprint_hash(fp)
    h2 = fingerprint_hash(environment_fingerprint())
    assert h1 == h2  # same machine -> same hash
    assert len(h1) == 8 and all(c in "0123456789abcdef" for c in h1)


def test_comparable_true_for_same_machine():
    assert comparable(environment_fingerprint(), environment_fingerprint())


def test_comparable_false_when_cpu_differs():
    a = environment_fingerprint()
    b = dict(a)
    b["cpu_brand"] = a["cpu_brand"] + " (other)"
    assert not comparable(a, b)

def test_perf_env_fixture(perf_env):
    assert "cpu_brand" in perf_env
    assert "lib_versions" in perf_env

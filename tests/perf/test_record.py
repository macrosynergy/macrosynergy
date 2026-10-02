from tests.perf.record import render_report, extract_env


def _bench(env, name="bench_x", mean=1.0):
    return {
        "machine_info": {"macrosynergy_env": env},
        "benchmarks": [{"name": name, "stats": {"mean": mean, "min": mean, "max": mean}}],
    }


SAME = {"cpu_brand": "TestCPU", "cpu_count_logical": 8, "cpu_arch": "x86_64", "os_system": "Linux"}
OTHER = {**SAME, "cpu_brand": "OtherCPU"}


def test_extract_env():
    assert extract_env(_bench(SAME)) == SAME


def test_same_machine_shows_verdict():
    report = render_report(_bench(SAME, mean=2.0), _bench(SAME, mean=1.0))
    assert "advisory only" not in report
    assert "50%" in report  # ~50% faster shown in the verdict column


def test_cross_machine_shows_banner():
    report = render_report(_bench(SAME, mean=2.0), _bench(OTHER, mean=1.0))
    assert "advisory only" in report

import pickle
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from macrosynergy.download.extra import load_data, save_data


class TestSaveData(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_dataframe_saved_as_parquet(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        out = save_data(self.tmpdir / "data.csv", df)
        self.assertEqual(out.suffix, ".parquet")
        self.assertTrue(out.exists())

    def test_dict_saved_as_json(self):
        data = {"key": "value", "num": 42}
        out = save_data(self.tmpdir / "data.pkl", data)
        self.assertEqual(out.suffix, ".json")
        self.assertTrue(out.exists())

    def test_list_saved_as_json(self):
        data = [1, 2, "three"]
        out = save_data(self.tmpdir / "data", data)
        self.assertEqual(out.suffix, ".json")
        self.assertTrue(out.exists())

    def test_other_object_saved_as_pickle(self):
        # sets are not dict or list, but are picklable
        obj = {1, 2, 3}
        out = save_data(self.tmpdir / "obj.json", obj)
        self.assertEqual(out.suffix, ".pkl")
        self.assertTrue(out.exists())

    def test_extension_overridden(self):
        df = pd.DataFrame({"x": [1]})
        out = save_data(self.tmpdir / "file.txt", df)
        self.assertEqual(out.suffix, ".parquet")
        self.assertFalse((self.tmpdir / "file.txt").exists())

    def test_returns_path_object(self):
        out = save_data(str(self.tmpdir / "data"), {"a": 1})
        self.assertIsInstance(out, Path)

    def test_missing_parent_raises(self):
        with self.assertRaises(FileNotFoundError):
            save_data(self.tmpdir / "nonexistent_dir" / "data", {"a": 1})


class TestLoadData(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_load_parquet(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        path = self.tmpdir / "data.parquet"
        df.to_parquet(path)
        result = load_data(path)
        self.assertIsInstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, df)

    def test_load_json_dict(self):
        import json

        data = {"key": "value", "num": 99}
        path = self.tmpdir / "data.json"
        path.write_text(json.dumps(data))
        result = load_data(path)
        self.assertEqual(result, data)

    def test_load_json_list(self):
        import json

        data = [1, 2, "three"]
        path = self.tmpdir / "data.json"
        path.write_text(json.dumps(data))
        result = load_data(path)
        self.assertEqual(result, data)

    def test_load_pickle(self):
        data = {"nested": [1, 2, 3]}
        path = self.tmpdir / "data.pkl"
        with open(path, "wb") as f:
            pickle.dump(data, f)
        result = load_data(path)
        self.assertEqual(result, data)

    def test_probe_parquet_when_no_extension(self):
        df = pd.DataFrame({"x": [10, 20]})
        df.to_parquet(self.tmpdir / "data.parquet")
        result = load_data(self.tmpdir / "data")
        self.assertIsInstance(result, pd.DataFrame)

    def test_probe_json_when_no_parquet(self):
        import json

        data = {"a": 1}
        (self.tmpdir / "data.json").write_text(json.dumps(data))
        result = load_data(self.tmpdir / "data")
        self.assertEqual(result, data)

    def test_probe_pickle_when_no_parquet_or_json(self):
        data = [42, 43]
        with open(self.tmpdir / "data.pkl", "wb") as f:
            pickle.dump(data, f)
        result = load_data(self.tmpdir / "data")
        self.assertEqual(result, data)

    def test_probe_prefers_parquet_over_json(self):
        import json

        df = pd.DataFrame({"x": [1]})
        df.to_parquet(self.tmpdir / "data.parquet")
        (self.tmpdir / "data.json").write_text(json.dumps({"x": 1}))
        result = load_data(self.tmpdir / "data")
        self.assertIsInstance(result, pd.DataFrame)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_data(self.tmpdir / "no_such_file.parquet")

    def test_missing_file_no_extension_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_data(self.tmpdir / "no_such_file")

    def test_accepts_string_path(self):
        df = pd.DataFrame({"z": [7]})
        path = self.tmpdir / "data.parquet"
        df.to_parquet(path)
        result = load_data(str(path))
        self.assertIsInstance(result, pd.DataFrame)


class TestSaveLoadRoundtrip(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_roundtrip_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        out = save_data(self.tmpdir / "df", df)
        result = load_data(out)
        pd.testing.assert_frame_equal(result, df)

    def test_roundtrip_dict(self):
        data = {"hello": "world", "count": 7}
        out = save_data(self.tmpdir / "d", data)
        self.assertEqual(load_data(out), data)

    def test_roundtrip_list(self):
        data = [1, "two", 3.0]
        out = save_data(self.tmpdir / "lst", data)
        self.assertEqual(load_data(out), data)

    def test_roundtrip_without_explicit_extension(self):
        df = pd.DataFrame({"v": [99]})
        save_data(self.tmpdir / "myfile", df)
        result = load_data(self.tmpdir / "myfile")
        pd.testing.assert_frame_equal(result, df)


if __name__ == "__main__":
    unittest.main()

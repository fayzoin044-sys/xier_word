import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from base.config import Config

class ConfigTests(unittest.TestCase):
    def test_public_template_has_no_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = Config("config.example.ini")
            self.assertEqual((cfg.MYSQL_PASSWORD, cfg.REDIS_PASSWORD, cfg.LLM_API_KEY), ("", "", ""))
            self.assertEqual(cfg.MYSQL_HOST, "127.0.0.1")

    def test_environment_overrides_local_values_without_changing_file(self):
        source = Path(__file__).resolve().parents[1] / "config.example.ini"
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "config.ini"
            target.write_bytes(source.read_bytes())
            before = target.read_bytes()
            with patch.dict(os.environ, {"MYSQL_PASSWORD": "test-only", "REDIS_PASSWORD": "test-redis", "MYSQL_PORT": "3308", "MILVUS_HOST": "localhost", "DEEPSEEK_API_KEY": "test-api"}, clear=True):
                cfg = Config(target)
                self.assertEqual(cfg.MYSQL_PASSWORD, "test-only")
                self.assertEqual(cfg.REDIS_PASSWORD, "test-redis")
                self.assertEqual(cfg.MYSQL_PORT, 3308)
                self.assertEqual(cfg.MILVUS_URI, "http://localhost:19530")
                self.assertEqual(cfg.LLM_API_KEY, "test-api")
            self.assertEqual(target.read_bytes(), before)

    def test_missing_explicit_config_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                Config(Path(folder) / "missing.ini")

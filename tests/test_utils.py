import json
import logging
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import get_logger, load_config


class TestLoadConfig(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        for f in os.listdir(self.tmpdir):
            os.remove(os.path.join(self.tmpdir, f))
        os.rmdir(self.tmpdir)

    def _write_config(self, data, filename='config.json'):
        path = os.path.join(self.tmpdir, filename)
        with open(path, 'w') as f:
            json.dump(data, f)
        return path

    def test_loads_valid_json(self):
        path = self._write_config({'market_cap_min': 1e9, 'top_n_picks': 10})
        cfg = load_config(path)
        self.assertEqual(cfg['market_cap_min'], 1e9)
        self.assertEqual(cfg['top_n_picks'], 10)

    def test_returns_dict(self):
        path = self._write_config({'key': 'value'})
        result = load_config(path)
        self.assertIsInstance(result, dict)

    def test_raises_for_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_config(os.path.join(self.tmpdir, 'nonexistent.json'))

    def test_raises_for_invalid_json(self):
        path = os.path.join(self.tmpdir, 'bad.json')
        with open(path, 'w') as f:
            f.write("{this is not valid json}")
        with self.assertRaises(json.JSONDecodeError):
            load_config(path)

    def test_loads_nested_config(self):
        data = {'webull': {'username': 'u', 'password': 'p'}, 'strategies': {'csp': {'enabled': True}}}
        path = self._write_config(data)
        cfg = load_config(path)
        self.assertEqual(cfg['webull']['username'], 'u')
        self.assertTrue(cfg['strategies']['csp']['enabled'])


class TestGetLogger(unittest.TestCase):

    def test_returns_logger_instance(self):
        logger = get_logger('test_logger')
        self.assertIsInstance(logger, logging.Logger)

    def test_logger_uses_given_name(self):
        logger = get_logger('my_app')
        self.assertEqual(logger.name, 'my_app')

    def test_default_level_is_info(self):
        logger = get_logger('level_test')
        self.assertEqual(logger.level, logging.INFO)

    def test_custom_level_is_applied(self):
        logger = get_logger('debug_test', level=logging.DEBUG)
        self.assertEqual(logger.level, logging.DEBUG)

    def test_same_name_returns_same_logger(self):
        a = get_logger('singleton')
        b = get_logger('singleton')
        self.assertIs(a, b)


class TestGetLoggerWithFile(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove RotatingFileHandler before cleanup to release file handles on Windows
        import logging.handlers
        logger_names = ['file_logger', 'dedup_logger', 'subdir_logger']
        for name in logger_names:
            lg = logging.getLogger(name)
            for h in list(lg.handlers):
                if isinstance(h, logging.handlers.RotatingFileHandler):
                    h.close()
                    lg.removeHandler(h)
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_log_file_creates_file(self):
        log_path = os.path.join(self.tmpdir, 'test.log')
        get_logger('file_logger', log_file=log_path)
        self.assertTrue(os.path.exists(log_path))

    def test_log_file_adds_rotating_handler(self):
        from logging.handlers import RotatingFileHandler
        log_path = os.path.join(self.tmpdir, 'test2.log')
        logger = get_logger('file_logger', log_file=log_path)
        handler_types = [type(h) for h in logger.handlers]
        self.assertIn(RotatingFileHandler, handler_types)

    def test_no_duplicate_handler_on_repeated_calls(self):
        from logging.handlers import RotatingFileHandler
        log_path = os.path.join(self.tmpdir, 'dedup.log')
        get_logger('dedup_logger', log_file=log_path)
        get_logger('dedup_logger', log_file=log_path)
        logger = logging.getLogger('dedup_logger')
        rfh_count = sum(1 for h in logger.handlers if isinstance(h, RotatingFileHandler))
        self.assertEqual(rfh_count, 1)

    def test_log_file_in_subdirectory_creates_dirs(self):
        log_path = os.path.join(self.tmpdir, 'subdir', 'app.log')
        get_logger('subdir_logger', log_file=log_path)
        self.assertTrue(os.path.exists(log_path))

    def test_no_log_file_no_rotating_handler(self):
        from logging.handlers import RotatingFileHandler
        logger = get_logger('no_file_logger')
        rfh_handlers = [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]
        self.assertEqual(len(rfh_handlers), 0)


if __name__ == '__main__':
    unittest.main()

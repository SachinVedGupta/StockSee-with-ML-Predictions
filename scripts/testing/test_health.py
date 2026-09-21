import sys
import subprocess
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from getPricesFlask import app, prediction_lock

class HealthTests(unittest.TestCase):
    def test_health_and_validation_do_not_load_tensorflow(self):
        # Other test modules import TensorFlow during discovery. Check startup
        # in a fresh interpreter so test order cannot mask an eager import.
        result = subprocess.run([sys.executable, '-c', """
import sys
from getPricesFlask import app
with app.test_client() as client:
    assert client.get('/health').json == {'status': 'ok'}
    assert client.get('/predicted_prices').status_code == 400
assert 'tensorflow' not in sys.modules
"""], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_busy_prediction_leaves_health_responsive_and_releases_lock(self):
        started, release = Event(), Event()
        def inference(_):
            started.set()
            release.wait(5)
            return {'historical_dates': ['2026-01-01'], 'historical_prices': [10],
                    'prediction_dates': ['2026-01-02'], 'predictions': [11]}
        def submit():
            with app.test_client() as client:
                return client.get('/predicted_prices?ticker=AAPL').status_code
        with patch('getPricesFlask.predict_stock_price', inference), ThreadPoolExecutor() as executor:
            first = executor.submit(submit)
            try:
                self.assertTrue(started.wait(2))
                with app.test_client() as client:
                    busy = client.get('/predicted_prices?ticker=FIG')
                    self.assertEqual(busy.status_code, 429)
                    self.assertEqual(busy.json['code'], 'PREDICTION_BUSY')
                    self.assertEqual(client.get('/health').status_code, 200)
            finally:
                release.set()
            self.assertEqual(first.result(), 200)
        self.assertFalse(prediction_lock.locked())

    def test_failed_inference_releases_slot(self):
        with patch('getPricesFlask.predict_stock_price', side_effect=ValueError('test')):
            with app.test_client() as client:
                self.assertEqual(client.get('/predicted_prices?ticker=AAPL').status_code, 500)
        self.assertFalse(prediction_lock.locked())

if __name__ == '__main__':
    unittest.main()

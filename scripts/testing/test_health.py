import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from getPricesFlask import app

class HealthTests(unittest.TestCase):
    def test_health_and_validation_do_not_load_tensorflow(self):
        with app.test_client() as client:
            self.assertEqual(client.get('/health').json, {'status': 'ok'})
            self.assertEqual(client.get('/predicted_prices').status_code, 400)
        self.assertNotIn('tensorflow', sys.modules)

if __name__ == '__main__':
    unittest.main()

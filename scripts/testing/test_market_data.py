import sys
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stock_prediction.getHistoricalDataFeatures import get_stock_features

class MarketDataRetryTests(unittest.TestCase):
    def test_empty_initial_response_retries_with_fresh_ticker(self):
        frame = pd.DataFrame({name: [10, 11, 12] for name in
                              ['Open', 'Close', 'High', 'Low', 'Volume']},
                             index=pd.date_range('2025-01-01', periods=3, name='Date'))
        first, second = Mock(), Mock()
        first.history.return_value = pd.DataFrame()
        second.history.return_value = frame
        with patch('stock_prediction.getHistoricalDataFeatures.yf.Ticker', side_effect=[first, second]) as factory, patch('stock_prediction.getHistoricalDataFeatures.sleep'):
            result = get_stock_features('AAPL')
        self.assertEqual(factory.call_count, 2)
        self.assertEqual(result['Close'].tolist(), [11, 12])

    def test_persistent_failure_stops_after_two_attempts(self):
        ticker = Mock()
        ticker.history.side_effect = RuntimeError('provider failed')
        with patch('stock_prediction.getHistoricalDataFeatures.yf.Ticker', return_value=ticker), patch('stock_prediction.getHistoricalDataFeatures.sleep'):
            with self.assertRaisesRegex(ValueError, 'Market data is unavailable'):
                get_stock_features('AAPL')
        self.assertEqual(ticker.history.call_count, 2)

if __name__ == '__main__':
    unittest.main()

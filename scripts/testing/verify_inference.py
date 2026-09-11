"""Live audit: run .venv/bin/python testing/verify_inference.py from scripts/."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import sys
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stock_prediction.stockPredictionWithSentimentModel as prediction
from sentiment.sentimentAnalysisModel import sentiment_from_sentence


def hashes():
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path('sentiment_storage').glob('*') if p.suffix in {'.keras', '.pickle'}}


def main():
    os.chdir(Path(__file__).resolve().parents[1])
    before = hashes()
    load_model = prediction.tf.keras.models.load_model
    get_features = prediction.get_stock_features_with_sentiment
    calls, frames = [], []

    def features(ticker):
        frame = get_features(ticker)
        frames.append(frame)
        return frame

    def load(*args, **kwargs):
        model = load_model(*args, **kwargs)
        original_predict = model.predict
        def predict(inputs, *predict_args, **predict_kwargs):
            predict_kwargs.setdefault('verbose', 0)
            output = original_predict(inputs, *predict_args, **predict_kwargs)
            calls.append({'input': np.array(inputs).copy(), 'output': np.array(output).copy()})
            return output
        model.predict = predict
        return model

    with patch.object(prediction.tf.keras.models, 'load_model', side_effect=load), \
         patch.object(prediction, 'get_stock_features_with_sentiment', side_effect=features), \
         patch.object(prediction.tf.keras.Model, 'fit', side_effect=AssertionError('Training forbidden')), \
         patch.object(prediction.tf.keras.Model, 'save', side_effect=AssertionError('Model saving forbidden')):
        result = prediction.predict_stock_price('AAPL')
        # Test input only; not represented as fetched news.
        score = float(sentiment_from_sentence('The company reported strong revenue growth.'))
    stock_calls = [c for c in calls if c['input'].ndim == 3]
    first = stock_calls[0]
    assert first['input'].shape == (1, 200, 2)
    assert first['output'].shape == (1, 50)
    assert np.isfinite(first['output']).all()
    expected = first['output'].copy()
    expected += first['input'][0, -1, 0] - expected[0, 0]
    scaler = StandardScaler().fit(frames[0]['Close'].values.reshape(-1, 1))
    expected = scaler.inverse_transform(expected)[0]
    np.testing.assert_allclose(result['predictions'], expected, rtol=1e-6)
    assert len(result['predictions']) == 50
    assert 0 <= score <= 1
    assert any(c['input'].shape == (1, 100) for c in calls)
    assert hashes() == before, 'Model artifacts changed'
    print(json.dumps({
        'stock_predict_calls': len(stock_calls),
        'input_shape': list(first['input'].shape), 'raw_output_shape': list(first['output'].shape),
        'returned_prices_match_model_after_existing_alignment': True,
        'sentiment_model_test_score': score,
        'historical_news_sentiment': frames[0].attrs.get('sentiment'),
        'model_artifacts_unchanged': True,
    }, indent=2))


if __name__ == '__main__':
    main()

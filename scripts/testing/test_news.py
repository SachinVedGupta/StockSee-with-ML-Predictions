import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import requests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sentiment.getNewsArticle import get_articles, get_article, NewsUnavailable, _fetch_articles


class NewsTests(unittest.TestCase):
    def setUp(self):
        _fetch_articles.cache_clear()

    def test_missing_token(self):
        with patch.dict(os.environ, {'NEWS_API_TOKEN': ''}):
            with self.assertRaisesRegex(NewsUnavailable, 'not configured'):
                get_articles('AAPL')

    @patch.dict(os.environ, {'NEWS_API_TOKEN': 'test-key'})
    @patch('sentiment.getNewsArticle.requests.get')
    def test_real_fields_and_safe_links(self, get):
        get.return_value = Mock(ok=True)
        get.return_value.json.return_value = {'data': [
            {'title': 'Example story', 'url': 'https://example.com/story', 'description': 'Revenue rose', 'snippet': 'More context', 'published_at': '2024-08-05T12:00:00Z'},
            {'title': 'Unsafe', 'url': 'javascript:alert(1)'},
        ]}
        self.assertEqual(len(get_articles('AAPL')), 1)
        text = get_article('AAPL', '2024-08-05')
        self.assertEqual(text, 'Example story Revenue rose More context')
        self.assertEqual(get.call_args.kwargs['params']['published_on'], '2024-08-05')

    @patch.dict(os.environ, {'NEWS_API_TOKEN': 'test-key'})
    @patch('sentiment.getNewsArticle.requests.get')
    def test_failure_does_not_leak_token(self, get):
        get.side_effect = requests.Timeout('private URL with test-key')
        with self.assertRaises(NewsUnavailable) as caught:
            get_articles('AAPL')
        self.assertNotIn('test-key', str(caught.exception))

    @patch.dict(os.environ, {'NEWS_API_TOKEN': 'test-key'})
    @patch('sentiment.getNewsArticle.requests.get')
    def test_auth_error_is_distinct_from_no_results(self, get):
        get.return_value = Mock(ok=False, status_code=401)
        with self.assertRaisesRegex(NewsUnavailable, 'HTTP 401'):
            get_articles('AAPL')
        get.return_value = Mock(ok=True)
        get.return_value.json.return_value = {'data': []}
        self.assertEqual(get_article('AAPL', '2024-08-05'), 'N/A')

    @patch.dict(os.environ, {'NEWS_API_TOKEN': 'test-key'})
    @patch('sentiment.getNewsArticle.requests.get')
    def test_successful_requests_are_cached(self, get):
        get.return_value = Mock(ok=True)
        get.return_value.json.return_value = {'data': []}
        get_articles('AAPL')
        get_articles('AAPL')
        self.assertEqual(get.call_count, 1)

    @patch.dict(os.environ, {'NEWS_API_TOKEN': 'primary', 'NEXT_NEWS_API_TOKEN': 'backup'})
    @patch('sentiment.getNewsArticle.requests.get')
    def test_secondary_token_after_quota_failure(self, get):
        success = Mock(ok=True)
        success.json.return_value = {'data': [{'title':'Story', 'url':'https://example.com/story'}]}
        get.side_effect = [Mock(ok=False, status_code=402), success]
        self.assertEqual(len(get_articles('AAPL')), 1)
        self.assertEqual([call.kwargs['params']['api_token'] for call in get.call_args_list], ['primary', 'backup'])


if __name__ == '__main__':
    unittest.main()

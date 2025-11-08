import requests
import os

def getArticle(ticker, date):
  # 2024-12-30 is format of date
  api_token = os.environ.get('NEWS_API_TOKEN', 'riTKpnJ9W4pSkEDcIKHAPg0okfsxDiJXDmCg4n18')
  url = f"https://api.thenewsapi.com/v1/news/all?api_token={api_token}&search={ticker}&published_on={date}"
  article = "N/A"


  response = requests.get(url)
  all_items = response.json().get('data', '') # includes title, description, url, language, categories (business, tech), relevance score

  if len(all_items) > 0:
    first_item = all_items[0]
    article = first_item.get('title', '') + first_item.get('description', '') + first_item.get('snippet', '')

  return article
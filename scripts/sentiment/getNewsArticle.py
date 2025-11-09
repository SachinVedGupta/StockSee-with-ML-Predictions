import requests
import os

# for a specific stock and date, get the most relevant article about them
def getArticle(ticker, date):
  api_token = os.environ.get('NEWS_API_TOKEN', 'riTKpnJ9W4pSkEDcIKHAPg0okfsxDiJXDmCg4n18')
  url = f"https://api.thenewsapi.com/v1/news/all?api_token={api_token}&search={ticker}&published_on={date}"
  
  article = "N/A"
  response = requests.get(url)
  
  all_articles = response.json().get('data', '')
  if len(all_articles) > 0:
    first_article = all_articles[0]
    article = first_article.get('title', '') + first_article.get('description', '') + first_article.get('snippet', '')

  return article
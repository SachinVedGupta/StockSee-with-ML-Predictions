import requests

def getArticle(ticker, date):
  # 2024-12-30 is format of date
  url = f"https://api.thenewsapi.com/v1/news/all?api_token=riTKpnJ9W4pSkEDcIKHAPg0okfsxDiJXDmCg4n18&search={ticker}&published_on={date}"
  article = "N/A"


  response = requests.get(url)
  all_items = response.json().get('data', '') # includes title, description, url, language, categories (business, tech), relevance score

  if len(all_items) > 0:
    first_item = all_items[0]
    article = first_item.get('title', '') + first_item.get('description', '') + first_item.get('snippet', '')

  return article
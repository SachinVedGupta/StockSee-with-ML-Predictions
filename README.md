## USE FULLY DEPLOYED APP --> https://stock-see.vercel.app/ 

Hackathon Project Link: https://devpost.com/software/stocksee 

## WHAT IS STOCKSEE

Correlating real-world events with stock prices
- To foster an understanding of the stock price curve and showcase that it doesn't just randomly move but is actually based on real life and the real world (industry trends, the socioeconomic state, company earnings reports, product releases)

```mermaid
graph TB
    subgraph "Frontend Layer"
        A[Next.js 14 App<br/>page.tsx] --> B["4. Chart.js Visualization"]
        A --> C["1. User Input Form"]
        A --> S["3. Significant Shift<br/>Detection"]
    end
    
    subgraph "API Layer"
        D[Flask Backend<br/>getPricesFlask.py]
        E[Next.js API<br/>route.ts]
    end
    
    subgraph "ML Pipeline"
        F[Stock Prediction<br/>LSTM Model]
        G[Sentiment Analysis<br/>NLP Model]
        H[Feature Engineering]
    end
    
    subgraph "External Services"
        I[Yahoo Finance API]
        J[TheNewsAPI]
        K[Google Gemini AI]
        L["5. Google Image Search"]
    end
    
    A -->|2. GET /predicted_prices| D
    S -->|POST /api/gemini<br/>with dates| E
    D --> F
    F --> H
    H --> I
    H --> G
    G --> J
    E --> K
    A --> L
    
    style A fill:#3b82f6,color:#fff
    style S fill:#8b5cf6,color:#fff
    style D fill:#10b981,color:#fff
    style E fill:#10b981,color:#fff
    style F fill:#f59e0b,color:#fff
    style G fill:#f59e0b,color:#fff
```

## WHAT IT DOES

Users are guided to a search bar where they can search a company stock for example "AAPL" and almost instantly they can see the stock price over the last two years as a graph, with green and red dots spread out on the line graph. When they hover over the dots, the green dots explain why there is a general increasing trend in the stock and a news article to back it up, along with the price change from the previous day and what it is predicted to be from. An image shows up on the side of the graph showing the company image as well.

## ABOUT THE PREDICTION MODEL

The yellow dots on the graph above represent predictions for the next 50 days of stock prices (into the future), made using a TensorFlow LSTM machine learning model. Each prediction is based on a batch of 200 previous daily stock prices, and the model forecasts the prices for the upcoming 50 days. A total of 1500 days of historical data is utilized in the training and validation process for each stock. By repeatedly training the model on all 30 stocks in the DOW JONES, a more comprehensive model has been created, of which is used to predicted the entered stock.

The model takes as input both the stock's daily closing prices and sentiment scores derived from public news articles. These sentiment scores are generated through a custom natural language processing (NLP) model, developed using TensorFlow and trained on a Kaggle dataset. The NLP model analyzes news articles related to the company, gathered via a news API, to assign a sentiment score for each day. By including not only the historical stock prices but also external factors like public sentiment and company news, the model is better equipped to predict future stock prices. Simply relying on past prices is insufficient, as factors such as company performance, innovation, and public perception play a critical role, making sentiment analysis an essential input for the prediction model. Furthermore, by creating a comprehensive model trained on 30 stocks (the ones in the DOW JONES), the prediction model used becomes even more accurate.

Note: The loss/accuracy curves below will stay constant in repeated entries since the pre-trained (on the 30 DOW JONES stocks) model is being loaded in. By going in LOCAL, one can then further train the prediction model and/or sentiment analysis ML models. It can also be set so that every new stock ticker entry further trains and improves the prediction model, though this is not a feature in the deployed version due to RAM constraints.

Can also view GRAPHS showcasing the ML model's performance metrics (loss, accuracy, etc...)

## IMAGES

![image](https://github.com/user-attachments/assets/6d1791ef-b6a0-4015-98b7-5327b8eaa264)
![image](https://github.com/user-attachments/assets/e47994ef-5cf1-4901-9921-4469549c07e5)
![image](https://github.com/user-attachments/assets/c2d170f4-9d62-441c-84ce-cce589c314c9)
![image](https://github.com/user-attachments/assets/3052b4ff-ed52-442e-9b32-cc46468e9b67)




## USING LOCALLY

When going deployed to local version, change...
  - (change url variable) page.tsx api request to the local backend not deployed backend url
  - can change training setting in stockPredictionsWithSentiment (switch between training on just entered companies historical date, further training ML model with entered company for continuous ML prediction model improvemnt, or just using the pre-trained model)
  - can also retrain the sentiment ML model if needed by calling the allocated sentimentML.py function

To run
- use "npm i" if needed then "npm run dev" in the overall directory (start frontend)
- within the "scripts" folder run "python getPricesFlask.py" (start backend)
- the application can now be used locally!

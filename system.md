# StockSee System Architecture

> **AI-Powered Stock Understanding Application**

A full-stack application that predicts stock prices 50 days into the future using LSTM neural networks trained on historical price data and news sentiment analysis.

---

## 🏗️ System Overview

```mermaid
graph TB
    subgraph "Frontend Layer"
        A[Next.js 14 App<br/>page.tsx] --> B[Chart.js Visualization]
        A --> C[User Input Form]
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
        L[Google Image Search]
    end
    
    A -->|GET /predicted_prices| D
    A -->|POST /api/gemini| E
    D --> F
    F --> H
    H --> I
    H --> G
    G --> J
    E --> K
    A --> L
    
    style A fill:#3b82f6,color:#fff
    style D fill:#10b981,color:#fff
    style E fill:#10b981,color:#fff
    style F fill:#f59e0b,color:#fff
    style G fill:#f59e0b,color:#fff

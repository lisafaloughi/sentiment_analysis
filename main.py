from transformers import pipeline

def load_sentiment_model(model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
    """Loads a sentiment analysis model from Hugging Face."""
    return pipeline("sentiment-analysis", model=model_name)

def analyze_sentiment(model, texts):
    """Analyzes sentiment for a list of financial news texts."""
    return model(texts)

if __name__ == "__main__":
    # Example fine-tuned financial sentiment model (replace with an actual fine-tuned one)
    model_name = "FinBERT-Tone/finance-sentiment-analysis"  # Replace with the correct model name from HF
    sentiment_model = load_sentiment_model(model_name)
    
    # Example financial news headlines
    financial_news = [
        "Federal Reserve signals rate hikes to continue amid inflation concerns",
        "Tech stocks surge as earnings beat expectations",
        "Oil prices fall as demand weakens in Asia"
    ]
    
    # Perform sentiment analysis
    results = analyze_sentiment(sentiment_model, financial_news)
    
    # Print results
    for news, sentiment in zip(financial_news, results):
        print(f"News: {news}\nSentiment: {sentiment['label']} (Score: {sentiment['score']:.2f})\n")

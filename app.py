import os
import requests
from flask import Flask, render_template, request, flash
from bs4 import BeautifulSoup
from newspaper import Article
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "your_secret_key")

# Hugging Face Inference API configuration
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"

headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def fetch_articles(query=""):
    """Fetch top news articles or based on search query"""
    if not NEWS_API_KEY:
        flash("News API key is missing.", "danger")
        return []
    
    url = (f"https://newsapi.org/v2/everything?q={query}&language=en&apiKey={NEWS_API_KEY}" if query else
           f"https://newsapi.org/v2/everything?q=india&language=en&apiKey={NEWS_API_KEY}")

    try:
        response = requests.get(url)
        print("API URL:", url)
        print("Status Code:", response.status_code)
        response.raise_for_status()
        data = response.json()
        print("API Response:", data)

        if data.get("status") != "ok":
            flash(f"API error: {data.get('message', 'Unknown error')}", "danger")
            return []

        articles = data.get("articles", [])[:10]
        if not articles:
            flash("No articles found for this query.", "warning")
        return articles
    except requests.RequestException as e:
        flash(f"Error fetching articles: {str(e)}", "danger")
        return []

def fetch_article_content(url):
    """Extract article content and image"""
    article = Article(url)
    try:
        article.download()
        article.parse()
        return article.text.strip(), article.top_image
    except Exception:
        return "Content not available.", None

def summarize_text(text):
    """Summarize the article content using Hugging Face Inference API"""
    # Remove word count restriction; summarize regardless of length
    truncated_text = " ".join(text.split()[:512])  # Still truncate to avoid API input limits
    input_text = truncated_text  # BART doesn't require "summarize:" prefix
    word_count = len(text.split())
    max_len = max(30, min(140, word_count // 2)) if word_count > 10 else 20  # Adjust max_len for very short texts

    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": input_text, "parameters": {"max_length": max_len, "min_length": max(10, max_len // 2), "do_sample": False}})
        response.raise_for_status()
        summary = response.json()
        if isinstance(summary, str):
            return summary
        elif isinstance(summary, list) and len(summary) > 0:
            return summary[0]["summary_text"]
        else:
            return "Summary unavailable."
    except requests.RequestException as e:
        return f"Summary unavailable: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def index():
    """Render homepage with news articles"""
    search_query = request.form.get("search", "").strip().lower() if request.method == 'POST' else ""
    
    articles = fetch_articles(search_query if search_query else "")
    summaries = []

    for entry in articles:
        content, image = fetch_article_content(entry['url'])
        summary = summarize_text(content)

        summaries.append({
            'title': entry.get('title', 'No Title'),
            'image': image or "https://via.placeholder.com/150",
            'link': entry.get('url', ''),
            'summary': summary,
            'source': entry.get('source', {}).get('name', 'Unknown'),
            'published': entry.get('publishedAt', 'Unknown'),
        })

    return render_template('index.html', articles=summaries, search_query=search_query)

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
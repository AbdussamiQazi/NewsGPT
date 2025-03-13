import os
from flask import Flask, render_template, request, flash
from bs4 import BeautifulSoup
from newspaper import Article
from transformers import BartForConditionalGeneration, BartTokenizer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "your_secret_key")

# Load DistilBART model and tokenizer at startup
MODEL_NAME = "sshleifer/distilbart-cnn-6-6"
print(f"Loading model: {MODEL_NAME}")
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
print("Model loaded successfully")

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def fetch_articles(query=""):
    """Fetch top news articles or based on search query"""
    if not NEWS_API_KEY:
        flash("News API key is missing.", "danger")
        return []
    
    url = (f"https://newsapi.org/v2/everything?q={query}&language=en&apiKey={NEWS_API_KEY}" if query else
           f"https://newsapi.org/v2/everything?q=india&language=en&apiKey={NEWS_API_KEY}")

    try:
        import requests
        response = requests.get(url)
        print("Status Code:", response.status_code)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "ok":
            flash(f"API error: {data.get('message', 'Unknown error')}", "danger")
            return []

        articles = data.get("articles", [])[:6]
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
    """Summarize the article content using local DistilBART model"""
    try:
        truncated_text = " ".join(text.split()[:512])  # Limit input size
        inputs = tokenizer(truncated_text, return_tensors="pt", max_length=1024, truncation=True)
        word_count = len(text.split())
        max_len = max(30, min(140, word_count // 2)) if word_count > 10 else 20

        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_len,
            min_length=max(10, max_len // 2),
            do_sample=False,
            num_beams=4,  # Beam search for better quality
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        print(f"Error summarizing text: {str(e)}")
        return "Summary unavailable."

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
    port = int(os.getenv("PORT", 5000))  # Use Render's PORT or default to 5000 locally
    app.run(host="0.0.0.0", port=port, debug=False)  # Bind to 0.0.0.0 for external access
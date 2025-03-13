import os
import logging
import requests
import nltk
from flask import Flask, render_template, request, flash
from bs4 import BeautifulSoup
from newspaper import Article
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Download NLTK punkt data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt data...")
    nltk.download('punkt', quiet=True)
    logger.info("NLTK punkt data downloaded successfully")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "your_secret_key")

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def fetch_articles(query=""):
    """Fetch top news articles or based on search query"""
    if not NEWS_API_KEY:
        logger.error("News API key is missing.")
        flash("News API key is missing.", "danger")
        return []
    
    url = (f"https://newsapi.org/v2/everything?q={query}&language=en&apiKey={NEWS_API_KEY}" if query else
           f"https://newsapi.org/v2/everything?q=india&language=en&apiKey={NEWS_API_KEY}")

    try:
        response = requests.get(url, timeout=10)
        logger.info(f"Fetching articles from {url}, Status Code: {response.status_code}")
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "ok":
            logger.error(f"API error: {data.get('message', 'Unknown error')}")
            flash(f"API error: {data.get('message', 'Unknown error')}", "danger")
            return []

        articles = data.get("articles", [])[:6]
        if not articles:
            logger.warning("No articles found for this query.")
            flash("No articles found for this query.", "warning")
        return articles
    except requests.RequestException as e:
        logger.error(f"Error fetching articles: {str(e)}")
        flash(f"Error fetching articles: {str(e)}", "danger")
        return []

def fetch_article_content(url):
    """Extract article content and image"""
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text.strip()
        image = article.top_image or "https://via.placeholder.com/150"
        return text[:2000], image  # Limit text size
    except Exception as e:
        logger.error(f"Error fetching content from {url}: {str(e)}")
        return "Content not available.", None

def summarize_text(text):
    """Summarize the article content using sumy LSA"""
    try:
        parser = PlaintextParser.from_string(text[:512], Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, 2)  # 2 sentences
        summary_text = " ".join(str(sentence) for sentence in summary)
        logger.info("Summary generated successfully")
        return summary_text
    except Exception as e:
        logger.error(f"Error summarizing text: {str(e)}")
        return "Summary unavailable."

@app.route('/', methods=['GET', 'POST'])
def index():
    """Render homepage with news articles"""
    try:
        search_query = request.form.get("search", "").strip().lower() if request.method == 'POST' else ""
        logger.info(f"Processing request with search query: {search_query}")
        
        articles = fetch_articles(search_query if search_query else "")
        summaries = []

        for entry in articles:
            content, image = fetch_article_content(entry['url'])
            summary = summarize_text(content) if content else "No summary available."

            summaries.append({
                'title': entry.get('title', 'No Title'),
                'image': image,
                'link': entry.get('url', ''),
                'summary': summary,
                'source': entry.get('source', {}).get('name', 'Unknown'),
                'published': entry.get('publishedAt', 'Unknown'),
            })

        return render_template('index.html', articles=summaries, search_query=search_query)
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        flash("An unexpected error occurred.", "danger")
        return render_template('index.html', articles=[], search_query="")

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    logger.info(f"Starting app on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
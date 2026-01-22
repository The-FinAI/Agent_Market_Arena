#!/usr/bin/env python3
"""
Enhanced Daily Crypto and Stock News and Price Fetcher
Supports both cryptocurrencies (BTC, ETH) and stocks (TSLA, AAPL)
"""

import os
import json
import logging
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from openai import OpenAI
from binance.client import Client
from typing import List, Dict, Optional, Tuple
import yfinance as yf
from time import sleep
import time
import random
import re
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_result,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler('get_daily_action.log'),
        logging.StreamHandler()
    ]
)

# API Keys
CRYPTONEWS_API_KEY = os.getenv('CRYPTONEWS_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
NEWSDATA_API_KEY = os.getenv('NEWSDATA_API_KEY')  # For stocks
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')  # For Finnhub API

# Constants
DATE_FORMAT = '%Y-%m-%d'
CRYPTONEWS_API_URL = 'https://cryptonews-api.com/api/v1'
NEWSDATA_API_URL = 'https://newsdata.io/api/1/news'

# Initialize API clients
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
binance_client = Client(None, None, tld='us')

# Define asset types
CRYPTO_SYMBOLS = ['BTC', 'ETH', 'ADA', 'SOL', 'DOT', 'LINK', 'UNI', 'MATIC', 'AVAX', 'ATOM']
STOCK_SYMBOLS = ['TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'NFLX', 'AMD', 'INTC', 'BMRN', 'MRNA', 'SIEN', 'MAZE']

# Stock symbol to CIK mapping for SEC filings
STOCK_CIK_MAPPING = {
    'AAPL': '0000320193',
    'MSFT': '0000789019', 
    'GOOGL': '0001652044',
    'AMZN': '0001018724',
    'TSLA': '0001318605',
    'NVDA': '0001045810',
    'META': '0001326801',
    'NFLX': '0001065280',
    'AMD': '0000002488',
    'INTC': '0000050863',
    'BMRN': '0001048477',
    'MRNA': '0001682852',
    'SIEN': '0001679873',
    'MAZE': '0001874676'
}

# Try to import asset_manager for smart asset identification
try:
    from asset_manager import get_asset_info
    ASSET_MANAGER_AVAILABLE = True
except ImportError:
    ASSET_MANAGER_AVAILABLE = False
    print("ℹ️  asset_manager not imported, using default identification logic")


# ==================== Google News Scraping ====================
def is_rate_limited(response):
    """Check if the response indicates rate limiting (status code 429)"""
    return response.status_code == 429


@retry(
    retry=(retry_if_result(is_rate_limited)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
)
def make_google_request(url, headers):
    """Make a request with retry logic for rate limiting"""
    # Random delay before each request to avoid detection
    time.sleep(random.uniform(2, 6))
    response = requests.get(url, headers=headers)
    return response


def get_google_news_data(query: str, start_date: str, end_date: str) -> List[Dict]:
    """
    Scrape Google News search results for a given query and date range.
    
    Args:
        query: str - search query (e.g., "TSLA", "Tesla")
        start_date: str - start date in the format yyyy-mm-dd
        end_date: str - end date in the format yyyy-mm-dd
    
    Returns:
        List[Dict]: List of news articles with title, snippet, source, etc.
    """
    # Convert dates to Google News format (mm/dd/yyyy)
    if "-" in start_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        start_date_fmt = start_dt.strftime("%m/%d/%Y")
    else:
        start_date_fmt = start_date
    if "-" in end_date:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        end_date_fmt = end_dt.strftime("%m/%d/%Y")
    else:
        end_date_fmt = end_date

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/101.0.4951.54 Safari/537.36"
        )
    }

    news_results = []
    page = 0
    max_pages = 3  # Limit to 3 pages to avoid too many requests
    
    while page < max_pages:
        offset = page * 10
        url = (
            f"https://www.google.com/search?q={query.replace(' ', '+')}"
            f"&tbs=cdr:1,cd_min:{start_date_fmt},cd_max:{end_date_fmt}"
            f"&tbm=nws&start={offset}"
        )

        try:
            response = make_google_request(url, headers)
            soup = BeautifulSoup(response.content, "html.parser")
            results_on_page = soup.select("div.SoaBEf")

            if not results_on_page:
                break  # No more results found

            for el in results_on_page:
                try:
                    link = el.find("a")["href"]
                    title = el.select_one("div.MBeuO").get_text()
                    snippet = el.select_one(".GI74Re").get_text()
                    date = el.select_one(".LfVVr").get_text()
                    source = el.select_one(".NUnG9d span").get_text()
                    news_results.append(
                        {
                            "link": link,
                            "title": title,
                            "snippet": snippet,
                            "text": snippet,  # Alias for compatibility
                            "date": date,
                            "source": source,
                            "source_name": source,
                            "source_type": "google_news"
                        }
                    )
                except Exception as e:
                    logging.debug(f"Error processing Google News result: {e}")
                    continue

            # Check for the "Next" link (pagination)
            next_link = soup.find("a", id="pnnext")
            if not next_link:
                break

            page += 1

        except Exception as e:
            logging.warning(f"Failed to fetch Google News: {e}")
            break

    logging.info(f"Found {len(news_results)} Google News articles for {query}")
    return news_results


# ==================== Finnhub API ====================
def get_finnhub_news(ticker: str, start_date: str, end_date: str) -> List[Dict]:
    """
    Get news from Finnhub API.
    
    Args:
        ticker: Stock ticker symbol (e.g., "TSLA")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    
    Returns:
        List[Dict]: List of news articles
    """
    if not FINNHUB_API_KEY:
        logging.debug("FINNHUB_API_KEY not set, skipping Finnhub news")
        return []
    
    url = "https://finnhub.io/api/v1/company-news"
    params = {
        'symbol': ticker,
        'from': start_date,
        'to': end_date,
        'token': FINNHUB_API_KEY
    }
    
    try:
        logging.info(f"Fetching Finnhub news for {ticker}...")
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            news_data = response.json()
            logging.info(f"Found {len(news_data)} Finnhub articles for {ticker}")
            
            formatted_news = []
            for article in news_data:
                # Convert timestamp to datetime string
                timestamp = article.get('datetime', 0)
                if timestamp:
                    try:
                        dt = datetime.fromtimestamp(timestamp)
                        date_str = dt.strftime("%Y-%m-%dT%H:%M:%S")
                    except:
                        date_str = str(timestamp)
                else:
                    date_str = ""
                
                formatted_article = {
                    'title': article.get('headline', ''),
                    'text': article.get('summary', ''),
                    'summary': article.get('summary', ''),
                    'url': article.get('url', ''),
                    'date': date_str,
                    'source': article.get('source', ''),
                    'source_name': article.get('source', ''),
                    'source_type': 'finnhub_api'
                }
                formatted_news.append(formatted_article)
            
            return formatted_news
        else:
            logging.warning(f"Finnhub API error: {response.status_code}")
            return []
            
    except Exception as e:
        logging.error(f"Error fetching Finnhub news: {e}")
        return []


# ==================== OpenAI Web Search ====================
def get_stock_news_openai(ticker: str, curr_date: str) -> Optional[str]:
    """
    Use OpenAI's web search capability to get stock news.
    
    Args:
        ticker: Stock ticker symbol (e.g., "TSLA")
        curr_date: Date in YYYY-MM-DD format
    
    Returns:
        str: News content from OpenAI web search, or None if failed
    """
    if not openai_client:
        logging.debug("OpenAI client not available, skipping web search")
        return None
    
    try:
        logging.info(f"Fetching OpenAI web search news for {ticker}...")
        
        response = openai_client.responses.create(
            model="gpt-4o",  # Use gpt-4o which supports web search
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"Can you search Social Media for {ticker} from 1 days before {curr_date} to {curr_date}? Make sure you only get the data posted during that period. Do not add any information not in this time. Be objective. Do not add last trade time or urls.",
                        }
                    ],
                }
            ],
            text={"format": {"type": "text"}},
            reasoning={},
            tools=[
                {
                    "type": "web_search_preview",
                    "user_location": {"type": "approximate"},
                    "search_context_size": "low",
                }
            ],
            temperature=1,
            max_output_tokens=4096,
            top_p=1,
            store=True,
        )
        
        # Extract content from response
        if hasattr(response, 'output') and len(response.output) > 1:
            content = response.output[1].content[0].text
            logging.info("OpenAI web search completed successfully")
            return content
        else:
            logging.warning("OpenAI web search returned no content")
            return None
            
    except Exception as e:
        logging.warning(f"OpenAI web search failed: {e}")
        return None

def is_crypto(symbol: str) -> bool:
    """Check if symbol is a cryptocurrency"""
    symbol_upper = symbol.upper()
    
    # First check hardcoded list
    if symbol_upper in CRYPTO_SYMBOLS:
        return True
    
    # If asset_manager is available, use smart identification
    if ASSET_MANAGER_AVAILABLE:
        try:
            asset_type, _, _ = get_asset_info(symbol_upper)
            return asset_type == "crypto"
        except Exception as e:
            print(f"⚠️ asset_manager identification failed: {e}")
    
    return False

def is_stock(symbol: str) -> bool:
    """Check if symbol is a stock"""
    symbol_upper = symbol.upper()
    
    # First check hardcoded list
    if symbol_upper in STOCK_SYMBOLS:
        return True
    
    # If asset_manager is available, use smart identification
    if ASSET_MANAGER_AVAILABLE:
        try:
            asset_type, _, _ = get_asset_info(symbol_upper)
            return asset_type == "stock"
        except Exception as e:
            print(f"⚠️ asset_manager identification failed: {e}")
    
    return False

def get_company_cik(symbol: str) -> Optional[str]:
    """Get CIK code for a stock symbol"""
    return STOCK_CIK_MAPPING.get(symbol.upper())

def fetch_sec_filings(symbol: str, date_str: str, filing_type: str = 'both') -> Tuple[List[str], List[str]]:
    """
    Fetch SEC 10K and 10Q filings for a stock symbol around a specific date.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        date_str: Date in YYYY-MM-DD format
        filing_type: 'both', '10-K', or '10-Q'
    
    Returns:
        Tuple of (10k_filings, 10q_filings) - lists of filing summaries
    """
    if not is_stock(symbol):
        return [], []
    
    cik = get_company_cik(symbol)
    if not cik:
        logging.warning(f"No CIK found for symbol {symbol}")
        return [], []
    
    target_date = datetime.strptime(date_str, DATE_FORMAT)
    ten_k_filings = []
    ten_q_filings = []
    
    try:
        # Fetch 10-K filings (annual reports)
        if filing_type in ['both', '10-K']:
            ten_k_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; SEC-Filing-Bot/1.0; +http://www.example.com/contact)',
                'Accept-Encoding': 'gzip, deflate',
                'Host': 'data.sec.gov'
            }
            
            response = requests.get(ten_k_url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                filings = data.get('filings', {}).get('recent', {})
                
                forms = filings.get('form', [])
                filing_dates = filings.get('filingDate', [])
                accession_numbers = filings.get('accessionNumber', [])
                
                for i, (form, filing_date, accession) in enumerate(zip(forms, filing_dates, accession_numbers)):
                    if form == '10-K':
                        filing_dt = datetime.strptime(filing_date, '%Y-%m-%d')
                        # Look for 10-K filings within 1 year of target date
                        if abs((filing_dt - target_date).days) <= 365:
                            filing_summary = f"10-K Filing for {symbol} filed on {filing_date} (Accession: {accession})"
                            ten_k_filings.append(filing_summary)
                            
                            if len(ten_k_filings) >= 2:  # Limit to 2 most recent
                                break
        
        # Fetch 10-Q filings (quarterly reports)  
        if filing_type in ['both', '10-Q']:
            if response.status_code == 200:  # Reuse the response from above
                for i, (form, filing_date, accession) in enumerate(zip(forms, filing_dates, accession_numbers)):
                    if form == '10-Q':
                        filing_dt = datetime.strptime(filing_date, '%Y-%m-%d')
                        # Look for 10-Q filings within 6 months of target date
                        if abs((filing_dt - target_date).days) <= 180:
                            filing_summary = f"10-Q Filing for {symbol} filed on {filing_date} (Accession: {accession})"
                            ten_q_filings.append(filing_summary)
                            
                            if len(ten_q_filings) >= 3:  # Limit to 3 most recent
                                break
            
        logging.info(f"Found {len(ten_k_filings)} 10-K and {len(ten_q_filings)} 10-Q filings for {symbol} around {date_str}")
        
    except Exception as e:
        logging.error(f"Error fetching SEC filings for {symbol}: {str(e)}")
    
    return ten_k_filings, ten_q_filings

def convert_date_format(date_str: str) -> str:
    """Convert YYYY-MM-DD to MMDDYYYY format for CryptoNews API"""
    date_obj = datetime.strptime(date_str, DATE_FORMAT)
    return date_obj.strftime('%m%d%Y')

def fetch_crypto_news_for_date(symbol: str, date_str: str) -> List[Dict]:
    """Fetch crypto news for a specific date and symbol"""
    articles = []
    
    if not CRYPTONEWS_API_KEY:
        logging.error("CRYPTONEWS_API_KEY not found in environment variables")
        return articles
    
    # Convert date for API
    api_date_format = f"{convert_date_format(date_str)}-{convert_date_format(date_str)}"
    
    params = {
        'tickers': symbol,
        'items': 50,
        'token': CRYPTONEWS_API_KEY,
        'date': api_date_format
    }
    
    try:
        logging.info(f"Fetching {symbol} crypto news for {date_str}...")
        response = requests.get(CRYPTONEWS_API_URL, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and data['data']:
                articles = data['data']
                logging.info(f"Found {len(articles)} crypto articles for {symbol} on {date_str}")
            else:
                logging.warning(f"No crypto articles found for {symbol} on {date_str}")
                
                # Try broader date range if no articles found
                logging.info(f"Trying broader date range for {symbol}...")
                broader_start = (datetime.strptime(date_str, DATE_FORMAT) - timedelta(days=1)).strftime(DATE_FORMAT)
                broader_end = (datetime.strptime(date_str, DATE_FORMAT)).strftime(DATE_FORMAT)
                
                broader_start_api = convert_date_format(broader_start)
                broader_end_api = convert_date_format(broader_end)
                broader_api_format = f"{broader_start_api}-{broader_end_api}"
                
                broader_params = {
                    'tickers': symbol,
                    'items': 100,
                    'token': CRYPTONEWS_API_KEY,
                    'date': broader_api_format
                }
                
                broader_response = requests.get(CRYPTONEWS_API_URL, params=broader_params, timeout=10)
                if broader_response.status_code == 200:
                    broader_data = broader_response.json()
                    if 'data' in broader_data and broader_data['data']:
                        articles = broader_data['data']
                        logging.info(f"Found {len(articles)} crypto articles for {symbol} in broader date range")
        else:
            logging.error(f"CryptoNews API error {response.status_code}: {response.text}")
    except Exception as e:
        logging.error(f"Error fetching crypto news: {str(e)}")
    
    return articles


def fetch_stock_news_and_filings_for_date(symbol: str, date_str: str, look_back_days: int = 1) -> Tuple[List[Dict], List[str], List[str]]:
    """
    Fetch stock news and SEC filings using local logic (no localhost API needed).
    
    Uses multiple sources:
    1. Google News scraping
    2. Finnhub API (if API key available)
    3. OpenAI Web Search (if API key available)
    
    Args:
        symbol: Stock ticker symbol (e.g., "TSLA")
        date_str: Date in YYYY-MM-DD format
        look_back_days: Number of days to look back for news
    
    Returns:
        Tuple of (news_list, 10k_filings, 10q_filings)
    """
    # Calculate date range
    end_date = datetime.strptime(date_str, "%Y-%m-%d")
    start_date = end_date - timedelta(days=look_back_days)
    start_date_str = start_date.strftime("%Y-%m-%d")
    
    all_news = []
    
    # 1. Get Google News
    try:
        print(f"🔍 Fetching Google News for {symbol}...")
        google_news = get_google_news_data(symbol, start_date_str, date_str)
        all_news.extend(google_news)
        print(f"📰 Found {len(google_news)} Google News articles")
    except Exception as e:
        logging.warning(f"Error fetching Google News for {symbol}: {e}")
    
    # 2. Get Finnhub News (if API key available)
    try:
        finnhub_news = get_finnhub_news(symbol, start_date_str, date_str)
        all_news.extend(finnhub_news)
        if finnhub_news:
            print(f"📰 Found {len(finnhub_news)} Finnhub news articles")
    except Exception as e:
        logging.warning(f"Error fetching Finnhub news for {symbol}: {e}")
    
    # 3. Get OpenAI Web Search News (if API key available)
    try:
        openai_news = get_stock_news_openai(symbol, date_str)
        if openai_news:
            all_news.append({
                'title': f'OpenAI Web Search Results for {symbol}',
                'text': openai_news,
                'content': openai_news,
                'source_type': 'openai_web_search',
                'source_name': 'OpenAI Web Search',
                'date': date_str
            })
            print(f"🤖 Got OpenAI Web Search results")
    except Exception as e:
        logging.warning(f"Error fetching OpenAI news for {symbol}: {e}")
    
    # 4. Get SEC filings (10K and 10Q)
    ten_k_filings, ten_q_filings = fetch_sec_filings(symbol, date_str)
    
    print(f"✅ Total: {len(all_news)} news articles, {len(ten_k_filings)} 10K filings, {len(ten_q_filings)} 10Q filings")
    return all_news, ten_k_filings, ten_q_filings

def fetch_stock_news_for_date(symbol: str, date_str: str) -> List[Dict]:
    """Compatibility function: returns news data only"""
    news, _, _ = fetch_stock_news_and_filings_for_date(symbol, date_str)
    return news

def fetch_yahoo_news_for_date(symbol: str, date_str: str) -> List[Dict]:
    sleep(10)
    """Fetch stock news for a specific date and symbol using Yahoo Finance, filtered by date."""
    articles = []
    try:
        logging.info(f"Fetching {symbol} news from Yahoo Finance...")
        ticker = yf.Ticker(symbol)
        news_items = ticker.news or []
        logging.info(f"Yahoo Finance returned {len(news_items)} raw news items for {symbol}")

        # Date boundary: only keep news <= current day 23:59:59 (avoid future data leakage)
        target_date = datetime.strptime(date_str, DATE_FORMAT)
        day_start = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0)
        day_end   = datetime(target_date.year, target_date.month, target_date.day, 23, 59, 59)

        filtered = []
        for i, item in enumerate(news_items[:50]):  # Check at most first 50 items
            try:
                # yfinance common field: providerPublishTime (int, unix seconds)
                # Some versions have it in item['content']['pubDate'], so we check both
                pub_ts = None
                if isinstance(item.get("providerPublishTime"), (int, float)):
                    pub_ts = int(item["providerPublishTime"])
                    pub_dt = datetime.utcfromtimestamp(pub_ts)  # yfinance timestamps are usually UTC
                else:
                    # Fallback: try content.pubDate (ISO format or date string)
                    content = item.get("content", {})
                    pub_str = content.get("pubDate") or content.get("pub_date") or ""
                    pub_dt = None
                    if pub_str:
                        # Parse only the date portion
                        try:
                            pub_dt = datetime.strptime(pub_str[:19], "%Y-%m-%dT%H:%M:%S")
                        except Exception:
                            try:
                                pub_dt = datetime.strptime(pub_str[:10], "%Y-%m-%d")
                            except Exception:
                                pub_dt = None

                if not pub_dt:
                    # Cannot parse time, skip to be conservative
                    continue

                # Filter condition: only keep news within the current day
                if day_start.date() <= pub_dt.date() <= day_end.date() and pub_dt <= day_end:
                    # Convert to standard structure
                    article = {
                        'title': item.get('title') or item.get('content', {}).get('title', 'No title'),
                        'text': item.get('summary') or item.get('content', {}).get('summary', ''),
                        'description': item.get('summary') or item.get('content', {}).get('summary', ''),
                        'source_name': item.get('publisher') or item.get('content', {}).get('provider', {}).get('displayName', 'Yahoo Finance'),
                        'source_id': 'yahoo_finance',
                        'pubDate': pub_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                        'sentiment': 'neutral'
                    }
                    filtered.append(article)

            except Exception as e:
                logging.debug(f"Error processing Yahoo Finance news item {i}: {e}")
                continue

        logging.info(f"Filtered to {len(filtered)} Yahoo articles for {symbol} on {date_str}")
        return filtered

    except Exception as e:
        logging.error(f"Error fetching Yahoo Finance news for {symbol}: {str(e)}")
        return articles


def fetch_news_for_date(symbol: str, date_str: str) -> List[Dict]:
    """Fetch news for a specific symbol and date (crypto or stock)"""
    articles = []
    
    if is_crypto(symbol):
        logging.info(f"Fetching crypto news for {symbol}...")
        articles = fetch_crypto_news_for_date(symbol, date_str)
    elif is_stock(symbol):
        logging.info(f"Fetching stock news for {symbol}...")
        # Try Yahoo Finance first for stocks
        articles = fetch_yahoo_news_for_date(symbol, date_str)
        
        # If no Yahoo Finance news, fall back to local API
        if not articles:
            logging.info(f"No Yahoo Finance news for {symbol}, trying local API...")
            articles = fetch_stock_news_for_date(symbol, date_str)
    else:
        # Try both APIs for unknown symbols
        logging.info(f"Unknown symbol {symbol}, trying both crypto and stock APIs...")
        crypto_articles = fetch_crypto_news_for_date(symbol, date_str)
        stock_articles = fetch_yahoo_news_for_date(symbol, date_str)
        
        # Combine and deduplicate articles
        all_articles = crypto_articles + stock_articles
        if all_articles:
            logging.info(f"Found {len(all_articles)} total articles for {symbol} on {date_str}")
        articles = all_articles
    
    # If still no articles, try alternative approaches
    if not articles:
        logging.info(f"No articles found for {symbol} on {date_str}, trying alternative approaches...")
        
        # Try searching with broader terms
        if is_stock(symbol):
            # For stocks, try company name variations
            alternative_terms = {
                'TSLA': 'Tesla',
                'AAPL': 'Apple',
                'MSFT': 'Microsoft',
                'GOOGL': 'Google',
                'AMZN': 'Amazon',
                'NVDA': 'NVIDIA',
                'META': 'Facebook Meta',
                'NFLX': 'Netflix',
                'AMD': 'Advanced Micro Devices',
                'INTC': 'Intel'
            }
            
            if symbol in alternative_terms:
                alt_symbol = alternative_terms[symbol]
                logging.info(f"Trying alternative term: {alt_symbol}")
                articles = fetch_yahoo_news_for_date(alt_symbol, date_str)
                if articles:
                    logging.info(f"Found {len(articles)} articles using alternative term {alt_symbol}")
    
    return articles

def analyze_sentiment_no_web_search(symbol: str, date_str: str, articles: List[Dict]) -> Tuple[str, str]:
    """
    Analyze sentiment from news articles WITHOUT web search.
    Returns: (analysis_text, sentiment)
    """
    if not openai_client:
        logging.error("OpenAI client not initialized")
        return "Error: OpenAI unavailable", "neutral"
    
    if not articles:
        return f"No news articles available for {symbol} on {date_str}.", "neutral"
    
    # If articles contain string format news, convert to dict format first
    processed_articles = []
    for article in articles:
        if isinstance(article, str):
            # Convert string news to dict format
            processed_articles.append({
                'title': 'News Summary',
                'text': article,
                'description': article,
                'source_name': 'Unknown',
                'sentiment': 'N/A'
            })
        elif isinstance(article, dict):
            processed_articles.append(article)
        else:
            logging.warning(f"Unknown news data format: {type(article)}")
    
    articles = processed_articles
    
    # Format articles for analysis
    articles_text = "\n\n".join([
        f"Article {i+1}:\n"
        f"Title: {article.get('title', 'N/A')}\n"
        f"Source: {article.get('source_name', article.get('source_id', 'Unknown'))}\n"
        f"Text: {article.get('text', article.get('description', 'No text'))[:800]}...\n"
        f"Sentiment Score: {article.get('sentiment', 'N/A')}"
        for i, article in enumerate(articles)  # Limit to top 20 articles
    ])
    
    # Step 1: Generate comprehensive analysis
    analysis_prompt = f"""
Analyze the following {symbol} news articles from {date_str} and provide a comprehensive summary.

IMPORTANT RULES:
- Base your analysis ONLY on the provided articles
- Do NOT search for additional information
- Do NOT reference current prices or future predictions
- Focus on the events and sentiment expressed in these specific articles

Articles from {date_str}:

{articles_text}

Please provide:
1. A comprehensive summary of the {symbol} news and events from these articles
2. Key themes and developments mentioned
3. Overall market sentiment based on these articles

Format your response as a cohesive narrative, objective summary, mentioning specific sources where relevant. Do not include the article numbers, and quotes etc...
"""
    
    try:
        # Get analysis without web search
        analysis_response = openai_client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": f"You are a financial news analyst. Analyze ONLY the provided articles without seeking additional information. And you have to be objective based"},
                {"role": "user", "content": analysis_prompt}
            ],
        )
        
        analysis_text = analysis_response.choices[0].message.content
        articles_text = "\n\n".join([
            f"Article {i+1}:\n"
            f"Title: {article.get('title', 'N/A')}\n"
            f"Source: {article.get('source_name', article.get('source_id', 'Unknown'))}\n"
            f"Text: {article.get('text', article.get('description', 'No text'))[:800]}...\n"
            f"Sentiment Score: {article.get('sentiment', 'N/A')}"
            for i, article in enumerate(articles[:50])  # Limit to top 20 articles
        ])
        # Step 2: Extract sentiment
        sentiment_prompt = f"""
Based on this news analysis, determine the overall market sentiment.

Analysis:
{analysis_text}

Classify the sentiment as exactly one of: bullish, bearish, or neutral

Consider:
- Positive vs negative news volume
- Institutional adoption mentions
- Regulatory developments
- Market participant actions
- Technical developments

Return your answer as a JSON object:
{{"sentiment": "bullish"}} or {{"sentiment": "bearish"}} or {{"sentiment": "neutral"}}
"""

        sentiment_response = openai_client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": "You are a sentiment classifier. Return only valid JSON."},
                {"role": "user", "content": sentiment_prompt}
            ],
            response_format={'type': 'json_object'},
        )
        
        sentiment_json = json.loads(sentiment_response.choices[0].message.content)
        sentiment = sentiment_json.get('sentiment', 'neutral').lower()
        
        # Validate sentiment
        if sentiment not in ['bullish', 'bearish', 'neutral']:
            logging.warning(f"Invalid sentiment '{sentiment}', defaulting to neutral")
            sentiment = 'neutral'
        articles_text = "\n\n".join([
            f"Article {i+1}:\n"
            f"Title: {article.get('title', 'N/A')}\n"
            f"Source: {article.get('source_name', article.get('source_id', 'Unknown'))}\n"
            f"Text: {article.get('text', article.get('description', 'No text'))[:800]}...\n"
            f"Sentiment Score: {article.get('sentiment', 'N/A')}"
            for i, article in enumerate(articles)  # Limit to top 20 articles
        ])
        # Return the complete analysis text as a single string
        return analysis_text, sentiment
        
    except Exception as e:
        logging.error(f"Error in sentiment analysis: {str(e)}")
        return "Error analyzing sentiment", "neutral"

def get_crypto_price(symbol: str, date_str: str) -> Optional[float]:
    """Get crypto closing price for a specific date from Binance"""
    try:
        date_obj = datetime.strptime(date_str, DATE_FORMAT)
        
        # Don't query future dates
        if date_obj > datetime.now():
            logging.warning(f"Cannot get price for future date {date_str}")
            return None
        
        # Convert symbol to Binance format
        binance_symbol = f"{symbol}USDT"
        
        d = datetime.strptime(date_str, DATE_FORMAT)
        start_utc = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc)
        end_utc   = start_utc + timedelta(days=1)
        start_ts  = int(start_utc.timestamp() * 1000)
        end_ts    = int(end_utc.timestamp() * 1000)

        klines = binance_client.get_historical_klines(
            binance_symbol,
            Client.KLINE_INTERVAL_1DAY,
            start_ts,
            end_ts
        )
        
        if klines and len(klines) > 0:
            closing_price = float(klines[0][4])
            return closing_price
        else:
            logging.warning(f"No crypto price data for {symbol} on {date_str}")
            return None
            
    except Exception as e:
        logging.error(f"Error fetching crypto price: {str(e)}")
        return None

def get_stock_price(symbol: str, date_str: str) -> Optional[float]:
    """Get stock closing price for a specific date using Yahoo Finance API
    If market is closed (weekends/holidays), use the previous trading day's price"""
    try:
        date_obj = datetime.strptime(date_str, DATE_FORMAT)
        target_date = date_obj.date()
        
        # Don't query future dates
        if date_obj > datetime.now():
            logging.warning(f"Cannot get price for future date {date_str}")
            return None
        
        # Use yfinance to get stock data
        ticker = yf.Ticker(symbol)
        
        # Get historical data for a broader range to find previous trading day
        start_date = date_obj - timedelta(days=10)  # Look back up to 10 days
        end_date = date_obj + timedelta(days=1)
        
        hist = ticker.history(start=start_date, end=end_date)
        
        if not hist.empty:
            # Check if we have data for the exact target date
            exact_date_data = None
            for date_index in hist.index:
                if date_index.date() == target_date:
                    exact_date_data = hist.loc[date_index]
                    break
            
            if exact_date_data is not None:
                # Found exact date data
                closing_price = float(exact_date_data['Close'])
                return closing_price
            else:
                # No data for exact date - find previous trading day
                available_dates = [date.date() for date in hist.index if date.date() <= target_date]
                
                if available_dates:
                    # Get the most recent trading day before or on target date
                    latest_available_date = max(available_dates)
                    
                    # Find the corresponding data
                    for date_index in hist.index:
                        if date_index.date() == latest_available_date:
                            closing_price = float(hist.loc[date_index, 'Close'])
                            return closing_price
                
                logging.warning(f"No previous trading day found for {symbol} around {date_str}")
                return None
        else:
            logging.warning(f"No stock price data available for {symbol} around {date_str}")
            return None
            
    except Exception as e:
        logging.error(f"Error fetching stock price for {symbol}: {str(e)}")
        return None

def check_and_fill_date_gaps(symbol: str, existing_data: Dict) -> Dict:
    """Check for date gaps and fill missing dates with real data if possible"""
    if not existing_data:
        return existing_data
    
    # Get all dates and sort them
    dates = sorted(existing_data.keys())
    if not dates:
        return existing_data
    
    first_date = datetime.strptime(dates[0], DATE_FORMAT)
    last_date = datetime.strptime(dates[-1], DATE_FORMAT)
    
    # Check for gaps and fill them
    current_date = first_date
    while current_date <= last_date:
        date_str = current_date.strftime(DATE_FORMAT)
        
        if date_str not in existing_data:
            logging.info(f"Found missing date {date_str} for {symbol}, attempting to fetch real data...")
            
            try:
                # Try to fetch real news data for missing date
                articles = fetch_news_for_date(symbol, date_str)
                
                # Check for earnings report
                earnings_report = check_earnings_report(symbol, date_str)
                
                if articles:
                    # Get real news analysis
                    news_analysis, sentiment = analyze_sentiment_no_web_search(symbol, date_str, articles)
                    logging.info(f"Successfully fetched real news data for {symbol} on {date_str}")
                else:
                    # No articles found, use placeholder
                    news_analysis = f"No news articles found for {date_str}"
                    sentiment = "neutral"
                    logging.warning(f"No news articles found for {symbol} on {date_str}")
                
                # Prepare news list
                news_list = [news_analysis]
                
                # Add earnings report if available
                if earnings_report:
                    news_list.append(earnings_report)
                    logging.info(f"Added earnings report for {symbol} on {date_str}")
                    
            except Exception as e:
                # If fetching fails, use placeholder
                news_analysis = f"Error fetching news for {date_str}: {str(e)}"
                sentiment = "neutral"
                news_list = [news_analysis]
                logging.error(f"Error fetching news for {symbol} on {date_str}: {e}")
            
            # Get SEC filings for stocks when filling gaps
            ten_k_gap, ten_q_gap = [], []
            if is_stock(symbol):
                try:
                    # Try localhost API
                    _, api_10k_gap, api_10q_gap = fetch_stock_news_and_filings_for_date(symbol, date_str)
                    ten_k_gap, ten_q_gap = api_10k_gap, api_10q_gap
                except Exception as e:
                    logging.warning(f"Error fetching SEC filings for gap date {date_str}: {e}")
                    # If API fails, keep empty lists
            
            # Fill missing date with real or placeholder data
            existing_data[date_str] = {
                "prices": None,
                "news": news_list,
                "raw_news": articles,  # Store raw news
                "momentum": sentiment,
                "future_price_diff": None,
                "10k": ten_k_gap,  # Store 10k reports separately
                "10q": ten_q_gap   # Store 10q reports separately
            }
            logging.info(f"Filled missing date {date_str} for {symbol}")
        
        current_date += timedelta(days=1)
    
    return existing_data

def update_existing_prices(symbol: str, existing_data: Dict) -> Dict:
    """Update prices for existing dates that don't have prices"""
    if not existing_data:
        return existing_data
    
    updated_count = 0
    for date_str, data in existing_data.items():
        if data.get("prices") is None:
            # Try to get price for this date
            try:
                price = get_asset_price(symbol, date_str)
                if price is not None:
                    existing_data[date_str]["prices"] = price
                    updated_count += 1
            except Exception as e:
                logging.warning(f"Failed to update price for {symbol} on {date_str}: {e}")
    
    if updated_count > 0:
        logging.info(f"Updated {updated_count} prices for {symbol}")
    
    return existing_data

def fill_future_price_diff(symbol: str, existing_data: Dict) -> Dict:
    """Fill missing future_price_diff based on next day's price"""
    if not existing_data:
        return existing_data
    
    dates = sorted(existing_data.keys())
    if len(dates) < 2:
        return existing_data
    
    # Fill future_price_diff for all dates except the last one
    for i in range(len(dates) - 1):
        current_date = dates[i]
        next_date = dates[i + 1]
        
        current_data = existing_data[current_date]
        next_data = existing_data[next_date]
        
        # If future_price_diff is missing and we have both prices
        if (current_data.get("future_price_diff") is None and 
            current_data.get("prices") is not None and 
            next_data.get("prices") is not None):
            
            future_price_diff = next_data["prices"] - current_data["prices"]
            existing_data[current_date]["future_price_diff"] = round(future_price_diff, 6)
            logging.info(f"Filled future_price_diff for {symbol} on {current_date}: {future_price_diff}")
    
    # Last date should have future_price_diff as null
    if dates:
        last_date = dates[-1]
        existing_data[last_date]["future_price_diff"] = None
    
    return existing_data

def fix_placeholder_news(symbol: str, existing_data: Dict) -> Dict:
    """Fix placeholder news data by fetching real news"""
    if not existing_data:
        return existing_data
    
    updated_count = 0
    for date_str, data in existing_data.items():
        news_content = data.get("news", [])
        
        # Check if this is placeholder data (handle both string and list formats)
        is_placeholder = False
        if isinstance(news_content, str):
            is_placeholder = ("Data not available for" in news_content or 
                            "No news articles found for" in news_content or
                            len(news_content.strip()) < 50)
        elif isinstance(news_content, list) and len(news_content) > 0:
            first_item = news_content[0]
            if isinstance(first_item, str):
                is_placeholder = ("Data not available for" in first_item or 
                                "No news articles found for" in first_item or
                                "Error processing data" in first_item or
                                len(first_item.strip()) < 50)
        
        if is_placeholder:
            logging.info(f"Attempting to fetch real news for {symbol} on {date_str}...")
            
            try:
                # Try to fetch real news data
                articles = fetch_news_for_date(symbol, date_str)
                
                # Check for earnings report
                earnings_report = check_earnings_report(symbol, date_str)
                
                if articles:
                    # Get real news analysis
                    news_analysis, sentiment = analyze_sentiment_no_web_search(symbol, date_str, articles)
                    
                    # Prepare news list
                    news_list = [news_analysis]
                    
                    # Add earnings report if available
                    if earnings_report:
                        news_list.append(earnings_report)
                        logging.info(f"Added earnings report for {symbol} on {date_str}")
                    
                    # Get SEC filings for updated news
                    ten_k_fix, ten_q_fix = [], []
                    if is_stock(symbol):
                        try:
                            # Try localhost API
                            _, api_10k_fix, api_10q_fix = fetch_stock_news_and_filings_for_date(symbol, date_str)
                            ten_k_fix, ten_q_fix = api_10k_fix, api_10q_fix
                        except Exception as e:
                            logging.warning(f"Error fetching SEC filings for news fix {date_str}: {e}")
                            # If API fails, keep empty lists
                    
                    existing_data[date_str]["news"] = news_list
                    existing_data[date_str]["raw_news"] = articles  # Update raw news
                    existing_data[date_str]["momentum"] = sentiment
                    existing_data[date_str]["10k"] = ten_k_fix
                    existing_data[date_str]["10q"] = ten_q_fix
                    updated_count += 1
                    logging.info(f"Successfully updated news for {symbol} on {date_str}")
                else:
                    logging.warning(f"Still no news articles found for {symbol} on {date_str}")
                    
            except Exception as e:
                logging.error(f"Error fetching news for {symbol} on {date_str}: {e}")
    
    if updated_count > 0:
        logging.info(f"Updated news for {updated_count} dates for {symbol}")
    
    return existing_data

def fix_news_for_symbol(symbol: str):
    """Fix placeholder news data for a specific symbol without changing other data"""
    filename = f"data/paper_trading_{symbol}.json"
    
    if not os.path.exists(filename):
        logging.warning(f"File {filename} does not exist")
        return
    
    try:
        # Load existing data
        with open(filename, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        
        logging.info(f"Fixing placeholder news for {filename}")
        
        # Fix placeholder news data
        existing_data = fix_placeholder_news(symbol, existing_data)
        
        # Sort dates and create ordered data
        sorted_dates = sorted(existing_data.keys())
        ordered_data = {}
        for date in sorted_dates:
            ordered_data[date] = existing_data[date]
        
        # Save updated data
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(ordered_data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Successfully fixed placeholder news for {filename}")
        
    except Exception as e:
        logging.error(f"Error fixing placeholder news for {filename}: {e}")

def fix_existing_asset_file(symbol: str):
    """Fix existing asset file by filling date gaps and future_price_diff"""
    filename = f"data/paper_trading_{symbol}.json"
    
    if not os.path.exists(filename):
        logging.warning(f"File {filename} does not exist")
        return
    
    try:
        # Load existing data
        with open(filename, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        
        logging.info(f"Fixing {filename} for {symbol}")
        
        # Check and fill date gaps
        existing_data = check_and_fill_date_gaps(symbol, existing_data)
        
        # Fix placeholder news data
        existing_data = fix_placeholder_news(symbol, existing_data)
        
        # Update existing prices that are missing
        existing_data = update_existing_prices(symbol, existing_data)
        
        # Fill missing future_price_diff
        existing_data = fill_future_price_diff(symbol, existing_data)
        
        # Sort dates and create ordered data
        sorted_dates = sorted(existing_data.keys())
        ordered_data = {}
        for date in sorted_dates:
            ordered_data[date] = existing_data[date]
        
        # Save fixed data
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(ordered_data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Successfully fixed {filename}")
        
    except Exception as e:
        logging.error(f"Error fixing {filename}: {e}")

def update_prices_for_symbol(symbol: str):
    """Update prices for a specific symbol without changing other data"""
    filename = f"data/paper_trading_{symbol}.json"
    
    if not os.path.exists(filename):
        logging.warning(f"File {filename} does not exist")
        return
    
    try:
        # Load existing data
        with open(filename, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        
        logging.info(f"Updating prices for {filename}")
        
        # Update existing prices that are missing
        existing_data = update_existing_prices(symbol, existing_data)
        
        # Fill missing future_price_diff
        existing_data = fill_future_price_diff(symbol, existing_data)
        
        # Sort dates and create ordered data
        sorted_dates = sorted(existing_data.keys())
        ordered_data = {}
        for date in sorted_dates:
            ordered_data[date] = existing_data[date]
        
        # Save updated data
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(ordered_data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Successfully updated prices for {filename}")
        
    except Exception as e:
        logging.error(f"Error updating prices for {filename}: {e}")

def save_asset_data_to_file(symbol: str, date_str: str, result: Dict):
    """Save individual asset data to a separate file (e.g., paper_trading_TSLA.json)"""
    filename = f"data/paper_trading_{symbol}.json"
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Load existing data if file exists
    existing_data = {}
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except Exception as e:
            logging.warning(f"Error loading existing {filename}: {e}")
    
    # Note: Data existence check is now done in get_crypto_data function
    # This function will only be called when we need to save new data
    
    # Create data in the format you want (without symbol and date fields)
    asset_data = {
        "prices": result["prices"],
        "news": result["news"],
        "raw_news": result.get("raw_news", []),  # Include raw news data
        "momentum": result["momentum"],
        "future_price_diff": result["future_price_diff"],
        "10k": result.get("10k", []),  # Store 10k reports separately
        "10q": result.get("10q", [])   # Store 10q reports separately
    }
    
    # Update with new data
    existing_data[date_str] = asset_data
    
    # Check and fill date gaps
    existing_data = check_and_fill_date_gaps(symbol, existing_data)
    
    # Update existing prices that are missing
    existing_data = update_existing_prices(symbol, existing_data)
    
    # Fill missing future_price_diff
    existing_data = fill_future_price_diff(symbol, existing_data)
    
    # Sort dates and create ordered data
    sorted_dates = sorted(existing_data.keys())
    ordered_data = {}
    for date in sorted_dates:
        ordered_data[date] = existing_data[date]
    
    # Save to file
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(ordered_data, f, indent=2, ensure_ascii=False)
        logging.info(f"Asset data saved to {filename}")
    except Exception as e:
        logging.error(f"Error saving to {filename}: {e}")

def get_asset_price(symbol: str, date_str: str) -> Optional[float]:
    """Get price for a specific asset (crypto or stock)"""
    if is_crypto(symbol):
        return get_crypto_price(symbol, date_str)
    elif is_stock(symbol):
        return get_stock_price(symbol, date_str)
    else:
        # Try both for unknown symbols
        crypto_price = get_crypto_price(symbol, date_str)
        if crypto_price is not None:
            return crypto_price
        
        stock_price = get_stock_price(symbol, date_str)
        return stock_price

def check_earnings_report(symbol: str, date_str: str) -> Optional[str]:
    """
    Check if there's an earnings report for the given symbol on the specified date.
    Returns earnings summary if found, None otherwise.
    """
    # Only check for stocks, not crypto
    if not is_stock(symbol):
        return None
    
    try:
        logging.info(f"Checking for earnings report for {symbol} on {date_str}...")
        ticker = yf.Ticker(symbol)
        
        # Get earnings dates
        earnings_dates = ticker.earnings_dates
        if earnings_dates is None or earnings_dates.empty:
            logging.info(f"No earnings dates available for {symbol}")
            return None
        
        # Convert target date to pandas datetime for comparison
        target_date = datetime.strptime(date_str, DATE_FORMAT).date()
        
        # Check if any earnings date matches the target date
        earnings_on_date = None
        for date_idx in earnings_dates.index:
            earnings_date = date_idx.date()
            if earnings_date == target_date:
                earnings_on_date = earnings_dates.loc[date_idx]
                break
        
        if earnings_on_date is None:
            logging.info(f"No earnings report found for {symbol} on {date_str}")
            return None
        
        # Get additional earnings information
        earnings_info = {
            'symbol': symbol,
            'date': date_str,
            'estimated_eps': getattr(earnings_on_date, 'EPS Estimate', 'N/A'),
            'reported_eps': getattr(earnings_on_date, 'Reported EPS', 'N/A'),
            'surprise_percent': getattr(earnings_on_date, 'Surprise(%)', 'N/A')
        }
        
        # Try to get quarterly earnings for more context
        try:
            quarterly_earnings = ticker.quarterly_earnings
            if quarterly_earnings is not None and not quarterly_earnings.empty:
                # Get the most recent earnings
                latest_earnings = quarterly_earnings.iloc[0] if len(quarterly_earnings) > 0 else None
                if latest_earnings is not None:
                    earnings_info['revenue'] = getattr(latest_earnings, 'Revenue', 'N/A')
                    earnings_info['earnings'] = getattr(latest_earnings, 'Earnings', 'N/A')
        except Exception as e:
            logging.debug(f"Could not get quarterly earnings for {symbol}: {e}")
        
        # Format earnings report
        earnings_report = f"""
EARNINGS REPORT - {symbol} ({date_str})
{'='*50}
Date: {date_str}
Symbol: {symbol}
Estimated EPS: {earnings_info.get('estimated_eps', 'N/A')}
Reported EPS: {earnings_info.get('reported_eps', 'N/A')}
Surprise %: {earnings_info.get('surprise_percent', 'N/A')}
"""
        
        if 'revenue' in earnings_info:
            earnings_report += f"Revenue: {earnings_info['revenue']}\n"
        if 'earnings' in earnings_info:
            earnings_report += f"Earnings: {earnings_info['earnings']}\n"
        
        earnings_report += f"""
This earnings report was released on {date_str} and may have significant impact on {symbol} stock price and market sentiment. The market typically reacts to earnings surprises (both positive and negative) and forward guidance provided by the company.
"""
        
        logging.info(f"Found earnings report for {symbol} on {date_str}")
        return earnings_report.strip()
        
    except Exception as e:
        logging.error(f"Error checking earnings for {symbol} on {date_str}: {str(e)}")
        return None

def get_crypto_data(symbol: str, date_str: str) -> Dict:
    """
    Get daily crypto/stock news and price for a specific symbol and date.
    
    Args:
        symbol (str): Asset symbol (e.g., 'BTC', 'ETH', 'TSLA', 'AAPL')
        date_str (str): Date in YYYY-MM-DD format
        
    Returns:
        Dict: Dictionary with news and price data
        {
            "symbol": str,
            "date": str,
            "prices": float,
            "news": List[str],
            "momentum": str,
            "future_price_diff": None
        }
    """
    print(f"📅 Getting {symbol} data for {date_str}...")
    
    # Check if data already exists before doing any expensive operations
    json_file = f"data/paper_trading_{symbol}.json"
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            if date_str in existing_data:
                existing_entry = existing_data[date_str]
                if (existing_entry.get("prices") is not None and 
                    isinstance(existing_entry.get("news"), list) and  # Ensure news is in list format
                    existing_entry.get("news") and  # Ensure list is not empty
                    existing_entry.get("momentum") and
                    len(existing_entry.get("news", [])[0]) > 10 and  # Check first news item length
                    "No news articles available" not in existing_entry.get("news", [])[0] and  # Ensure not placeholder news
                    "Error processing data" not in existing_entry.get("news", [])[0]):  # Ensure not error message
                    
                    print(f"⏭️  Data for {symbol} on {date_str} already exists and is complete, skipping...")
                    logging.info(f"Data for {symbol} on {date_str} already exists and is complete, skipping...")
                    
                    # Return existing data
                    return {
                        "symbol": symbol,
                        "date": date_str,
                        "prices": existing_entry["prices"],
                        "news": existing_entry["news"],
                        "momentum": existing_entry["momentum"],
                        "future_price_diff": existing_entry.get("future_price_diff")
                    }
        except Exception as e:
            logging.warning(f"Error checking existing data: {e}")
    
    try:
        # Step 1: Fetch news with smart symbol detection
        articles = fetch_news_for_date(symbol, date_str)
        
        # Step 2: Check for earnings report
        earnings_report = check_earnings_report(symbol, date_str)
        
        # Step 2.5: Fetch SEC 10K/10Q filings for stocks
        ten_k_filings, ten_q_filings = [], []
        if is_stock(symbol):
            try:
                # Try to get 10K/10Q from localhost API
                api_news, api_10k, api_10q = fetch_stock_news_and_filings_for_date(symbol, date_str)
                ten_k_filings, ten_q_filings = api_10k, api_10q
                logging.info(f"Got {len(ten_k_filings)} 10-K and {len(ten_q_filings)} 10-Q from localhost API")
            except Exception as e:
                logging.warning(f"Error fetching SEC filings for {symbol}: {e}")
                # If API fails, keep empty lists
        
        # Step 3: Analyze sentiment (no web search)
        news_analysis, sentiment = analyze_sentiment_no_web_search(symbol, date_str, articles)
        
        # Step 4: Get asset price
        asset_price = get_asset_price(symbol, date_str)
        
        # Prepare news list
        news_list = [news_analysis]
        
        # Add earnings report if available
        if earnings_report:
            news_list.append(earnings_report)
            print(f"📊 Added earnings report for {symbol} on {date_str}")
        
        # Create result
        result = {
            "symbol": symbol,
            "date": date_str,
            "prices": asset_price,
            "news": news_list,
            "raw_news": articles,  # Store all original news articles
            "momentum": sentiment,
            "future_price_diff": None,
            "10k": ten_k_filings,  # Store 10k reports separately
            "10q": ten_q_filings   # Store 10q reports separately
        }
        
        # Save to individual asset file (matching TSLA_market_data.json format)
        save_asset_data_to_file(symbol, date_str, result)
        
        price_display = f"${asset_price:,.2f}" if asset_price is not None else "N/A"
        print(f"✅ Completed {symbol} for {date_str}: {price_display}, Sentiment: {sentiment}")
        print(f"📰 News items: {len(news_list)} (analysis + {'earnings' if earnings_report else 'no earnings'})")
        print(f"🔍 Articles processed: {len(articles)}")
        
        return result
        
    except Exception as e:
        logging.error(f"Error processing {symbol} for {date_str}: {str(e)}")
        print(f"❌ Error processing {symbol} for {date_str}: {str(e)}")
        return {
            "symbol": symbol,
            "date": date_str,
            "prices": None,
            "news": ["Error processing data"],
            "raw_news": [],  # Empty raw news for error case
            "momentum": "neutral",
            "future_price_diff": None,
            "10k": [],  # Store 10k reports separately
            "10q": []   # Store 10q reports separately
        }

def main():
    """Example usage of get_crypto_data function"""
    import sys
    
    if len(sys.argv) == 3:
        symbol = sys.argv[1].upper()
        date_str = sys.argv[2]
        
        # Validate date format
        try:
            datetime.strptime(date_str, DATE_FORMAT)
        except ValueError:
            print(f"Invalid date format. Please use YYYY-MM-DD format.")
            print(f"Usage: python get_daily_action.py <SYMBOL> <DATE>")
            print(f"Examples:")
            print(f"  python get_daily_action.py BTC 2025-08-15")
            print(f"  python get_daily_action.py TSLA 2025-08-15")
            return
        
        result = get_crypto_data(symbol, date_str)
        print(f"\n📊 Result for {symbol} on {date_str}:")
        print(json.dumps(result, indent=2))
        
    elif len(sys.argv) == 2:
        if sys.argv[1].upper() == 'FIX':
            # Fix all existing asset files
            print("🔧 Fixing all existing asset files...")
            all_symbols = CRYPTO_SYMBOLS + STOCK_SYMBOLS
            for symbol in all_symbols:
                print(f"Fixing {symbol}...")
                fix_existing_asset_file(symbol)
            print("✅ All files fixed!")
        elif sys.argv[1].upper() == 'UPDATE_PRICES':
            # Update prices for all existing asset files
            print("💰 Updating prices for all existing asset files...")
            all_symbols = CRYPTO_SYMBOLS + STOCK_SYMBOLS
            for symbol in all_symbols:
                print(f"Updating prices for {symbol}...")
                update_prices_for_symbol(symbol)
            print("✅ All prices updated!")
        elif sys.argv[1].upper() == 'FIX_NEWS':
            # Fix placeholder news data for all existing asset files
            print("📰 Fixing placeholder news data for all existing asset files...")
            all_symbols = CRYPTO_SYMBOLS + STOCK_SYMBOLS
            for symbol in all_symbols:
                print(f"Fixing news for {symbol}...")
                fix_news_for_symbol(symbol)
            print("✅ All placeholder news fixed!")
        else:
            symbol = sys.argv[1].upper()
            # Default: get today's data
            today = datetime.now().strftime(DATE_FORMAT)
            print(f"No date specified, using today: {today}")
            result = get_crypto_data(symbol, today)
            print(f"\n📊 Result for {symbol} on {today}:")
            print(json.dumps(result, indent=2))
        
    else:
        print("Enhanced Daily Crypto and Stock News & Price Fetcher")
        print("=" * 55)
        print("Usage:")
        print("  python get_daily_action.py <SYMBOL> <DATE>")
        print("  python get_daily_action.py <SYMBOL>")
        print("  python get_daily_action.py FIX")
        print("  python get_daily_action.py UPDATE_PRICES")
        print("\nSupported Symbols:")
        print("  Cryptocurrencies: BTC, ETH, ADA, SOL, DOT, LINK, UNI, MATIC, AVAX, ATOM")
        print("  Stocks: TSLA, AAPL, MSFT, GOOGL, AMZN, NVDA, META, NFLX, AMD, INTC")
        print("\nExamples:")
        print("  python get_daily_action.py BTC 2025-08-15")
        print("  python get_daily_action.py TSLA 2025-08-15")
        print("  python get_daily_action.py ETH")  # Gets today's data
        print("  python get_daily_action.py FIX")  # Fix existing files
        print("  python get_daily_action.py UPDATE_PRICES")  # Update prices only

if __name__ == "__main__":
    main()
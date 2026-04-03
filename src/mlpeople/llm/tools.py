from typing import List
from langchain.tools import tool
from serpapi import GoogleSearch

import os


# create a tool that directly call SerpAPI’s Google Scholar API
# SerpAPI docs for Google Scholar engine:
# https://serpapi.com/google-scholar-api
# Google scholar search
# https://scholar.google.com/
@tool
def scholar_search(query: str) -> List[dict]:
    """
    Search Google Scholar for recent scientific publications.

    Returns a list of publications with:
    - title
    - authors
    - snippet (short description)
    """
    params = {
        "engine": "google_scholar",  # Other engines: "google", "google_news", "bing", etc.
        "q": query,  # search query string
        "hl": "en",  # ensures English titles/snippets
        "num": 5,  # number of results - exactly 5 publications
        "sort": "date",  # Sort results by newest publications - add latest publications to results
        "api_key": os.getenv("SERPAPI_API_KEY"),  # authenticates the request
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    publications = []
    for item in results.get("organic_results", []):
        title = item.get("title")
        authors = item.get("publication_info", {}).get("summary")
        snippet = item.get("snippet")

        publications.append(
            f"Title: {title}\nAuthors: {authors}\nDescription: {snippet}\n"
        )

    return "\n".join(publications)


# create few other tools for web-search
@tool
def web_search(query: str) -> str:
    """Search the web using Google via SerpAPI.."""
    params = {
        "q": query,
        "api_key": os.getenv("SERPAPI_API_KEY"),
        "engine": "google",
        "num": 3,
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    snippets = [r.get("snippet", "") for r in results.get("organic_results", [])[:3]]
    return "\n".join(snippets)


@tool
def bing_search(query: str) -> str:
    """Search the web using Bing via SerpAPI."""
    params = {
        "engine": "bing",  # Bing engine
        "q": query,  # Search query
        "api_key": os.getenv("SERPAPI_API_KEY"),
        "num": 3,  # Top 3 results
        "hl": "en",  # Language
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    snippets = [r.get("snippet", "") for r in results.get("organic_results", [])[:3]]
    return "\n".join(snippets)


@tool
def google_news_search(query: str) -> str:
    """Search Google News for recent articles via SerpAPI."""
    params = {
        "engine": "google_news",  # Google News engine
        "q": query,
        "api_key": os.getenv("SERPAPI_API_KEY"),
        "num": 5,  # Top 5 articles
        "hl": "en",
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    snippets = []

    # Google News returns a list of news articles
    for article in results.get("news_results", [])[:5]:
        title = article.get("title", "")
        source = article.get("source", "")
        snippet = article.get("snippet", "")
        snippets.append(f"{title} ({source}): {snippet}")

    return "\n".join(snippets)


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple math expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


@tool
def python_interpreter(code: str) -> str:
    """Execute Python code using print() to return results."""
    local_vars = {}
    try:
        exec(code, {}, local_vars)
        return str(local_vars)
    except Exception as e:
        return f"Error: {e}"

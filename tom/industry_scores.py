from datetime import datetime, timedelta
from textwrap import dedent
from pathlib import Path
import ast
import json
import logging
import sys
import re
import traceback
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import yaml

#TODO Use data classes for some type predictability
#TODO unit tests
#TODO save the payloads to files for unit testing & debugging

logging.basicConfig(level=logging.INFO, format="%(lineno)d\t%(message)s")


def mk_datestring(date):
    """Create a datestring for use in filenames.
    Non-side-effecting"""
    return date.strftime("%Y_%m_%d")


def extract_claude_json(ranking_payload) -> list[dict]:
    """Claude's response contains a JSON payload that needs to be extracted
    from unsanitized text. This function extracts the JSON payload from the
    blob. Non-side-effecting function."""
    pattern = r'```json\s*(.*?)\s*```'

    #extract JSON using regex
    json_match = re.search(pattern, ranking_payload, re.DOTALL)

    if json_match:
        extracted_json = json_match.group(1)
        return extracted_json
    else:
        return []


def fetch_configfile(filepath="/Users/mruggiero/bin/anthropic.yml") -> list[str]:
    """Fetch the configuration file for the API keys. Side-effecting"""
    with open(filepath, mode="rt", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    return payload


def get_article_text(url) -> str:
    """Retrieve the text of an article from a URL. Side-effecting"""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        article_text = ' '.join([p.get_text() for p in soup.find_all('p')])
        return article_text
    except:
        return "Error retrieving article text."


def get_stock_data(ticker, years) -> tuple:
    """Retrieve stock data for a given ticker over a specified number of years.
    Side-effecting"""
    end_date = datetime.now().date()
    days = years * 365
    start_date = end_date - timedelta(days=days)

    stock = yf.Ticker(ticker)

    # historical price data
    hist_data = stock.history(start=start_date, end=end_date)

    # balance sheet
    balance_sheet = stock.balance_sheet

    # financial statements
    financials = stock.financials

    # news articles
    news = stock.news

    return (hist_data, balance_sheet, financials, news)


def get_sentiment_analysis(ticker, news) -> str:
    """Retrieve sentiment analysis for a given ticker based on news articles.
    Side-effecting"""

    prompt_template = '''You are a sentiment analysis assistant. Analyze the
    sentiment of the given news articles for {ticker} and provide a summary of
    the overall sentiment and any notable changes over time. Be measured and
    discerning. You are a skeptical investor.'''
    system_prompt = dedent(prompt_template).format(ticker=ticker)

    news_text = ""
    for article in news:
        article_text = get_article_text(article['link'])
        timestamp = datetime.fromtimestamp(article['providerPublishTime']).strftime("%Y-%m-%d")
        news_text += f"\n\n---\n\nDate: {timestamp}\nTitle: {article['title']}\nText: {article_text}"

    content_template = '''News articles for {ticker}: {news_text} ---- Provide
    a summary of the overall sentiment and any notable changes over time.'''

    messages = [
        {"role": "user", "content": dedent(content_template).format(ticker=ticker,
                                                                    news_text=news_text)}
    ]

    model = 'claude-3-haiku-20240307'
    max_tokens = 2000
    temperature = 0.5
    response = fetch_claude_response(model,
                                     max_tokens,
                                     temperature,
                                     system_prompt,
                                     messages)
    try:
        response_text = response.json()['content'][0]['text']
    except KeyError as e:
        logging.error(f"Error retrieving sentiment analysis for {ticker}: {traceback.format_exc()}")
        response_text = 'N/A'

    return response_text


def get_analyst_ratings(ticker, artifact_directory) -> str:
    """Retrieve analyst ratings for a given ticker. Side-effecting"""
    stock = yf.Ticker(ticker)
    recommendations = stock.recommendations
    if recommendations is None or recommendations.empty:
        return 'No analyst ratings available.'
    logging.info(f"recommendations: {recommendations}")

    # let's save the recommendation
    datestring = mk_datestring(datetime.now())
    filepath = artifact_directory / f"{datestring}_{ticker}_analyst_ratings.csv"
    recommendations.to_csv(filepath, encoding='utf-8', index=False)

    last_period = recommendations.iloc[-1]
    parsed = {
              'strong buy': last_period.get('strongBuy', 0),
              'buy': last_period.get('buy', 0),
              'hold': last_period.get('hold', 0),
              'sell': last_period.get('sell', 0),
              'strong sell': last_period.get('strongSell', 0)
              }

    # make a string of the parsed data
    analysis = ', '.join([f'{k} = {v}' for k, v in parsed.items()])
    rating_summary = f"Latest analyst rating for {ticker}: {analysis}"

    return rating_summary


def get_industry_analysis(ticker, industry) -> str:
    """Retrieve an analysis of the industry and sector for a given ticker.
    Side-effecting"""
    # TODO update to use search to find recent data!!

    stock = yf.Ticker(ticker)
    sector = "N/A"
    try:
        industry = stock.info['industry']
        sector = stock.info['sector']
    except KeyError as e:
        logging.error(f"Error retrieving industry and sector for {ticker}: {traceback.format_exc()}")

    prompt_template = '''You are an industry analysis assistant. Provide an
    analysis of the {industry} industry and {sector} sector, including trends,
    growth prospects, regulatory changes, and competitive landscape. Be
    measured and discerning. Truly think about the positives and negatives of
    the stock. Be sure of your analysis. You are a skeptical investor.'''
    system_prompt = dedent(prompt_template).format(industry=industry, sector=sector)

    messages = [
        {"role": "user", "content": f"Provide an analysis of the {industry} industry and {sector} sector."},
    ]

    model = 'claude-3-haiku-20240307'
    max_tokens = 2000
    temperature = 0.5
    response = fetch_claude_response(model,
                                     max_tokens,
                                     temperature,
                                     system_prompt,
                                     messages)
    try:
        response_text = response.json()['content'][0]['text']
    except KeyError as e:
        logging.error(f"Error retrieving industry analysis for {ticker}: {traceback.format_exc()}")
        response_text = 'N/A'

    return response_text


def get_final_analysis(ticker,
                       comparisons,
                       sentiment_analysis,
                       analyst_ratings,
                       industry_analysis) -> str:
    """Generate a final analysis for a given ticker based on the provided data.
    Side-effecting"""

    prompt_template = '''You are a financial analyst providing a final
    investment recommendation for {ticker} based on the given data and
    analyses. Be measured and discerning. Truly think about the positives and
    negatives of the stock. Be sure of your analysis. You are a skeptical
    investor.'''
    system_prompt = dedent(prompt_template).format(ticker=ticker)

    comparative_analysis = json.dumps(comparisons, indent=2)

    content_template = '''Ticker: {ticker}
    Comparative Analysis: {comparative_analysis} Sentiment
    Analysis: {sentiment_analysis} Analyst Ratings: {analyst_ratings}
    Industry Analysis: {industry_analysis}
    Based on the provided data and analyses, please provide a comprehensive
    investment analysis and recommendation for {ticker}.
    Consider the company's financial strength, growth prospects,
    competitive position, and potential risks. Provide a clear and concise
    recommendation on whether to buy, hold, or sell the stock, along with
    supporting rationale.'''
    content = dedent(content_template).format(ticker=ticker,
                                              comparative_analysis=comparative_analysis,
                                              sentiment_analysis=sentiment_analysis,
                                              analyst_ratings=analyst_ratings,
                                              industry_analysis=industry_analysis)
    # logging.info(f"content: {content}")

    messages = [{ "role": "user",
                "content": content}]

    model = 'claude-3-opus-20240229'
    max_tokens = 3000
    temperature = 0.5
    response = fetch_claude_response(model,
                                     max_tokens,
                                     temperature,
                                     system_prompt,
                                     messages)
    try:
        response_text = response.json()['content'][0]['text']
    except KeyError as e:
        logging.error(f"Error retrieving final analysis for {ticker}: {traceback.format_exc()}")
        response_text = 'N/A'
    return response_text


def generate_ticker_ideas(industry, ticker_count=5) -> list[str]:
    """Generate a list of ticker symbols for major companies in a given industry. Side-effecting"""

    prompt_template = '''You are a financial analyst assistant. Generate a list
    of {ticker_count} ticker symbols for major companies in the {industry} industry, as a
    Python-parseable list.'''
    system_prompt = dedent(prompt_template).format(industry=industry,
                                                   ticker_count=ticker_count)

    content_template = '''Please provide a list of {ticker_count} ticker symbols for major
    companies in the {industry} industry as a Python-parseable list. Only
    respond with the list, no other text."'''

    messages = [
        {"role": "user",
         "content": dedent(content_template).format(industry=industry,
                                                    ticker_count=ticker_count)}
    ]
    model = 'claude-3-haiku-20240307'
    max_tokens = 200
    temperature = 0.5
    response = fetch_claude_response(model,
                                     max_tokens,
                                     temperature,
                                     system_prompt,
                                     messages)
    ticker_list = []
    try:
        response_text = response.json()['content'][0]['text']
        ticker_list = ast.literal_eval(response_text)
    except KeyError as e:
        logging.error(f"Error retrieving ticker ideas for {industry}: {traceback.format_exc()}")

    return [ticker.strip() for ticker in ticker_list]


def get_current_price(ticker) -> str:
    """Retrieve the current price of a stock. Side-effecting"""
    stock = yf.Ticker(ticker)
    try:
        data = stock.history(period='1d', interval='1m')
        logging.info(f"data: {data}")

        last_row = data.iloc[-1]
        closing = last_row['Close']
        return closing
    except IndexError as e:
        logging.error(f"Error retrieving current price for {ticker}: {traceback.format_exc()}")
        return "N/A"


def rank_companies(industry, analyses, prices) -> str:
    """Rank companies based on analyses and prices. Side-effecting"""

    template = '''You are a financial analyst providing a ranking of companies
    in the {industry} industry based on their investment potential. Be
    discerning and sharp. Truly think about whether a stock is valuable or not.
    You are a skeptical investor.'''
    system_prompt = dedent(template).format(industry=industry)

    analysis_text = "\n\n".join(
        f"Ticker: {ticker}\nCurrent Price: {prices.get(ticker, 'N/A')}\nAnalysis:\n{analysis}"
        for ticker, analysis in analyses.items()
    )
    content_template = '''Industry: {industry} Company
    Analyses: {analysis_text} Based on the provided analyses, please rank the
    companies from most attractive to least attractive for investment.  Provide
    a brief rationale for your ranking. In each rationale, include the current
    price (if available) and a price target. Please return the ticker, ranking,
    price, target price and rationale formatted as JSON.'''
    content = content_template.format(industry=industry, analysis_text=analysis_text)
    # logging.info(f"content: {content}")

    messages = [{ "role": "user", "content": content }]

    model = 'claude-3-opus-20240229'
    max_tokens = 3000
    temperature = 0.5
    response = fetch_claude_response(model,
                                     max_tokens,
                                     temperature,
                                     system_prompt,
                                     messages)

    try:
        response_text = response.json()['content'][0]['text']
    except KeyError as e:
        logging.error(f"Error retrieving company ranking for {industry}: {traceback.format_exc()}")
        response_text = 'N/A'

    return response_text


def fetch_claude_response(model,
                          max_tokens,
                          temperature,
                          system_prompt,
                          messages) -> requests.models.Response:
    """Fetch a response from the Claude API. Side-effecting"""
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    data = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system_prompt,
        "messages": messages,
    }
    url = "https://api.anthropic.com/v1/messages"
    response = requests.post(url,
                             headers=headers,
                             json=data)
    return response


def json_to_csv(json_payload, filename) -> None:
    """Convert JSON payload to a CSV file. Side-effecting"""
    df = pd.read_json(json_payload)
    df.to_csv(filename, encoding='utf-8', index=False)


def main(industry, tickers, artifact_directory, years=1, debugging=False):
    """Main function to analyze an industry and rank companies based on the
    analysis. Side-effecting"""
    analyses = {}
    prices = {}
    for ticker in tickers:
        try:
            hist_data, balance_sheet, financials, news = get_stock_data(ticker, years)

            sentiment_analysis = get_sentiment_analysis(ticker, news)
            logging.info(f"sentiment_analysis: {sentiment_analysis}")

            analyst_ratings = get_analyst_ratings(ticker, artifact_directory)
            logging.info(f"analyst_ratings: {analyst_ratings}")

            industry_analysis = get_industry_analysis(ticker, industry)
            logging.info(f"industry_analysis: {industry_analysis}")

            final_analysis = get_final_analysis(ticker,
                                                {},
                                                sentiment_analysis,
                                                analyst_ratings,
                                                industry_analysis)
            logging.info(f"final_analysis: {final_analysis}")

            analyses[ticker] = final_analysis
            prices[ticker] = get_current_price(ticker)

        except Exception as e:
            logging.error(f"Error analyzing {ticker}: {traceback.format_exc()}")
            sys.exit(1)

    ranking = rank_companies(industry, analyses, prices)
    logging.info(f"\nRanking of Companies in {industry}:")
    logging.info(ranking)

    datestring = mk_datestring(datetime.now())
    sanitized_industry = industry.replace(" ", "_")
    filename = f"{datestring}_{sanitized_industry}_analysis_.csv"
    filepath = artifact_directory / filename
    extracted_json = extract_claude_json(ranking)

    df = pd.read_json(extracted_json)
    df.to_csv(filepath, encoding='utf-8', index=False)


if __name__ == "__main__":
    config = fetch_configfile()

    # TODO make this not a global
    ANTHROPIC_API_KEY = config['api_keys']['anthropic']
    CSV_PATH = Path(config['filepaths']['csv_directory'])

    DEBUGGING = False
    if DEBUGGING:
        industry = 'pharmaceutic'
        tickers = ['MRK']
    else:
        # User input
        industry = input("Enter the industry to analyze: ")

        # Generate ticker ideas for the industry
        tickers = generate_ticker_ideas(industry)

    logging.info(f"\nTicker Ideas for {industry} Industry:")
    logging.info(", ".join(tickers))

    main(industry, tickers, CSV_PATH, years=1, debugging=DEBUGGING)

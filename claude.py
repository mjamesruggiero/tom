from datetime import datetime, timedelta
from textwrap import dedent
import ast
import traceback
import json
import logging
import sys
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import yaml

#TODO Add error handling for API requests


logging.basicConfig(level=logging.INFO, format="%(lineno)d\t%(message)s")


def fetch_configfile(filepath="/Users/mruggiero/bin/anthropic.yml") -> list[str]:
    with open(filepath, mode="rt", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    return payload


def get_article_text(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        article_text = ' '.join([p.get_text() for p in soup.find_all('p')])
        return article_text
    except:
        return "Error retrieving article text."


def get_stock_data(ticker, years):
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=years*365)

    stock = yf.Ticker(ticker)

    # Retrieve historical price data
    hist_data = stock.history(start=start_date, end=end_date)

    # Retrieve balance sheet
    balance_sheet = stock.balance_sheet

    # Retrieve financial statements
    financials = stock.financials

    # Retrieve news articles
    news = stock.news

    return hist_data, balance_sheet, financials, news


def get_claude_comps_analysis(ticker, hist_data, balance_sheet, financials, news):
    prompt_template = '''You are a financial analyst assistant. Analyze the
    given data for {ticker} and suggest a few comparable companies to consider.
    Do so in a Python-parseable list.'''
    system_prompt = dedent(prompt_template).format(ticker=ticker)

    news = ""

    for article in news:
        article_text = get_article_text(article['link'])
        news = news + f"\n\n---\n\nTitle: {article['title']}\nText: {article_text}"

    hist_data = hist_data.tail().to_string()
    balance_sheet = balance_sheet.to_string()
    financials = financials.to_string()
    news = news.strip()

    content_template = '''Historical price
    data: {hist_data}  Balance
    Sheet: {balance_sheet}  Financial
    Statements: {financials}  News
    articles: {news}  ----  Now, suggest a few comparable
    companies to consider, in a Python-parseable list. Return nothing but the
    list. Make sure the companies are in the form of their tickers.'''
    messages = [
        {"role": "user", "content": dedent(content_template).format(hist_data=hist_data,
                                                                    balance_sheet=balance_sheet,
                                                                    financials=financials,
                                                                    news=news)}
    ]


    model = 'claude-3-haiku-20240307'
    max_tokens = 2000
    temperature = 0.5
    response = fetch_claude_response(model,
                                     max_tokens,
                                     temperature,
                                     system_prompt,
                                     messages)
    response_text = response.json()['content'][0]['text']

    return ast.literal_eval(response_text)


def compare_companies(main_ticker, main_data, comp_ticker, comp_data):
    prompt_template = '''You are a financial analyst assistant. Compare the
    data of {main_ticker} against {comp_ticker} and provide a detailed
    comparison, like a world-class analyst would. Be measured and discerning.
    Truly think about the positives and negatives of each company. Be sure of
    your analysis. You are a skeptical investor.'''

    system_prompt = dedent(prompt_template).format(main_ticker=main_ticker, comp_ticker=comp_ticker)

    historical_data = main_data['hist_data'].tail().to_string()
    balance_sheet = main_data['balance_sheet'].to_string()
    financials = main_data['financials'].to_string()
    comparison_data = comp_data['hist_data'].tail().to_string()
    comp_balance_sheet = comp_data['balance_sheet'].to_string()
    comp_financials = comp_data['financials'].to_string()

    content_template = '''Data for {main_ticker}:  Historical price
    data: {historical_data} Balance Sheet: {balance_sheet} Financial
    Statements: {financials} ---- Data for {comp_ticker}: Historical price
    data: {comparison_data} Balance Sheet: {comp_balance_sheet} Financial
    Statements: {comp_financials} ---- Now, provide a detailed comparison of
    {main_ticker} against {comp_ticker}. Explain your
    thinking very clearly.'''

    messages = [
            {"role": "user",
             "content": dedent(content_template).format(main_ticker=main_ticker,
                                                        historical_data=historical_data,
                                                        balance_sheet=balance_sheet,
                                                        financials=financials,
                                                        comp_ticker=comp_ticker,
                                                        comparison_data=comparison_data,
                                                        comp_balance_sheet=comp_balance_sheet,
                                                        comp_financials=comp_financials)
             }
            ]

    model = 'claude-3-haiku-20240307'
    max_tokens = 3000
    temperature = 0.5
    response_text = fetch_claude_response(model,
                                          max_tokens,
                                          temperature,
                                          system_prompt,
                                          messages)

    # return json.loads(response_text)
    return response_text


def get_sentiment_analysis(ticker, news):
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
    response_text = response.json()['content'][0]['text']

    return response_text


def get_analyst_ratings(ticker):
    stock = yf.Ticker(ticker)
    recommendations = stock.recommendations
    if recommendations is None or recommendations.empty:
        return "No analyst ratings available."

    latest_rating = recommendations.iloc[-1]

    firm = latest_rating.get('Firm', 'N/A')
    to_grade = latest_rating.get('To Grade', 'N/A')
    action = latest_rating.get('Action', 'N/A')

    rating_summary = f"Latest analyst rating for {ticker}:\nFirm: {firm}\nTo Grade: {to_grade}\nAction: {action}"

    return rating_summary


def get_industry_analysis(ticker, industry):
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
    response_text = response.json()['content'][0]['text']

    return response_text


def get_final_analysis(ticker,
                       comparisons,
                       sentiment_analysis,
                       analyst_ratings,
                       industry_analysis):

    prompt_template = '''You are a financial analyst providing a final
    investment recommendation for {ticker} based on the given data and
    analyses. Be measured and discerning. Truly think about the positives and
    negatives of the stock. Be sure of your analysis. You are a skeptical
    investor.'''
    system_prompt = dedent(prompt_template).format(ticker=ticker)

    comparative_analysis = json.dumps(comparisons, indent=2)

    content_template = '''Ticker: {ticker} Comparative
    Analysis: {comparative_analysis} Sentiment
    Analysis: {sentiment_analysis} Analyst
    Ratings: {analyst_ratings} Industry
    Analysis: {industry_analysis} Based on the provided data and analyses,
    please provide a comprehensive investment analysis and recommendation for
    {ticker}. Consider the company's financial strength, growth prospects,
    competitive position, and potential risks. Provide a clear and concise
    recommendation on whether to buy, hold, or sell the stock, along with
    supporting rationale.'''
    content = dedent(content_template).format(ticker=ticker,
                                                    comparative_analysis=comparative_analysis,
                                                    sentiment_analysis=sentiment_analysis,
                                                    analyst_ratings=analyst_ratings,
                                                    industry_analysis=industry_analysis)
    # logging.info(f"content: {content}")

    messages = [
            {
                "role": "user",
                "content": content
                }
            ]

    model = 'claude-3-opus-20240229'
    max_tokens = 3000
    temperature = 0.5
    response = fetch_claude_response(model,
                                     max_tokens,
                                     temperature,
                                     system_prompt,
                                     messages)
    response_text = response.json()['content'][0]['text']
    return response_text


def generate_ticker_ideas(industry):

    prompt_template = '''You are a financial analyst assistant. Generate a list
    of 5 ticker symbols for major companies in the {industry} industry, as a
    Python-parseable list.'''
    system_prompt = dedent(prompt_template).format(industry=industry)

    content_template = '''Please provide a list of 5 ticker symbols for major
    companies in the {industry} industry as a Python-parseable list. Only
    respond with the list, no other text."'''

    messages = [
        {"role": "user",
         "content": dedent(content_template).format(industry=industry)}
    ]

    model = 'claude-3-haiku-20240307'
    max_tokens = 200
    temperature = 0.5
    response = fetch_claude_response(model,
                                     max_tokens,
                                     temperature,
                                     system_prompt,
                                     messages)
    response_text = response.json()['content'][0]['text']

    ticker_list = ast.literal_eval(response_text)
    return [ticker.strip() for ticker in ticker_list]


def get_current_price(ticker):
    stock = yf.Ticker(ticker)
    try:
        data = stock.history(period='1d', interval='1m')
        # logging.info(f"data: {data}")
        last_row = data.iloc[-1]
        closing = last_row['Close']
        return closing
    except IndexError as e:
        logging.error(f"Error retrieving current price for {ticker}: {traceback.format_exc()}")
        return "N/A"


def rank_companies(industry, analyses, prices):
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
    Analyses: {analysis_text} Based on the provided analyses, please rank
    the companies from most attractive to least attractive for investment.
    Provide a brief rationale for your ranking. In each rationale, include the
    current price (if available) and a price target.'''
    content = content_template.format(industry=industry, analysis_text=analysis_text)
    # logging.info(f"content: {content}")

    messages = [
            {
                "role": "user",
                "content": content
                },
            ]

    model = 'claude-3-opus-20240229'
    max_tokens = 3000
    temperature = 0.5
    response = fetch_claude_response(model,
                                     max_tokens,
                                     temperature,
                                     system_prompt,
                                     messages)

    response_text = response.json()['content'][0]['text']
    return response_text


def fetch_claude_response(model, max_tokens, temperature, system_prompt, messages):
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
    response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data)
    return response


if __name__ == "__main__":
    config = fetch_configfile()
    ANTHROPIC_API_KEY = config['api_keys']['anthropic']

    DEBUGGING = False
    if DEBUGGING:
        industry = 'fashion'
        tickers = ['CPRI']
    else:
        # User input
        industry = input("Enter the industry to analyze: ")

        # Generate ticker ideas for the industry
        tickers = generate_ticker_ideas(industry)

    print(f"\nTicker Ideas for {industry} Industry:")
    print(", ".join(tickers))

    # Perform analysis for each company
    years = 1 # int(input("Enter the number of years for analysis: "))
    analyses = {}
    prices = {}
    for ticker in tickers:
        try:
            print(f"\nAnalyzing {ticker}...")
            hist_data, balance_sheet, financials, news = get_stock_data(ticker, years)
            main_data = {
                    'hist_data': hist_data,
                    'balance_sheet': balance_sheet,
                    'financials': financials,
                    'news': news
                    }
            sentiment_analysis = get_sentiment_analysis(ticker, news)
            logging.info(f"sentiment_analysis: {sentiment_analysis}")

            analyst_ratings = get_analyst_ratings(ticker)
            logging.info(f"analyst_ratings: {analyst_ratings}")

            industry_analysis = get_industry_analysis(ticker, industry)
            logging.info(f"industry_analysis: {industry_analysis}")

            final_analysis = get_final_analysis(ticker, {}, sentiment_analysis, analyst_ratings, industry_analysis)
            logging.info(f"final_analysis: {final_analysis}")

            analyses[ticker] = final_analysis
            prices[ticker] = get_current_price(ticker)
        except Exception as e:
            logging.error(f"Error analyzing {ticker}: {traceback.format_exc()}")
            sys.exit(1)

    # Rank the companies based on their analyses
    ranking = rank_companies(industry, analyses, prices)
    print(f"\nRanking of Companies in the {industry} Industry:")
    print(ranking)

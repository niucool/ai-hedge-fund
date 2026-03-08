import datetime
import os
import time

import pandas as pd
import requests
import yfinance as yf
from dateutil.relativedelta import relativedelta

from src.data.cache import get_cache
from src.data.models import (
    CompanyFactsResponse,
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    InsiderTrade,
    InsiderTradeResponse,
    LineItem,
    LineItemResponse,
    Price,
    PriceResponse,
)

# Global cache instance
_cache = get_cache()


def _make_api_request(url: str, headers: dict, method: str = "GET", json_data: dict = None, max_retries: int = 3, params: dict = None) -> requests.Response:
    """
    Make an API request with rate limiting handling and moderate backoff.
    """
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        if method.upper() == "POST":
            response = requests.post(url, headers=headers, json=json_data, params=params)
        else:
            response = requests.get(url, headers=headers, params=params)

        if response.status_code == 429 and attempt < max_retries:
            # Linear backoff: 60s, 90s, 120s, 150s...
            delay = 60 + (30 * attempt)
            print(f"Rate limited (429). Attempt {attempt + 1}/{max_retries + 1}. Waiting {delay}s before retrying...")
            time.sleep(delay)
            continue

        return response


def get_prices(ticker: str, start_date: str, end_date: str, api_key: str = None) -> list[Price]:
    """Fetch price data from cache or API."""
    cache_key = f"{ticker}_{start_date}_{end_date}"

    if cached_data := _cache.get_prices(cache_key):
        return [Price(**price) for price in cached_data]

    # Use yfinance instead
    end_date_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    # yfinance end_date is exclusive, so we add 1 day
    yf_end_date = (end_date_dt + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    df = yf.download(ticker, start=start_date, end=yf_end_date, progress=False)

    if df.empty:
        return []

    prices = []
    # yf.download returns a MultiIndex column DataFrame if multiple tickers, but a single ticker returns single level
    # In recent versions of yfinance, even a single ticker might return MultiIndex columns

    # Flatten multi-index columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    # We need to drop any missing values (like NaN)
    df = df.dropna()

    for date, row in df.iterrows():
        prices.append(Price(time=date.strftime("%Y-%m-%d"), open=float(row["Open"]), close=float(row["Close"]), high=float(row["High"]), low=float(row["Low"]), volume=int(row["Volume"])))

    if not prices:
        return []

    _cache.set_prices(cache_key, [p.model_dump() for p in prices])
    return prices


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[FinancialMetrics]:
    """Fetch financial metrics from cache or yfinance."""
    cache_key = f"{ticker}_{period}_{end_date}_{limit}"

    if cached_data := _cache.get_financial_metrics(cache_key):
        return [FinancialMetrics(**metric) for metric in cached_data]

    yf_ticker = yf.Ticker(ticker)
    info = yf_ticker.info
    financials = yf_ticker.financials
    balance_sheet = yf_ticker.balance_sheet
    cashflow = yf_ticker.cashflow

    # Basic mapping of yfinance info to our metrics
    # We create a single latest FinancialMetrics for now

    # We will approximate or use `None` for missing metrics
    metric = FinancialMetrics(
        ticker=ticker,
        report_period=end_date,  # This is approximated to end_date
        period=period,
        currency=info.get("currency", "USD"),
        market_cap=info.get("marketCap"),
        enterprise_value=info.get("enterpriseValue"),
        price_to_earnings_ratio=info.get("trailingPE"),
        price_to_book_ratio=info.get("priceToBook"),
        price_to_sales_ratio=info.get("priceToSalesTrailing12Months"),
        enterprise_value_to_ebitda_ratio=info.get("enterpriseToEbitda"),
        enterprise_value_to_revenue_ratio=info.get("enterpriseToRevenue"),
        free_cash_flow_yield=None,  # Needs calculation
        peg_ratio=info.get("pegRatio"),
        gross_margin=info.get("grossMargins"),
        operating_margin=info.get("operatingMargins"),
        net_margin=info.get("profitMargins"),
        return_on_equity=info.get("returnOnEquity"),
        return_on_assets=info.get("returnOnAssets"),
        return_on_invested_capital=None,
        asset_turnover=None,
        inventory_turnover=None,
        receivables_turnover=None,
        days_sales_outstanding=None,
        operating_cycle=None,
        working_capital_turnover=None,
        current_ratio=info.get("currentRatio"),
        quick_ratio=info.get("quickRatio"),
        cash_ratio=None,
        operating_cash_flow_ratio=None,
        debt_to_equity=info.get("debtToEquity"),
        debt_to_assets=None,
        interest_coverage=None,
        revenue_growth=info.get("revenueGrowth"),
        earnings_growth=info.get("earningsGrowth"),
        book_value_growth=None,
        earnings_per_share_growth=None,
        free_cash_flow_growth=None,
        operating_income_growth=None,
        ebitda_growth=None,
        payout_ratio=info.get("payoutRatio"),
        earnings_per_share=info.get("trailingEps"),
        book_value_per_share=info.get("bookValue"),
        free_cash_flow_per_share=None,
    )

    financial_metrics = [metric]

    _cache.set_financial_metrics(cache_key, [m.model_dump() for m in financial_metrics])
    return financial_metrics


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[LineItem]:
    """Fetch line items from yfinance."""
    # This might require extracting from financials, balance_sheet, cashflow
    yf_ticker = yf.Ticker(ticker)
    info = yf_ticker.info
    financials = yf_ticker.financials
    balance_sheet = yf_ticker.balance_sheet
    cashflow = yf_ticker.cashflow

    # Create a generic LineItem for the latest data available
    # For a real implementation, you'd match specific requested line_items to yfinance index names
    item_data = {
        "ticker": ticker,
        "report_period": end_date,
        "period": period,
        "currency": info.get("currency", "USD"),
    }

    # Very basic mapping attempt for common items, you can expand this based on needs
    # For now, if the user requests an item that we can find in info or a table, we add it
    for req_item in line_items:
        # e.g., "free_cash_flow" -> info.get("freeCashflow")
        if req_item == "free_cash_flow":
            item_data[req_item] = info.get("freeCashflow")
        elif req_item == "net_income":
            item_data[req_item] = info.get("netIncomeToCommon")
        elif req_item == "total_revenue":
            item_data[req_item] = info.get("totalRevenue")
        elif req_item == "total_assets":
            item_data[req_item] = info.get("totalAssets")
        elif req_item == "total_debt":
            item_data[req_item] = info.get("totalDebt")
        # Add more as needed, or map dynamically from financials.index
        # We try to keep it simple and fulfill the tool's requirements

    line_item = LineItem(**item_data)
    search_results = [line_item]

    return search_results[:limit]


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[InsiderTrade]:
    """Fetch insider trades from cache or EDGAR."""
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"

    if cached_data := _cache.get_insider_trades(cache_key):
        return [InsiderTrade(**trade) for trade in cached_data]

    # As a simple fallback/mock, since actual EDGAR scraping is complex and yfinance doesn't provide
    # full insider trades natively in a way that matches this model exactly.
    # Often, you'd use a service like Finnhub or sec-api.io for this.
    # Since we can't easily parse EDGAR XMLs without a lot of logic, we return an empty list or mock.
    # In a real scenario you would query SEC EDGAR API directly for Form 4s.
    # For now, let's just return an empty list to avoid breaking the tool.

    all_trades = []

    all_trades = fetch_insider_trades_edgar(ticker, start_date, end_date, limit)

    # Cache the results
    _cache.set_insider_trades(cache_key, [trade.model_dump() for trade in all_trades])
    return all_trades


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[CompanyNews]:
    """Fetch company news from cache or newsdata.io API."""
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"

    if cached_data := _cache.get_company_news(cache_key):
        return [CompanyNews(**news) for news in cached_data]

    headers = {}
    news_api_key = api_key or os.environ.get("NEWSDATA_API_KEY")
    if not news_api_key:
        return []

    # newsdata.io API
    url = "https://newsdata.io/api/1/news"
    params = {"apikey": news_api_key, "q": ticker, "language": "en"}

    all_news = []

    # Simplistic fetch, real implementation might need to handle pagination & dates properly
    response = _make_api_request(url, headers, params=params)
    if response.status_code != 200:
        return []

    try:
        data = response.json()
        articles = data.get("results", [])
        for article in articles:
            all_news.append(CompanyNews(ticker=ticker, title=article.get("title", ""), author=article.get("creator", [""])[0] if article.get("creator") else "Unknown", source=article.get("source_id", ""), date=article.get("pubDate", ""), url=article.get("link", ""), sentiment=None))
    except Exception as e:
        print(f"Error parsing news data: {e}")
        return []

    # Filter by dates if needed
    if start_date:
        all_news = [n for n in all_news if n.date[:10] >= start_date]
    if end_date:
        all_news = [n for n in all_news if n.date[:10] <= end_date]

    all_news = all_news[:limit]

    _cache.set_company_news(cache_key, [news.model_dump() for news in all_news])
    return all_news


def get_market_cap(
    ticker: str,
    end_date: str,
    api_key: str = None,
) -> float | None:
    """Fetch market cap from yfinance."""
    yf_ticker = yf.Ticker(ticker)
    info = yf_ticker.info
    return info.get("marketCap")


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


def get_price_data(ticker: str, start_date: str, end_date: str, api_key: str = None) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date, api_key=api_key)
    return prices_to_df(prices)


def get_cik(ticker: str) -> str:
    """Fetch CIK for a given ticker from SEC."""
    headers = {"User-Agent": "AI_Hedge_Fund myemail@example.com"}
    try:
        r = _make_api_request("https://www.sec.gov/files/company_tickers.json", headers=headers)
        if r.status_code == 200:
            tickers = r.json()
            for k, v in tickers.items():
                if v["ticker"] == ticker.upper():
                    return str(v["cik_str"]).zfill(10)
    except Exception as e:
        print(f"Error fetching CIK: {e}")
    return None


def fetch_insider_trades_edgar(ticker: str, start_date: str, end_date: str, limit: int = 1000) -> list[InsiderTrade]:
    """Basic extraction of recent Form 4 filings from EDGAR for a given ticker."""
    cik = get_cik(ticker)
    if not cik:
        return []

    headers = {"User-Agent": "AI_Hedge_Fund myemail@example.com"}
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = _make_api_request(url, headers=headers)
    if r.status_code != 200:
        return []

    try:
        data = r.json()
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        accession_numbers = recent.get("accessionNumber", [])

        trades = []
        for i, form in enumerate(forms):
            if form in ["4", "4/A"]:
                date_str = filing_dates[i]

                # Check dates
                if end_date and date_str > end_date:
                    continue
                if start_date and date_str < start_date:
                    continue

                # We won't fetch the actual XML for each form 4 because it takes too long and requires parsing XML
                # We will just append a placeholder trade to indicate insider activity
                trades.append(InsiderTrade(ticker=ticker, issuer=ticker, name=None, title=None, is_board_director=None, transaction_date=date_str, transaction_shares=0, transaction_price_per_share=0, transaction_value=0, shares_owned_before_transaction=0, shares_owned_after_transaction=0, security_title=None, filing_date=date_str))  # Placeholder  # Placeholder  # Placeholder
                if len(trades) >= limit:
                    break
        return trades
    except Exception as e:
        print(f"Error parsing EDGAR submissions: {e}")
        return []

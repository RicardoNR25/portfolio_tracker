import json
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from tqdm import tqdm
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
from dateutil.relativedelta import relativedelta
import uuid
from scipy import stats

PORTFOLIO_FILE = 'portfolio.json'

# --- Currency Helper Function ---
def get_currency_from_ticker(ticker):
    """
    Determines currency based on ticker symbol.
    European stocks (ending in .AS, .PA, .DE, etc.) use EUR, others use USD.
    """
    european_exchanges = ['.AS', '.PA', '.DE', '.MI', '.MC', '.SW', '.BR', '.HE', '.OL', '.ST', '.CO']
    for exchange in european_exchanges:
        if ticker.endswith(exchange):
            return 'EUR'
    return 'USD'

def get_eur_usd_rate():
    """
    Fetches the most recent EUR/USD exchange rate.
    Returns None if unable to fetch.
    """
    try:
        eurusd = yf.Ticker('EURUSD=X')
        hist = eurusd.history(period='1d')
        if not hist.empty:
            return hist['Close'].iloc[-1]
    except:
        pass
    # Fallback to a slightly longer period if today's data isn't available
    try:
        eurusd = yf.Ticker('EURUSD=X')
        hist = eurusd.history(period='5d')
        if not hist.empty:
            return hist['Close'].iloc[-1]
    except:
        pass
    return None

# --- Asset Database
ASSET_DATABASE = {
    'Stocks': [
        {'name': 'Apple (AAPL)', 'ticker': 'AAPL', 'sector': 'Tech', 'asset_class': 'Stock'},
        {'name': 'Microsoft (MSFT)', 'ticker': 'MSFT', 'sector': 'Tech', 'asset_class': 'Stock'},
        {'name': 'Google (GOOGL)', 'ticker': 'GOOGL', 'sector': 'Tech', 'asset_class': 'Stock'},
        {'name': 'Amazon (AMZN)', 'ticker': 'AMZN', 'sector': 'Tech', 'asset_class': 'Stock'},
        {'name': 'NVIDIA (NVDA)', 'ticker': 'NVDA', 'sector': 'Tech', 'asset_class': 'Stock'},
        {'name': 'Tesla (TSLA)', 'ticker': 'TSLA', 'sector': 'Automotive', 'asset_class': 'Stock'},
        {'name': 'JPMorgan Chase (JPM)', 'ticker': 'JPM', 'sector': 'Finance', 'asset_class': 'Stock'},
        {'name': 'Berkshire Hathaway (BRK-B)', 'ticker': 'BRK-B', 'sector': 'Conglomerate', 'asset_class': 'Stock'},
        {'name': 'ASML (ASML.AS)', 'ticker': 'ASML.AS', 'sector': 'Tech', 'asset_class': 'Stock'},
        {'name': 'Adyen (ADYEN.AS)', 'ticker': 'ADYEN.AS', 'sector': 'Finance', 'asset_class': 'Stock'},
        {'name': 'Unilever (UNA.AS)', 'ticker': 'UNA.AS', 'sector': 'Consumer', 'asset_class': 'Stock'},
        {'name': 'Shell (SHEL)', 'ticker': 'SHEL', 'sector': 'Energy', 'asset_class': 'Stock'},
        {'name': 'a.s.r. Nederland (ASRNL.AS)', 'ticker': 'ASRNL.AS', 'sector': 'Finance', 'asset_class': 'Stock'},
    ],
    'Stock ETFs': [
        {'name': 'Vanguard S&P 500 (VOO)', 'ticker': 'VOO', 'sector': 'Index', 'asset_class': 'Stock ETF'},
        {'name': 'Invesco QQQ (QQQ)', 'ticker': 'QQQ', 'sector': 'Index', 'asset_class': 'Stock ETF'},
        {'name': 'Vanguard Total Stock Market (VTI)', 'ticker': 'VTI', 'sector': 'Index', 'asset_class': 'Stock ETF'},
        {'name': 'iShares MSCI Emerging Markets (EEM)', 'ticker': 'EEM', 'sector': 'Index', 'asset_class': 'Stock ETF'},
    ],
    'Bond ETFs': [
        {'name': 'iShares 20+ Year Treasury (TLT)', 'ticker': 'TLT', 'sector': 'Bonds', 'asset_class': 'Bond ETF'},
        {'name': 'iShares 7-10 Year Treasury (IEF)', 'ticker': 'IEF', 'sector': 'Bonds', 'asset_class': 'Bond ETF'},
        {'name': 'Vanguard Total Bond Market (AGG)', 'ticker': 'AGG', 'sector': 'Bonds', 'asset_class': 'Bond ETF'},
    ],
    'Commodity ETFs': [
        {'name': 'SPDR Gold Trust (GLD)', 'ticker': 'GLD', 'sector': 'Commodity', 'asset_class': 'Commodity ETF'},
        {'name': 'iShares Silver Trust (SLV)', 'ticker': 'SLV', 'sector': 'Commodity', 'asset_class': 'Commodity ETF'},
        {'name': 'United States Oil Fund (USO)', 'ticker': 'USO', 'sector': 'Commodity', 'asset_class': 'Commodity ETF'},
    ]
}

def get_available_assets():
    """Returns a flat list of all assets."""
    all_assets = []
    for category, assets in ASSET_DATABASE.items():
        all_assets.extend(assets)
    return all_assets

def get_rich_asset_database():
    """
    Returns asset database enriched with live prices and change %.
    Used for the interactive menu when adding transactions.
    Returns a dictionary with same structure as ASSET_DATABASE.
    """
    # Get all tickers from all categories
    all_tickers = []
    for category, assets in ASSET_DATABASE.items():
        all_tickers.extend([asset['ticker'] for asset in assets])
    # Fetch live prices
    live_prices_df = fetch_live_prices(all_tickers)
    # Create a lookup dictionary
    price_lookup = {}
    if not live_prices_df.empty:
        for _, row in live_prices_df.iterrows():
            price_lookup[row['Ticker']] = {
                'price': row['Price'],
                'change': row['Change (%)']
            }
    # Enrich the database structure
    rich_database = {}
    for category, assets in ASSET_DATABASE.items():
        rich_assets = []
        for asset in assets:
            ticker = asset['ticker']
            asset_copy = asset.copy()
            # Add currency based on ticker
            asset_copy['currency'] = get_currency_from_ticker(ticker)
            if ticker in price_lookup:
                asset_copy['currentPrice'] = price_lookup[ticker]['price']
                asset_copy['dayChangePct'] = price_lookup[ticker]['change']
            else:
                asset_copy['currentPrice'] = 0.0
                asset_copy['dayChangePct'] = 0.0
            rich_assets.append(asset_copy)
        rich_database[category] = rich_assets
    return rich_database

def fetch_live_prices(tickers):
    """Fetches live prices using yfinance."""
    if not tickers:
        return pd.DataFrame()
    results = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            # Get 2 days of data to calculate change
            hist = stock.history(period='2d')
            if not hist.empty and len(hist) > 0:
                last_close = hist['Close'].iloc[-1]
                # Calculate change if we have previous data
                if len(hist) >= 2:
                    prev_close = hist['Close'].iloc[-2]
                    change_pct = ((last_close - prev_close) / prev_close * 100) if prev_close != 0 else 0
                else:
                    change_pct = 0
                results.append({
                    'Ticker': ticker,
                    'Price': float(last_close),
                    'Change (%)': float(change_pct),
                    'Currency': get_currency_from_ticker(ticker)
                })
        except Exception as e:
            print(f"Warning: Could not fetch price for {ticker}: {e}")
            continue
    return pd.DataFrame(results)

# --- Transaction Class ---
class Transaction:
    def __init__(self, ticker, name, sector, asset_class, quantity, price, date, transaction_type, tx_id=None, currency=None):
        self.ticker = ticker
        self.name = name
        self.sector = sector
        self.asset_class = asset_class
        self.quantity = float(quantity)
        self.price = float(price)
        self.tx_id = tx_id if tx_id else str(uuid.uuid4())
        # Auto-determine currency if not provided
        self.currency = currency if currency else get_currency_from_ticker(ticker)
        if transaction_type.upper() not in ['BUY', 'SELL']:
            raise ValueError("transaction_type must be 'BUY' or 'SELL'")
        self.transaction_type = transaction_type.upper()
        if isinstance(date, str):
            self.date = parse_date(date)
        else:
            self.date = date
        self.transaction_value = self.quantity * self.price
        if self.transaction_type == 'SELL':
            self.transaction_value *= -1

    def to_dict(self):
        return {
            'tx_id': self.tx_id,
            'ticker': self.ticker,
            'name': self.name,
            'sector': self.sector,
            'asset_class': self.asset_class,
            'quantity': self.quantity,
            'price': self.price,
            'date': self.date.isoformat(),
            'transaction_type': self.transaction_type,
            'currency': self.currency
        }
    @classmethod
    def from_dict(cls, data):
        return cls(
            ticker=data['ticker'],
            name=data['name'],
            sector=data['sector'],
            asset_class=data['asset_class'],
            quantity=data['quantity'],
            price=data['price'],
            date=data['date'],
            transaction_type=data['transaction_type'],
            tx_id=data.get('tx_id'),
            currency=data.get('currency')  # Backward compatibility
        )

# --- Portfolio Class ---
class Portfolio:
    def __init__(self):
        self.transactions = []

    def add_transaction(self, transaction):
        if transaction.transaction_type == 'SELL':
            current_qty = self.get_current_quantity(transaction.ticker)
            if transaction.quantity > current_qty:
                raise ValueError(f"Cannot sell {transaction.quantity} {transaction.ticker}, only have {current_qty}.")
        self.transactions.append(transaction)

    def remove_transaction_by_id(self, tx_id):
        for i, tx in enumerate(self.transactions):
            if tx.tx_id == tx_id:
                del self.transactions[i]
                return True
        return False

    def get_current_quantity(self, ticker):
        qty = 0
        for tx in self.transactions:
            if tx.ticker == ticker:
                if tx.transaction_type == 'BUY':
                    qty += tx.quantity
                else:
                    qty -= tx.quantity
        return qty

    def get_holdings(self):
        holdings = {}
        for tx in self.transactions:
            if tx.ticker not in holdings:
                holdings[tx.ticker] = {
                    'ticker': tx.ticker,
                    'name': tx.name,
                    'sector': tx.sector,
                    'asset_class': tx.asset_class,
                    'quantity': 0,
                    'total_cost': 0,
                    'currency': tx.currency
                }
            if tx.transaction_type == 'BUY':
                holdings[tx.ticker]['quantity'] += tx.quantity
                holdings[tx.ticker]['total_cost'] += tx.transaction_value
            else:  # SELL
                # Calculate the average cost per share BEFORE the sell
                if holdings[tx.ticker]['quantity'] > 0:
                    avg_cost_per_share = holdings[tx.ticker]['total_cost'] / holdings[tx.ticker]['quantity']
                    # Remove the cost basis of the shares being sold
                    cost_basis_removed = avg_cost_per_share * tx.quantity
                    holdings[tx.ticker]['total_cost'] -= cost_basis_removed
                # Reduce quantity
                holdings[tx.ticker]['quantity'] -= tx.quantity
        holdings = {k: v for k, v in holdings.items() if v['quantity'] > 0.001}
        return holdings

    def is_empty(self):
        return len(self.get_holdings()) == 0

    def get_all_transactions_df(self):
        if not self.transactions:
            return pd.DataFrame()
        data = []
        for tx in self.transactions:
            data.append({
                'tx_id': tx.tx_id,  # Full tx_id needed for removal
                'ID': tx.tx_id[:8],  # Short version for display
                'Date': tx.date.strftime('%Y-%m-%d'),
                'Type': tx.transaction_type,
                'Ticker': tx.ticker,
                'Name': tx.name,
                'Quantity': tx.quantity,
                'Price': tx.price,
                'Value': abs(tx.transaction_value),
                'Currency': tx.currency
            })
        return pd.DataFrame(data)

# --- Save/Load Functions ---
def save_portfolio(portfolio):
    data = [tx.to_dict() for tx in portfolio.transactions]
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def load_portfolio():
    portfolio = Portfolio()
    try:
        with open(PORTFOLIO_FILE, 'r') as f:
            data = json.load(f)
            for tx_dict in data:
                tx = Transaction.from_dict(tx_dict)
                portfolio.transactions.append(tx)
    except FileNotFoundError:
        pass
    return portfolio

# --- Dashboard Data Functions ---
def get_dashboard_data(portfolio, target_currency=None):
    """
    Returns portfolio data with option to convert to target currency.
    target_currency: 'EUR', 'USD', or None (native currencies)
    """
    holdings = portfolio.get_holdings()
    if not holdings:
        return pd.DataFrame(), 0
    tickers = list(holdings.keys())
    live_prices_df = fetch_live_prices(tickers)
    if live_prices_df.empty:
        print("Warning: Could not fetch any live prices. Check your internet connection.")
        return pd.DataFrame(), 0
    print(f"Successfully fetched prices for {len(live_prices_df)} assets.")
    # Get EUR/USD rate if conversion needed
    eur_usd_rate = None
    if target_currency in ['EUR', 'USD']:
        eur_usd_rate = get_eur_usd_rate()
    data = []
    for ticker in tickers:
        holding = holdings[ticker]
        price_row = live_prices_df[live_prices_df['Ticker'] == ticker]
        if price_row.empty:
            continue
        current_price = price_row['Price'].values[0]
        change_pct = price_row['Change (%)'].values[0]
        native_currency = holding['currency']
        # Convert price if needed
        display_currency = target_currency if target_currency else native_currency
        if target_currency and native_currency != target_currency and eur_usd_rate:
            if native_currency == 'EUR' and target_currency == 'USD':
                current_price *= eur_usd_rate
            elif native_currency == 'USD' and target_currency == 'EUR':
                current_price /= eur_usd_rate
        quantity = holding['quantity']
        market_value = current_price * quantity
        avg_cost = holding['total_cost'] / quantity
        # Convert avg_cost if needed
        if target_currency and native_currency != target_currency and eur_usd_rate:
            if native_currency == 'EUR' and target_currency == 'USD':
                avg_cost *= eur_usd_rate
            elif native_currency == 'USD' and target_currency == 'EUR':
                avg_cost /= eur_usd_rate
        cost_basis = avg_cost * quantity
        total_gain = market_value - cost_basis
        total_gain_pct = (total_gain / cost_basis * 100) if cost_basis != 0 else 0
        day_gain = market_value * (change_pct / 100)
        data.append({
            'Ticker': ticker,
            'Name': holding['name'],
            'Sector': holding['sector'],
            'Asset Class': holding['asset_class'],
            'Quantity': quantity,
            'Avg Cost': avg_cost,
            'Current Price': current_price,
            'Market Value': market_value,
            "Day's Gain ($)": day_gain,
            "Day's Gain (%)": change_pct,
            "Total Gain ($)": total_gain,
            "Total Gain (%)": total_gain_pct,
            'Currency': display_currency
        })
    df = pd.DataFrame(data)
    total_value = df['Market Value'].sum() if not df.empty else 0
    # Add Weight (%) column
    if not df.empty and total_value > 0:
        df['Weight (%)'] = (df['Market Value'] / total_value * 100)
    elif not df.empty:
        df['Weight (%)'] = 0.0
    return df, total_value

def get_summary_by_group(portfolio_df, group_column):
    if portfolio_df.empty:
        return pd.DataFrame()
    grouped = portfolio_df.groupby(group_column).agg({
        'Market Value': 'sum',
        "Total Gain ($)": 'sum'
    }).reset_index()
    total_value = grouped['Market Value'].sum()
    grouped['Weight (%)'] = (grouped['Market Value'] / total_value * 100) if total_value > 0 else 0
    grouped = grouped.sort_values('Market Value', ascending=False)
    return grouped

def get_simple_portfolio_df(dashboard_df):
    """Simplified version for simulation."""
    if dashboard_df.empty:
        return pd.DataFrame(), 0
    sim_df = dashboard_df[['Ticker', 'Quantity', 'Market Value']].copy()
    total_value = sim_df['Market Value'].sum()
    sim_df['Weight'] = sim_df['Market Value'] / total_value if total_value > 0 else 0
    return sim_df, total_value

# --- Price History Functions ---
def get_price_history(ticker, period='1y'):
    """Fetches historical price data."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            return pd.DataFrame()
        hist = hist[['Close']].copy()
        hist.columns = ['Price']
        hist.index.name = 'Date'
        return hist.reset_index()
    except Exception as e:
        print(f"Error fetching history for {ticker}: {e}")
        return pd.DataFrame()

def get_historical_data(ticker, period='1y'):
    """
    Fetches historical price data for a single ticker.
    Returns DataFrame with Date and Close price columns.
    Period can be: '1mo', '3mo', '6mo', '1y', '5y', 'max'
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            return pd.DataFrame()
        # Keep only the Close price
        result = pd.DataFrame()
        result['Date'] = hist.index
        result['Close'] = hist['Close'].values
        # Make sure date is timezone-naive
        if hasattr(result['Date'].dtype, 'tz') and result['Date'].dt.tz is not None:
            result['Date'] = result['Date'].dt.tz_localize(None)
        return result.reset_index(drop=True)
    except Exception as e:
        print(f"Error fetching historical data for {ticker}: {e}")
        return pd.DataFrame()

def get_portfolio_performance(portfolio, period='1y', target_currency=None):
    """
    Calculates portfolio performance over time.
    target_currency: 'EUR', 'USD', or None (native)
    """
    # Get all unique tickers from transaction history (including sold assets)
    all_tickers = list(set(tx.ticker for tx in portfolio.transactions))
    if not all_tickers:
        return pd.DataFrame()
    # Build holdings info for all tickers that have ever been in the portfolio
    holdings = {}
    for tx in portfolio.transactions:
        if tx.ticker not in holdings:
            holdings[tx.ticker] = {
                'ticker': tx.ticker,
                'name': tx.name,
                'sector': tx.sector,
                'asset_class': tx.asset_class,
                'currency': tx.currency
            }
    # Get EUR/USD historical rate if conversion needed
    eur_usd_hist = None
    if target_currency in ['EUR', 'USD']:
        try:
            # Map 'all' to 'max' for yfinance compatibility
            yf_period = 'max' if period == 'all' else period
            eurusd = yf.Ticker('EURUSD=X')
            eur_usd_hist = eurusd.history(period=yf_period)['Close']
            # Make timezone-naive
            if hasattr(eur_usd_hist.index, 'tz') and eur_usd_hist.index.tz is not None:
                eur_usd_hist.index = eur_usd_hist.index.tz_localize(None)
        except Exception:
            # Silently continue if exchange rate fetch fails
            pass
    # Determine date range
    period_map = {'1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '5y': 1825, 'all': None}
    days = period_map.get(period)
    if days:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
    else:
        earliest_tx = min(tx.date for tx in portfolio.transactions)
        start_date = earliest_tx
        end_date = datetime.now()
    # Get earliest transaction date per ticker
    first_tx_date = {}
    for ticker in holdings.keys():
        ticker_txs = [tx for tx in portfolio.transactions if tx.ticker == ticker]
        if ticker_txs:
            first_tx_date[ticker] = min(tx.date for tx in ticker_txs)
    all_price_data = {}
    tickers = list(holdings.keys())
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
            if not hist.empty:
                # Make sure the index is timezone-naive
                close_series = hist['Close']
                if hasattr(close_series.index, 'tz') and close_series.index.tz is not None:
                    close_series.index = close_series.index.tz_localize(None)
                all_price_data[ticker] = close_series
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
    if not all_price_data:
        return pd.DataFrame()
    # Combine all dates
    all_dates = pd.DatetimeIndex([])
    for prices in all_price_data.values():
        all_dates = all_dates.union(prices.index)
    all_dates = all_dates.sort_values()
    # Make sure all dates are timezone-naive
    if hasattr(all_dates, 'tz') and all_dates.tz is not None:
        all_dates = all_dates.tz_localize(None)
    # Calculate holdings over time
    holdings_over_time = pd.DataFrame(index=all_dates)
    cost_over_time = pd.DataFrame(index=all_dates)
    # Only process tickers where we successfully got price data
    for ticker in all_price_data.keys():  # Changed from holdings.keys()
        if ticker not in holdings:
            continue  # Skip if ticker not in holdings dict
        native_currency = holdings[ticker]['currency']
        holdings_df = pd.DataFrame(index=all_dates, columns=['Quantity', 'Cost'], dtype=float)
        holdings_df.fillna(0, inplace=True)
        ticker_txs = [tx for tx in portfolio.transactions if tx.ticker == ticker]
        ticker_txs.sort(key=lambda x: x.date)
        # Track cumulative quantity and cost to calculate correct cost basis removal
        cumulative_qty = 0
        cumulative_cost = 0
        for tx in ticker_txs:
            # Make sure tx.date is timezone-naive to match index
            tx_date = tx.date.replace(tzinfo=None) if hasattr(tx.date, 'tzinfo') and tx.date.tzinfo is not None else tx.date
            mask = holdings_df.index >= tx_date
            if tx.transaction_type == 'BUY':
                cumulative_qty += tx.quantity
                cumulative_cost += tx.transaction_value
                holdings_df.loc[mask, 'Quantity'] += tx.quantity
                holdings_df.loc[mask, 'Cost'] += tx.transaction_value
            else:  # SELL
                # Calculate proportional cost basis to remove
                if cumulative_qty > 0:
                    avg_cost_per_unit = cumulative_cost / cumulative_qty
                    cost_basis_removed = avg_cost_per_unit * tx.quantity
                else:
                    cost_basis_removed = 0
                # Update cumulative tracking
                cumulative_qty -= tx.quantity
                cumulative_cost -= cost_basis_removed
                # Update DataFrame
                holdings_df.loc[mask, 'Quantity'] -= tx.quantity
                holdings_df.loc[mask, 'Cost'] -= cost_basis_removed
        holdings_over_time[ticker] = holdings_df['Quantity']
        cost_over_time[ticker] = holdings_df['Cost']
    # Calculate prices over time
    prices_over_time = pd.DataFrame(index=all_dates)
    for ticker, prices in all_price_data.items():
        prices_over_time[ticker] = prices.reindex(all_dates, method='ffill')
    # Convert currencies if needed
    if target_currency and eur_usd_hist is not None:
        eur_usd_aligned = eur_usd_hist.reindex(all_dates, method='ffill')
        for ticker in prices_over_time.columns:  # Changed from holdings.keys()
            if ticker not in holdings:
                continue
            native_currency = holdings[ticker]['currency']
            if native_currency != target_currency:
                if native_currency == 'EUR' and target_currency == 'USD':
                    prices_over_time[ticker] = prices_over_time[ticker] * eur_usd_aligned
                    cost_over_time[ticker] = cost_over_time[ticker] * eur_usd_aligned
                elif native_currency == 'USD' and target_currency == 'EUR':
                    prices_over_time[ticker] = prices_over_time[ticker] / eur_usd_aligned
                    cost_over_time[ticker] = cost_over_time[ticker] / eur_usd_aligned
    # Calculate portfolio value
    # Only use tickers that exist in both dataframes
    common_tickers = holdings_over_time.columns.intersection(prices_over_time.columns)
    if len(common_tickers) == 0:
        return pd.DataFrame()
    portfolio_value = (holdings_over_time[common_tickers] * prices_over_time[common_tickers]).sum(axis=1)
    portfolio_cost = cost_over_time[common_tickers].sum(axis=1)
    result_df = pd.DataFrame({
        'Market Value': portfolio_value,
        'Cost Basis': portfolio_cost
    })
    result_df['Return (%)'] = ((result_df['Market Value'] - result_df['Cost Basis']) / result_df['Cost Basis'] * 100).fillna(0)
    return result_df

# --- Asset Return & Volatility Overview ---
def get_asset_assumptions_overview(scenario='base_case'):
    """
    Returns a comprehensive overview of expected returns and volatility 
    assumptions for all asset classes and specific tickers.
    Only includes assets that are available in ASSET_DATABASE.
    """
    # Get all available tickers from ASSET_DATABASE
    available_tickers = set()
    for category, assets in ASSET_DATABASE.items():
        for asset in assets:
            available_tickers.add(asset['ticker'])
    # Ticker-specific volatility mappings
    HIGH_VOL_TECH = {
        'NVDA': 0.40, 'TSLA': 0.45, 'AMD': 0.38, 'PLTR': 0.42, 'SNOW': 0.40,
        'NET': 0.38, 'COIN': 0.50, 'RBLX': 0.42, 'U': 0.40, 'DDOG': 0.38
    }
    MEDIUM_HIGH_VOL = {
        'AAPL': 0.25, 'MSFT': 0.24, 'GOOGL': 0.26, 'META': 0.32, 'AMZN': 0.28,
        'NFLX': 0.35, 'SHOP': 0.36, 'SQ': 0.38, 'PYPL': 0.32, 'ADBE': 0.28
    }
    MEDIUM_VOL = {
        'JPM': 0.24, 'BAC': 0.28, 'WFC': 0.26, 'GS': 0.30,
        'XOM': 0.28, 'CVX': 0.26, 'COP': 0.30,
        'UNH': 0.20, 'JNJ': 0.16, 'PFE': 0.22, 'ABBV': 0.20,
        'DIS': 0.24, 'BA': 0.32, 'CAT': 0.26, 'GE': 0.28
    }
    LOW_VOL_DEFENSIVE = {
        'UL': 0.13, 'PG': 0.14, 'KO': 0.15, 'PEP': 0.15, 'WMT': 0.17,
        'TGT': 0.21, 'COST': 0.19, 'MCD': 0.17, 'KHC': 0.21, 'CL': 0.15
    }
    # Filter to only include available tickers
    HIGH_VOL_TECH = {k: v for k, v in HIGH_VOL_TECH.items() if k in available_tickers}
    MEDIUM_HIGH_VOL = {k: v for k, v in MEDIUM_HIGH_VOL.items() if k in available_tickers}
    MEDIUM_VOL = {k: v for k, v in MEDIUM_VOL.items() if k in available_tickers}
    LOW_VOL_DEFENSIVE = {k: v for k, v in LOW_VOL_DEFENSIVE.items() if k in available_tickers}
    # Scenarios with returns
    SCENARIOS = {
        'base_case': {
            'name': 'Base Case (Normal Economy)',
            'returns': {
                'bond': 0.03, 'stock': 0.08, 'stock_etf': 0.08, 
'commodity': 0.04,
                'tech': 0.10, 'defensive': 0.06, 'energy': 0.07, 'financial': 0.08
            },
            'vol_multiplier': 1.0
        },
        'ai_bubble_burst': {
            'name': 'AI Bubble Bursts',
            'returns': {
                'bond': 0.04, 'stock': 0.03, 'stock_etf': 0.04,
'commodity': 0.03,
                'tech': -0.15, 'defensive': 0.07, 'energy': 0.05, 'financial': 0.04
            },
            'vol_multiplier': 1.3
        },
        'agi_success': {
            'name': 'AGI Breakthrough',
            'returns': {
                'bond': 0.02, 'stock': 0.12, 'stock_etf': 0.12,
'commodity': 0.02,
                'tech': 0.25, 'defensive': 0.04, 'energy': -0.02, 'financial': 0.10
            },
            'vol_multiplier': 1.2
        },
        'trade_war': {
            'name': 'Global Trade War',
            'returns': {
                'bond': 0.01, 'stock': 0.02, 'stock_etf': 0.02,
'commodity': 0.08,
                'tech': -0.05, 'defensive': 0.05, 'energy': 0.06, 'financial': 0.01
            },
            'vol_multiplier': 1.3
        },
        'financial_crisis': {
            'name': 'Financial Crisis 2.0',
            'returns': {
                'bond': 0.05, 'stock': -0.05, 'stock_etf': -0.03,
'commodity': -0.02,
                'tech': -0.08, 'defensive': 0.02, 'energy': -0.05, 'financial': -0.15
            },
            'vol_multiplier': 1.6
        },
        'green_transition': {
            'name': 'Green Energy Revolution',
            'returns': {
                'bond': 0.03, 'stock': 0.09, 'stock_etf': 0.08,
'commodity': 0.06,
                'tech': 0.12, 'defensive': 0.05, 'energy': -0.05, 'financial': 0.07
            },
            'vol_multiplier': 1.1
        },
    }
    scenario_data = SCENARIOS.get(scenario, SCENARIOS['base_case'])
    # Base volatilities per asset class (based on 2020-2024 historical data)
    BASE_VOLATILITY = {
        'bond': 0.07,        # Long-term treasuries: conservative estimate ~7%
        'stock': 0.18,       # Individual stocks: ~18%
        'stock_etf': 0.14,   # Diversified ETFs (SPY, VOO): ~14%
        'commodity': 0.16,   # Gold and commodities: ~15-17%
        'tech': 0.28,        # Tech sector average: ~25-30%
        'defensive': 0.14,   # Defensive stocks: ~13-16%
        'energy': 0.26,      # Energy sector: ~24-28%
        'financial': 0.24,   # Financial sector: ~22-26%
        'default': 0.16      # Default fallback: ~16%
    }
    return {
        'scenario_name': scenario_data['name'],
        'vol_multiplier': scenario_data['vol_multiplier'],
        'returns': scenario_data['returns'],
        'base_volatility': BASE_VOLATILITY,
        'high_vol_tech': HIGH_VOL_TECH,
        'medium_high_vol': MEDIUM_HIGH_VOL,
        'medium_vol': MEDIUM_VOL,
        'low_vol_defensive': LOW_VOL_DEFENSIVE
    }

# --- Monte Carlo Simulation ---
def run_monte_carlo_simulation(portfolio_df, years=15, simulations=100000, **kwargs):
    tickers = portfolio_df['Ticker'].unique()
    weights = portfolio_df.set_index('Ticker')['Weight'].to_dict()
    # Get asset classes
    ticker_to_asset_class = {}
    if 'Asset Class' in portfolio_df.columns:
        for _, row in portfolio_df.iterrows():
            ticker_to_asset_class[row['Ticker']] = row['Asset Class']
    # ==================== TICKER-SPECIFIC VOLATILITY ====================
    # High volatility tech stocks (based on 2020-2024 historical data)
    HIGH_VOL_TECH = {
        'NVDA': 0.40, 'TSLA': 0.45, 'AMD': 0.38, 'PLTR': 0.42, 'SNOW': 0.40,
        'NET': 0.38, 'COIN': 0.50, 'RBLX': 0.42, 'U': 0.40, 'DDOG': 0.38
    }
    # Medium-high volatility growth stocks (FAANG + growth)
    MEDIUM_HIGH_VOL = {
        'AAPL': 0.25, 'MSFT': 0.24, 'GOOGL': 0.26, 'META': 0.32, 'AMZN': 0.28,
        'NFLX': 0.35, 'SHOP': 0.36, 'SQ': 0.38, 'PYPL': 0.32, 'ADBE': 0.28
    }
    # Medium volatility established companies
    MEDIUM_VOL = {
        'JPM': 0.24, 'BAC': 0.28, 'WFC': 0.26, 'GS': 0.30,  # Financials
        'XOM': 0.28, 'CVX': 0.26, 'COP': 0.30,  # Energy
        'UNH': 0.20, 'JNJ': 0.16, 'PFE': 0.22, 'ABBV': 0.20,  # Healthcare
        'DIS': 0.24, 'BA': 0.32, 'CAT': 0.26, 'GE': 0.28  # Industrials
    }
    # Low volatility defensive stocks
    LOW_VOL_DEFENSIVE = {
        'UL': 0.13, 'PG': 0.14, 'KO': 0.15, 'PEP': 0.15, 'WMT': 0.17,
        'TGT': 0.21, 'COST': 0.19, 'MCD': 0.17, 'KHC': 0.21, 'CL': 0.15
    }
    # ==================== ECONOMIC SCENARIOS ====================
    SCENARIOS = {
        'base_case': {
            'name': 'Base Case (Normal Economy)',
            'description': 'Stable growth, moderate inflation, balanced markets',
            'returns': {
                'bond': 0.03, 'stock': 0.08, 'stock_etf': 0.08, 
'commodity': 0.04,
                'tech': 0.10, 'defensive': 0.06, 'energy': 0.07, 'financial': 0.08
            },
            'vol_multiplier': 1.0
        },
        'ai_bubble_burst': {
            'name': 'AI Bubble Bursts',
            'description': 'AI hype collapses, tech stocks crash, rotation to value',
            'returns': {
                'bond': 0.04, 'stock': 0.03, 'stock_etf': 0.04,
'commodity': 0.03,
                'tech': -0.15, 'defensive': 0.07, 'energy': 0.05, 'financial': 0.04
            },
            'vol_multiplier': 1.3
        },
        'agi_success': {
            'name': 'AGI Breakthrough',
            'description': 'Artificial General Intelligence achieved, productivity boom',
            'returns': {
                'bond': 0.02, 'stock': 0.12, 'stock_etf': 0.12,
'commodity': 0.02,
                'tech': 0.25, 'defensive': 0.04, 'energy': -0.02, 'financial': 0.10
            },
            'vol_multiplier': 1.2
        },
        'trade_war': {
            'name': 'Global Trade War',
            'description': 'Severe tariffs, supply chain disruption, inflation spike',
            'returns': {
                'bond': 0.01, 'stock': 0.02, 'stock_etf': 0.02,
'commodity': 0.08,
                'tech': -0.05, 'defensive': 0.05, 'energy': 0.06, 'financial': 0.01
            },
            'vol_multiplier': 1.3
        },
        'financial_crisis': {
            'name': 'Financial Crisis 2.0',
            'description': 'Banking collapse, credit freeze, deep recession',
            'returns': {
                'bond': 0.05, 'stock': -0.05, 'stock_etf': -0.03,
'commodity': -0.02,
                'tech': -0.08, 'defensive': 0.02, 'energy': -0.05, 'financial': -0.15
            },
            'vol_multiplier': 1.6
        },
        'green_transition': {
            'name': 'Green Energy Revolution',
            'description': 'Rapid shift to renewables, fossil fuel phase-out accelerates',
            'returns': {
                'bond': 0.03, 'stock': 0.09, 'stock_etf': 0.08,
'commodity': 0.06,
                'tech': 0.12, 'defensive': 0.05, 'energy': -0.05, 'financial': 0.07
            },
            'vol_multiplier': 1.1
        },
    }
    # Get scenario (default to base_case)
    scenario_key = kwargs.get('scenario', 'base_case')
    scenario = SCENARIOS.get(scenario_key, SCENARIOS['base_case'])
    print(f"\n{'='*70}")
    print(f"🎯 ACTIVE SCENARIO: {scenario['name'].upper()}")
    print(f"📝 Description: {scenario['description']}")
    print(f"🔑 Scenario Key: '{scenario_key}'")
    print(f"📊 Volatility Multiplier: {scenario['vol_multiplier']}x")
    print(f"{'='*70}\n")
    # List of known asset tickers by class
    BOND_TICKERS = ['TLT', 'IEF', 'AGG', 'BND', 'SHY', 'LQD', 'HYG', 'VGIT', 'GOVT', 'TIP']
    COMMODITY_TICKERS = ['GLD', 'SLV', 'USO', 'DBC', 'PDBC', 'GSG', 'GCC', 'DBA', 'CORN', 'WEAT', 'SOYB']
    # Tech stock identifiers
    TECH_TICKERS = list(HIGH_VOL_TECH.keys()) + list(MEDIUM_HIGH_VOL.keys()) + [
        'INTC', 'CSCO', 'ORCL', 'IBM', 'CRM', 'NOW'
    ]
    # Defensive stock identifiers  
    DEFENSIVE_TICKERS = list(LOW_VOL_DEFENSIVE.keys()) + ['VZ', 'T', 'NEE', 'DUK', 'SO']
    # Energy stock identifiers
    ENERGY_TICKERS = ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO']
    # Financial stock identifiers
    FINANCIAL_TICKERS = ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'BK']
    print(f"\nCreating synthetic data for ALL {len(tickers)} asset(s)...")
    # Define base volatilities per asset class (based on 2020-2024 historical data)
    BASE_VOLATILITY = {
        'bond': 0.07,        # Long-term treasuries: conservative estimate ~7%
        'stock': 0.18,       # Individual stocks: ~18%
        'stock_etf': 0.14,   # Diversified ETFs (SPY, VOO): ~14%
        'commodity': 0.16,   # Gold and commodities: ~15-17%
        'tech': 0.28,        # Tech sector average: ~25-30%
        'defensive': 0.14,   # Defensive stocks: ~13-16%
        'energy': 0.26,      # Energy sector: ~24-28%
        'financial': 0.24,   # Financial sector: ~22-26%
        'default': 0.16      # Default fallback: ~16%
    }
    # Create synthetic historical returns for ALL tickers
    historical_returns = {}
    synthetic_dates = pd.date_range(end=pd.Timestamp.now(), periods=252*5, freq='B')
    for ticker in tickers:
        asset_class = ticker_to_asset_class.get(ticker, 'Stock')
        # Determine base volatility from ticker-specific mapping
        if ticker in HIGH_VOL_TECH:
            vol = HIGH_VOL_TECH[ticker] * scenario['vol_multiplier']
            asset_type = 'tech'
        elif ticker in MEDIUM_HIGH_VOL:
            vol = MEDIUM_HIGH_VOL[ticker] * scenario['vol_multiplier']
            asset_type = 'tech'
        elif ticker in MEDIUM_VOL:
            vol = MEDIUM_VOL[ticker] * scenario['vol_multiplier']
            # Determine sub-type
            if ticker in ENERGY_TICKERS:
                asset_type = 'energy'
            elif ticker in FINANCIAL_TICKERS:
                asset_type = 'financial'
            else:
                asset_type = 'stock'
        elif ticker in LOW_VOL_DEFENSIVE:
            vol = LOW_VOL_DEFENSIVE[ticker] * scenario['vol_multiplier']
            asset_type = 'defensive'
        elif ticker in BOND_TICKERS or 'Bond' in asset_class or 'bond' in asset_class.lower():
            vol = BASE_VOLATILITY['bond'] * scenario['vol_multiplier']
            asset_type = 'bond'
        elif ticker in COMMODITY_TICKERS or 'Commodity' in asset_class or 'commodity' in asset_class.lower():
            vol = BASE_VOLATILITY['commodity'] * scenario['vol_multiplier']
            asset_type = 'commodity'
        elif 'Stock ETF' in asset_class or 'ETF' in asset_class:
            vol = BASE_VOLATILITY['stock_etf'] * scenario['vol_multiplier']
            asset_type = 'stock_etf'
        else:
            # Unknown ticker - classify by name heuristics
            if any(x in ticker.upper() for x in ['TECH', 'SOFT', 'DATA', 'CYBER', 'CLOUD']):
                vol = BASE_VOLATILITY['tech'] * scenario['vol_multiplier']
                asset_type = 'tech'
            else:
                vol = BASE_VOLATILITY['default'] * scenario['vol_multiplier']
                asset_type = 'stock'
        # Create synthetic returns with zero mean and target volatility
        daily_vol = vol / np.sqrt(252)
        synthetic_returns = pd.Series(
            np.random.normal(0, daily_vol, len(synthetic_dates)),
            index=synthetic_dates
        )
        historical_returns[ticker] = synthetic_returns
        print(f"  {ticker} ({asset_type}): {vol:.1%} volatility")
    if not historical_returns:
        print("No historical data available.")
        return None, None
    returns_df = pd.DataFrame(historical_returns)
    returns_df = returns_df.dropna()
    if returns_df.empty:
        print("No overlapping data.")
        return None, None
    # Keep the historical covariance matrix (preserves volatility and correlations)
    cov_matrix = returns_df.cov()
    # Set realistic expected returns based on asset class AND scenario
    adjusted_mean_returns = pd.Series(index=returns_df.columns, dtype=float)
    # Get scenario returns
    scenario_returns = scenario['returns']
    print("\n" + "="*70)
    print("📈 SCENARIO EXPECTED RETURNS (Annual)")
    print("="*70)
    for asset_type, return_val in scenario_returns.items():
        print(f"  {asset_type.capitalize():<20} {return_val:>+7.1%}")
    print("="*70)
    print("\n--- Applying Returns to Portfolio Assets ---")
    for ticker in adjusted_mean_returns.index:
        asset_class = ticker_to_asset_class.get(ticker, 'Stock')
        # Determine asset type and get scenario return
        if ticker in BOND_TICKERS or 'Bond' in asset_class or 'bond' in asset_class.lower():
            annual_return = scenario_returns['bond']
            asset_type = 'bond'
        elif ticker in COMMODITY_TICKERS or 'Commodity' in asset_class or 'commodity' in asset_class.lower():
            annual_return = scenario_returns['commodity']
            asset_type = 'commodity'
        elif ticker in TECH_TICKERS or 'tech' in asset_class.lower():
            annual_return = scenario_returns['tech']
            asset_type = 'tech'
        elif ticker in DEFENSIVE_TICKERS or 'defensive' in asset_class.lower():
            annual_return = scenario_returns['defensive']
            asset_type = 'defensive'
        elif ticker in ENERGY_TICKERS or 'energy' in asset_class.lower():
            annual_return = scenario_returns['energy']
            asset_type = 'energy'
        elif ticker in FINANCIAL_TICKERS or 'financial' in asset_class.lower():
            annual_return = scenario_returns['financial']
            asset_type = 'financial'
        elif 'Stock ETF' in asset_class or 'ETF' in asset_class:
            annual_return = scenario_returns['stock_etf']
            asset_type = 'stock_etf'
        else:
            annual_return = scenario_returns['stock']
            asset_type = 'stock'
        # Convert annual return to daily return
        adjusted_mean_returns[ticker] = annual_return / 252
        base_return = SCENARIOS['base_case']['returns'].get(asset_type, 0.08)
        change = annual_return - base_return
        print(f"  {ticker} ({asset_type}): {annual_return:+.1%} annual (base: {base_return:+.1%}, change: {change:+.1%})")
    mean_returns = adjusted_mean_returns
    print("---")
    # Debug: Show what mean returns we're actually using
    print("\n--- Mean Returns Being Used ---")
    for ticker in mean_returns.index:
        annual_return = mean_returns[ticker] * 252
        print(f"  {ticker}: {annual_return:.2%} annual")
    print("---\n")
    # Completely rebuild covariance matrix with target volatilities
    # This ensures bonds have very low volatility
    adjusted_cov = pd.DataFrame(index=cov_matrix.index, columns=cov_matrix.columns, dtype=float)
    # First, set target volatilities for each asset
    target_vols = {}
    for ticker in cov_matrix.index:
        asset_class = ticker_to_asset_class.get(ticker, 'Stock')
        if 'Bond' in asset_class or 'bond' in asset_class.lower():
            # Bonds: 0.5% annual volatility → extremely stable
            target_annual_vol = 0.005
            target_vols[ticker] = target_annual_vol / np.sqrt(252)
        elif 'Commodity' in asset_class:
            # Commodities: use original or cap at 12%
            current_vol = np.sqrt(cov_matrix.loc[ticker, ticker])
            target_annual_vol = min(current_vol * np.sqrt(252), 0.12)
            target_vols[ticker] = target_annual_vol / np.sqrt(252)
        else:
            # Stocks: keep original volatility
            target_vols[ticker] = np.sqrt(cov_matrix.loc[ticker, ticker])
    # Rebuild covariance matrix
    for i, ticker1 in enumerate(cov_matrix.index):
        for j, ticker2 in enumerate(cov_matrix.columns):
            if i == j:
                # Diagonal: variance
                adjusted_cov.loc[ticker1, ticker2] = target_vols[ticker1] ** 2
            else:
                # Off-diagonal: covariance based on original correlation
                orig_vol1 = np.sqrt(cov_matrix.loc[ticker1, ticker1])
                orig_vol2 = np.sqrt(cov_matrix.loc[ticker2, ticker2])
                if orig_vol1 > 0 and orig_vol2 > 0:
                    correlation = cov_matrix.loc[ticker1, ticker2] / (orig_vol1 * orig_vol2)
                    # Clamp correlation to valid range
                    correlation = max(-1, min(1, correlation))
                    adjusted_cov.loc[ticker1, ticker2] = correlation * target_vols[ticker1] * target_vols[ticker2]
                else:
                    adjusted_cov.loc[ticker1, ticker2] = 0
    cov_matrix = adjusted_cov
    trading_days = 252
    time_horizon = years * trading_days
    # Convert weights to numpy array aligned with mean_returns
    weight_array = np.array([weights.get(ticker, 0) for ticker in mean_returns.index])
    # Define target volatilities per asset (matching synthetic data with scenario adjustments)
    target_vols_daily = {}
    print("\n" + "="*70)
    print("📊 VOLATILITY SETTINGS (Base × Scenario Multiplier)")
    print("="*70)
    for ticker in mean_returns.index:
        asset_class = ticker_to_asset_class.get(ticker, 'Stock')
        # Get base volatility (same logic as synthetic data generation)
        if ticker in HIGH_VOL_TECH:
            base_vol = HIGH_VOL_TECH[ticker]
        elif ticker in MEDIUM_HIGH_VOL:
            base_vol = MEDIUM_HIGH_VOL[ticker]
        elif ticker in MEDIUM_VOL:
            base_vol = MEDIUM_VOL[ticker]
        elif ticker in LOW_VOL_DEFENSIVE:
            base_vol = LOW_VOL_DEFENSIVE[ticker]
        elif ticker in BOND_TICKERS or 'Bond' in asset_class or 'bond' in asset_class.lower():
            base_vol = 0.025
        elif ticker in COMMODITY_TICKERS or 'Commodity' in asset_class or 'commodity' in asset_class.lower():
            base_vol = 0.22
        elif 'Stock ETF' in asset_class or 'ETF' in asset_class:
            base_vol = 0.16
        else:
            base_vol = 0.18
        # Apply scenario volatility multiplier
        target_annual_vol = base_vol * scenario['vol_multiplier']
        target_vols_daily[ticker] = target_annual_vol / np.sqrt(252)
        print(f"  {ticker}: {target_annual_vol:.1%} (base: {base_vol:.1%} × {scenario['vol_multiplier']:.1f}x)")
    print("="*70 + "\n")
    # Use batch processing to avoid memory issues
    batch_size = 5000
    num_batches = (simulations + batch_size - 1) // batch_size
    # SANITY CHECK: Print actual daily returns being used
    print("\n=== SIMULATION SANITY CHECK ===")
    print("Daily mean returns being used in simulation:")
    for ticker in mean_returns.index:
        daily_return = mean_returns[ticker]
        annual_return = daily_return * 252
        print(f"  {ticker}: {daily_return:.6f} daily = {annual_return:.2%} annual")
    print("="*60 + "\n")
    print(f"Running {simulations:,} simulations in {num_batches} batches...")
    all_end_values = []
    sample_paths = {}
    # Progress bar for batches
    with tqdm(total=simulations, desc="Simulating", unit="sims") as pbar:
        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, simulations - batch_idx * batch_size)
            # Generate returns for each asset INDEPENDENTLY
            # Shape: (current_batch_size, time_horizon, n_assets)
            batch_returns = np.zeros((current_batch_size, time_horizon, len(mean_returns)))
            for i, ticker in enumerate(mean_returns.index):
                # Independent normal distribution for each asset
                batch_returns[:, :, i] = np.random.normal(
                    mean_returns.iloc[i],
                    target_vols_daily[ticker],
                    size=(current_batch_size, time_horizon)
                )
            # Calculate portfolio returns: weighted sum across assets
            portfolio_returns = np.dot(batch_returns, weight_array)
            # Calculate cumulative portfolio values
            # Shape: (current_batch_size, time_horizon + 1)
            cumulative_values = np.ones((current_batch_size, time_horizon + 1))
            cumulative_values[:, 1:] = np.cumprod(1 + portfolio_returns, axis=1)
            # Extract end values for this batch
            batch_end_values = cumulative_values[:, -1]
            all_end_values.extend(batch_end_values)
            # Store some sample paths for visualization
            # We'll collect paths throughout to get good percentile representation
            if batch_idx == 0 or batch_idx == num_batches // 2 or batch_idx == num_batches - 1:
                for i in range(min(3, current_batch_size)):
                    sample_paths[len(sample_paths)] = cumulative_values[i, :].tolist()
            # Update progress bar
            pbar.update(current_batch_size)
    # Convert to numpy array
    end_values = np.array(all_end_values)
    # Get representative paths for percentiles from all collected paths
    sorted_indices = np.argsort(end_values)
    p10_idx = sorted_indices[int(simulations * 0.1)]
    p30_idx = sorted_indices[int(simulations * 0.3)]
    p50_idx = sorted_indices[int(simulations * 0.5)]
    p70_idx = sorted_indices[int(simulations * 0.7)]
    p90_idx = sorted_indices[int(simulations * 0.9)]
    # We need to recalculate the specific percentile paths
    # For efficiency, we'll approximate by running 5 more simulations at the end values
    print("Calculating percentile paths (P10, P30, P50, P70, P90)...")
    target_end_values = {
        10: end_values[p10_idx],
        30: end_values[p30_idx],
        50: end_values[p50_idx],
        70: end_values[p70_idx],
        90: end_values[p90_idx]
    }
    percentile_paths = {}
    for percentile, target_value in target_end_values.items():
        # Generate realistic volatile path that reaches the target value
        path_found = False
        for attempt in range(100):  # Try up to 100 times to get close
            single_returns = np.random.multivariate_normal(
                mean_returns.values, 
                cov_matrix.values, 
                size=time_horizon
            )
            portfolio_return = np.dot(single_returns, weight_array)
            path = np.ones(time_horizon + 1)
            path[1:] = np.cumprod(1 + portfolio_return)
            # Check if this path ends close to target
            if abs(path[-1] - target_value) / target_value < 0.1:  # Within 10%
                percentile_paths[percentile] = path.tolist()
                path_found = True
                break
        if not path_found:
            # If we can't find a close match, scale a random path to match target
            single_returns = np.random.multivariate_normal(
                mean_returns.values, 
                cov_matrix.values, 
                size=time_horizon
            )
            portfolio_return = np.dot(single_returns, weight_array)
            path = np.ones(time_horizon + 1)
            path[1:] = np.cumprod(1 + portfolio_return)
            # Scale to match target
            scale_factor = target_value / path[-1]
            path = path * scale_factor
            percentile_paths[percentile] = path.tolist()
    print(f"Generated realistic volatile paths for percentiles: {list(percentile_paths.keys())}")
    return end_values, percentile_paths
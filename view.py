from tabulate import tabulate
import click
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, PercentFormatter # <-- Import PercentFormatter
import pandas as pd
import inquirer
import model
from dateutil.parser import parse as parse_date
from datetime import datetime
from colorama import Fore, Style

# --- Currency Formatting Helpers ---
def format_currency(value, currency='USD'):
    """Formats a value with the appropriate currency symbol."""
    symbol = '€' if currency == 'EUR' else '$'
    return f"{symbol}{value:,.2f}"

def get_currency_symbol(currency='USD'):
    """Returns the appropriate currency symbol."""
    return '€' if currency == 'EUR' else '$'

# --- Kleur-opmaak Functie ---

def format_gain_loss(value, is_percent=False, currency='USD'):
    """Formatteert een getal als een gekleurde string (rood/groen)."""
    if value > 0:
        color = Fore.GREEN
        sign = "+"
    elif value < 0:
        color = Fore.RED
        sign = ""
    else:
        color = Fore.WHITE
        sign = ""

    if is_percent:
        return f"{color}{sign}{value:,.2f}%{Style.RESET_ALL}"
    else:
        symbol = '€' if currency == 'EUR' else '$'
        return f"{color}{sign}{symbol}{value:,.2f}{Style.RESET_ALL}"

# --- Dashboard View ---

def display_dashboard(portfolio_df, currency=None):
    """
    Toont de "Yahoo Finance" stijl dashboard tabel.
    portfolio_df is het DataFrame van model.get_dashboard_data()
    currency: 'EUR', 'USD', or None (for dynamic display)
    """
    if portfolio_df.empty:
        click.echo("Your portfolio is empty. Add a transaction from the main menu.")
        return

    df_display = portfolio_df.copy()
    # Determine currency symbol for headers
    # If currency is specified, use that. Otherwise, check if all holdings are in same currency
    if currency == 'EUR':
        currency_symbol = '€'
    elif currency == 'USD':
        currency_symbol = '$'
    else:
        # Default to $ if mixed or not specified
        currency_symbol = '$'

    # Apply currency-aware formatting for gains
    if 'Currency' in df_display.columns:
        df_display[f"Day's Gain ({currency_symbol})"] = df_display.apply(lambda row: format_gain_loss(row["Day's Gain ($)"], is_percent=False, currency=row.get('Currency', 'USD')), axis=1)
        df_display[f"Total Gain ({currency_symbol})"] = df_display.apply(lambda row: format_gain_loss(row["Total Gain ($)"], is_percent=False, currency=row.get('Currency', 'USD')), axis=1)
    else:
        df_display[f"Day's Gain ({currency_symbol})"] = df_display["Day's Gain ($)"].apply(lambda x: format_gain_loss(x, is_percent=False))
        df_display[f"Total Gain ({currency_symbol})"] = df_display["Total Gain ($)"].apply(lambda x: format_gain_loss(x, is_percent=False))
    df_display["Day's Gain (%)"] = df_display["Day's Gain (%)"].apply(lambda x: format_gain_loss(x, is_percent=True))
    df_display["Total Gain (%)"] = df_display["Total Gain (%)"].apply(lambda x: format_gain_loss(x, is_percent=True))

    columns_to_show = [
        'Ticker',
        'Name',
        'Quantity',
        'Market Value',
        f"Day's Gain ({currency_symbol})",
        "Day's Gain (%)",
        f"Total Gain ({currency_symbol})",
        "Total Gain (%)",
        "Weight (%)"
    ]

    if 'Currency' in df_display.columns:
        df_display['Market Value'] = df_display.apply(lambda row: format_currency(row['Market Value'], row.get('Currency', 'USD')), axis=1)
    else:
        df_display['Market Value'] = df_display['Market Value'].map('${:,.2f}'.format)
    df_display['Weight (%)'] = df_display['Weight (%)'].map('{:,.2f}%'.format)

    click.echo("\n--- Portfolio Holdings ---")
    table = tabulate(
        df_display[columns_to_show],
        headers='keys',
        tablefmt='psql',
        showindex=False
    )
    click.echo(table)

# --- Performance Grafiek View ---

def plot_portfolio_performance(performance_df, chart_type, currency='Native'):
    """
    Toont een grafiek van de portfolio performance (Value of Return).
    """
    # <<< CORRECT INDENTATION HERE >>>
    if performance_df.empty:
        click.echo(click.style("No performance data to plot. (Check transaction dates)", fg='yellow'))
        return

    plt.figure(figsize=(12, 7))

    if chart_type == 'Value':
        plt.plot(performance_df.index, performance_df['Market Value'], label='Portfolio Market Value')
        plt.plot(performance_df.index, performance_df['Cost Basis'], label='Total Cost Basis', linestyle='--')
        plt.title(f'Portfolio Performance: Value vs. Cost ({currency})')
        plt.ylabel(f'Value ({currency})')
        currency_symbol = '€' if currency == 'EUR' else '$' if currency == 'USD' else '$'
        formatter = FuncFormatter(lambda x, pos: f'{currency_symbol}{x:,.0f}')
        plt.gca().yaxis.set_major_formatter(formatter)

    elif chart_type == 'Return':
        plt.plot(performance_df.index, performance_df['Return (%)'], label='Return (%)', color='green')
        plt.title('Portfolio Performance: Total Return')
        plt.ylabel('Return (%)')
        # Use PercentFormatter for the y-axis
        formatter = PercentFormatter()
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.axhline(0, color='grey', linestyle='--', linewidth=1)

    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- 'Add' menu functies ---

def prompt_for_graph_period(ticker):
    """Asks for period and plots individual asset graph."""
    period_choices = [
        ('1 Month', '1mo'), ('6 Months', '6mo'), ('1 Year', '1y'),
        ('5 Years', '5y'), ('Max', 'max'), ('<- Back', 'BACK')
    ]
    question = [inquirer.List('period', message=f"Select period for {ticker}", choices=period_choices, carousel=True)]
    answer = inquirer.prompt(question)
    if not answer or answer['period'] == 'BACK': return
    period = answer['period']
    click.echo(f"Fetching data for {ticker} ({period})...")
    data = model.get_historical_data(ticker, period)
    # Determine currency from ticker
    currency = model.get_currency_from_ticker(ticker)
    plot_price_history(data, [ticker], currency=currency)

def prompt_for_asset(rich_asset_database):
    """Shows hierarchical menu to add a BUY transaction."""
    asset_info = None
    while True:
        # Sort categories alphabetically (A-Z)
        asset_classes = sorted(list(rich_asset_database.keys()))
        menu_choices = asset_classes + ["--- Other (Manual Input) ---", "--- Cancel ---"]
        class_question = [inquirer.List('asset_class', message="Choose an asset category", choices=menu_choices, carousel=True)]
        class_answer = inquirer.prompt(class_question)
        if not class_answer or class_answer['asset_class'] == "--- Cancel ---": return None
        chosen_class = class_answer['asset_class']

        if chosen_class == "--- Other (Manual Input) ---":
            click.echo("Manual Input selected.")
            try:
                asset_info = {}
                asset_info['ticker'] = click.prompt('Ticker')
                asset_info['name'] = asset_info['ticker'] # Use ticker as name
                asset_info['sector'] = click.prompt('Sector')
                asset_info['asset_class'] = click.prompt('Asset Class')
            except click.Abort: return None
        else:
            # Sort assets by name (A-Z, case-insensitive)
            assets_in_class = sorted(rich_asset_database[chosen_class], key=lambda x: x['name'].lower())
            ticker_choices = []
            for asset in assets_in_class:
                price = asset.get('currentPrice', 0)
                change = asset.get('dayChangePct', 0)
                # Use correct currency symbol based on asset currency
                currency = asset.get('currency', 'USD')
                currency_symbol = '€' if currency == 'EUR' else '$'
                label = (f"{asset['name']:<28} | {asset['sector']:<12} | {currency_symbol}{price:,.2f} ({change:+.2f}%)")
                ticker_choices.append((label, asset['ticker']))
            ticker_choices.append(("<- Back to categories", "BACK_TO_CATEGORIES"))
            ticker_choices.append(("--- Cancel ---", "CANCEL"))
            ticker_question = [inquirer.List('ticker', message=f"Choose an asset from '{chosen_class}'", choices=ticker_choices, carousel=True)]
            ticker_answer = inquirer.prompt(ticker_question)
            if not ticker_answer or ticker_answer['ticker'] == 'CANCEL': return None
            if ticker_answer['ticker'] == 'BACK_TO_CATEGORIES': continue
            chosen_ticker = ticker_answer['ticker']
            asset_info = next(item for item in assets_in_class if item['ticker'] == chosen_ticker)

        while True:
            action_question = [inquirer.List('action', message=f"What to do with {asset_info['name']} ({asset_info['ticker']})?",
                                            choices=[('Add transaction', 'ADD'), ('View historical data', 'VIEW'), ('<- Back to asset list', 'BACK_TO_LIST')],
                                            carousel=False)]
            action_answer = inquirer.prompt(action_question)
            if not action_answer: return None
            action = action_answer['action']
            if action == 'ADD': break
            if action == 'VIEW':
                prompt_for_graph_period(asset_info['ticker'])
                continue
            if action == 'BACK_TO_LIST': break

        if action == 'BACK_TO_LIST': continue
        if action == 'ADD': break

    try:
        quantity = click.prompt('Quantity', type=float)
        purchase_price = click.prompt('Purchase Price', type=float)
        while True:
            date_str = click.prompt(f"Purchase Date (DD-MM-YYYY)")
            try:
                purchase_date = parse_date(date_str)
                if purchase_date > datetime.now():
                    click.echo("Error: Purchase date cannot be in the future. Please try again.")
                else:
                    break
            except ValueError:
                click.echo("Error: Invalid date format. Please try again (e.g., '2023-10-25').")

        final_asset_data = {
            'ticker': asset_info['ticker'], 'name': asset_info['name'],
            'sector': asset_info['sector'], 'asset_class': asset_info['asset_class'],
            'quantity': quantity, 'price': purchase_price, # Renamed key
            'date': purchase_date, # Renamed key
            'transaction_type': 'BUY' # Specify type
        }
        return final_asset_data
    except click.Abort:
        click.echo("\nAdd operation canceled.")
        return None
    except Exception as e:
        click.echo(f"An error occurred: {e}")
        return None

# --- Summary & Plot ---

def display_summary(summary_df, title, currency=None):
    """Displays summary table (Sector/Class)."""
    if summary_df.empty:
        return
    click.echo(f"\n--- {title} ---")
    df_display = summary_df.copy()
    # Determine currency symbol
    if currency == 'EUR':
        currency_symbol = '€'
    elif currency == 'USD':
        currency_symbol = '$'
    else:
        currency_symbol = '$'  # Default
    # Rename Total Gain ($) column to use correct currency symbol
    if 'Total Gain ($)' in df_display.columns:
        df_display.rename(columns={'Total Gain ($)': f'Total Gain ({currency_symbol})'}, inplace=True)
    if 'Currency' in df_display.columns:
        df_display['Market Value'] = df_display.apply(lambda row: format_currency(row['Market Value'], row.get('Currency', 'USD')), axis=1)
    else:
        df_display['Market Value'] = df_display['Market Value'].map('${:,.2f}'.format)
    df_display['Weight (%)'] = df_display['Weight (%)'].map('{:,.2f}%'.format)
    table = tabulate(df_display, headers='keys', tablefmt='psql', floatfmt=".2f", showindex=False)
    click.echo(table)

def plot_price_history(data, tickers, currency=None):
    """Plots historical price for one or more assets."""
    if data.empty:
        click.echo("Could not find data to plot.")
        return
    plt.figure(figsize=(12, 7))
    # Handle DataFrame from get_historical_data which has 'Date' and 'Close' columns
    if isinstance(data, pd.DataFrame):
        if 'Date' in data.columns and 'Close' in data.columns:
            # Single ticker historical data
            plt.plot(data['Date'], data['Close'], label=tickers[0])
        else:
            # Multiple tickers or different structure
            for col in data.columns:
                if col.upper() == 'CLOSE' and len(tickers) == 1:
                    plt.plot(data.index, data[col], label=tickers[0])
                elif col in tickers:
                    plt.plot(data.index, data[col], label=col)
    elif isinstance(data, pd.Series):
         # If it's a Series, data.name might be the ticker
         plt.plot(data.index, data, label=data.name if data.name else tickers[0])
    plt.title('Historical Price Performance')
    plt.xlabel('Date')
    currency_label = f' ({currency})' if currency else ''
    plt.ylabel(f'Adjusted Close Price{currency_label}')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Sell Menu ---

def prompt_for_sell_details(holdings_df):
    """Shows menu to select asset and details for selling."""
    if holdings_df.empty:
        click.echo("You have no holdings to sell.")
        return None

    holding_choices = []
    for _, row in holdings_df.iterrows():
        label = (
            f"{row['Name']:<25} ({row['Ticker']}) | "
            f"Qty: {row['Quantity']:<10.4f} | "
            f"Value: ${row['Market Value']:<15,.2f}"
        )
        holding_choices.append((label, row['Ticker']))
    holding_choices.append(("-- Cancel --", "CANCEL"))

    q_ticker = [inquirer.List('ticker', message="Select asset to sell", choices=holding_choices, carousel=True)]
    a_ticker = inquirer.prompt(q_ticker)
    if not a_ticker or a_ticker['ticker'] == 'CANCEL': return None
    ticker_to_sell = a_ticker['ticker']

    holding_info = holdings_df[holdings_df['Ticker'] == ticker_to_sell].iloc[0]
    max_qty = holding_info['Quantity']

    try:
        while True:
            quantity = click.prompt(f"Quantity to sell (Max: {max_qty:.4f})", type=float)
            if 0 < quantity <= max_qty:
                break
            else:
                click.echo(f"Error: Quantity must be between 0 and {max_qty:.4f}")

        sell_price = click.prompt('Sell Price per unit', type=float)

        while True:
            date_str = click.prompt(f"Sell Date (DD-MM-YYYY)")
            try:
                sell_date = parse_date(date_str)
                if sell_date > datetime.now():
                    click.echo("Error: Sell date cannot be in the future.")
                # Basic check: Ensure sell date is not before the first purchase? (Complex)
                # else if sell_date < first_purchase_date_for_ticker:
                #    click.echo("Error: Sell date cannot be before the first purchase.")
                else:
                    break
            except ValueError:
                click.echo("Error: Invalid date format.")

        return {
            'ticker': ticker_to_sell,
            'name': holding_info['Name'],
            'sector': holding_info['Sector'],
            'asset_class': holding_info['Asset Class'],
            'quantity': quantity,
            'price': sell_price, # Use 'price' key
            'date': sell_date,   # Use 'date' key
            'transaction_type': 'SELL'
        }
    except click.Abort:
        click.echo("\nSell operation canceled.")
        return None
    except Exception as e:
        click.echo(f"An error occurred: {e}")
        return None

# --- Risk View ---
# --- Transaction Table View ---

def display_transactions_table(transactions_df):
    """Displays table of all transactions."""
    if transactions_df.empty:
        click.echo("No transactions found.")
        return

    df_display = transactions_df.copy()
    if 'Currency' in df_display.columns:
        df_display['Price'] = df_display.apply(lambda row: format_currency(row['Price'], row.get('Currency', 'USD')), axis=1)
    else:
        df_display['Price'] = df_display['Price'].map('${:,.2f}'.format)
    if 'Currency' in df_display.columns:
        df_display['Value'] = df_display.apply(lambda row: format_currency(row['Value'], row.get('Currency', 'USD')), axis=1)
    else:
        df_display['Value'] = df_display['Value'].map('${:,.2f}'.format)
    # Ensure correct columns are selected
    columns_to_show = ['Date', 'Type', 'Ticker', 'Name', 'Quantity', 'Price', 'Value']
    df_display = df_display[columns_to_show] # Reorder/select columns

    click.echo("\n--- All Transactions (Sorted by Date) ---")
    table = tabulate(df_display, headers='keys', tablefmt='psql', showindex=False)
    click.echo(table)

# --- Remove Menu ---

def prompt_for_transaction_to_remove(transactions_df):
    """Shows menu to select a transaction to remove (for corrections)."""
    if transactions_df.empty:
        click.echo("No transactions to remove.")
        return None

    tx_choices = []
    # Sort by date for the menu display as well
    for _, row in transactions_df.sort_values(by='Date', ascending=False).iterrows():
        currency = row.get('Currency', 'USD')
        symbol = get_currency_symbol(currency)
        label = (f"{row['Date']} | {row['Type']} | {row['Ticker']:<7} | {row['Quantity']} @ {symbol}{row['Price']:,.2f}")
        tx_choices.append((label, row['tx_id']))
    tx_choices.append(("-- Cancel --", "CANCEL"))

    question = [inquirer.List('tx_id', message="Select a transaction to remove", choices=tx_choices, carousel=True)]
    answer = inquirer.prompt(question)
    if not answer or answer['tx_id'] == 'CANCEL': return None
    return answer['tx_id']

# --- Simulation Views ---

def plot_simulation_paths(paths_df):
    """Plots the percentile paths from the simulation."""
    if paths_df is None:
        click.echo("Could not find paths to plot.")
        return
    click.echo("\nOpening plot for percentile paths...")
    ax = paths_df.plot(figsize=(12, 7), title='Portfolio Value Simulation Over Time', grid=True)
    ax.set_xlabel('Years')
    ax.set_ylabel('Portfolio Value')

    def currency_formatter(x, pos): return f'${x:,.0f}'
    ax.yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    ax.legend()
    plt.show()

def display_simulation_results(end_values, percentile_paths, years, initial_value):
    """Displays simulation statistics."""
    if end_values is None:
        click.echo("Simulation could not be executed.")
        return
    click.echo(f"\n--- Monte Carlo Simulation Results ({years} Years) ---")
    mean_value = np.mean(end_values) * initial_value
    median_value = np.median(end_values) * initial_value
    
    # Calculate comprehensive percentiles
    percentile_1 = np.percentile(end_values, 1) * initial_value
    percentile_10 = np.percentile(end_values, 10) * initial_value
    percentile_30 = np.percentile(end_values, 30) * initial_value
    percentile_70 = np.percentile(end_values, 70) * initial_value
    percentile_90 = np.percentile(end_values, 90) * initial_value
    percentile_99 = np.percentile(end_values, 99) * initial_value
    min_value = np.min(end_values) * initial_value
    max_value = np.max(end_values) * initial_value
    click.echo(f"  Starting portfolio value: ${initial_value:,.2f}")
    click.echo(f"  Average final value:      ${mean_value:,.2f}")
    click.echo(f"  Median final value:       ${median_value:,.2f}")
    click.echo(f"\n📊 Complete Percentile Distribution:")
    click.echo(f"  {'Percentile':<20} {'Value':>15}")
    click.echo(f"  {'-'*35}")
    click.echo(click.style(f"  {'Worst Case (Min)':<20} ${min_value:>14,.2f}", fg='red', bold=True))
    click.echo(click.style(f"  {'P1 (1%)':<20} ${percentile_1:>14,.2f}", fg='red'))
    click.echo(f"  {'P10 (10%)':<20} ${percentile_10:>14,.2f}")
    click.echo(f"  {'P30 (30%)':<20} ${percentile_30:>14,.2f}")
    click.echo(click.style(f"  {'P50 (50% - Median)':<20} ${median_value:>14,.2f}", bold=True))
    click.echo(f"  {'P70 (70%)':<20} ${percentile_70:>14,.2f}")
    click.echo(f"  {'P90 (90%)':<20} ${percentile_90:>14,.2f}")
    click.echo(click.style(f"  {'P99 (99%)':<20} ${percentile_99:>14,.2f}", fg='green'))
    click.echo(click.style(f"  {'Best Case (Max)':<20} ${max_value:>14,.2f}", fg='green', bold=True))
    # Show range statistics
    range_80 = percentile_90 - percentile_10
    range_98 = percentile_99 - percentile_1
    total_range = max_value - min_value
    click.echo(f"\n📈 Range Statistics:")
    click.echo(f"  80% of outcomes (P10-P90): ${range_80:,.2f} spread")
    click.echo(f"  98% of outcomes (P1-P99):  ${range_98:,.2f} spread")
    click.echo(f"  Total range (Min-Max):     ${total_range:,.2f} spread")

def plot_simulation_results(end_values, percentile_paths, years, initial_value):
    """Plots the simulation results with 5 percentile paths showing realistic volatility."""
    if end_values is None or not percentile_paths:
        click.echo("Could not generate simulation chart.")
        return
    # Create time axis (in years)
    n_periods = len(percentile_paths[50])
    time_years = np.linspace(0, years, n_periods)
    # Convert relative values to actual dollar values
    p10_values = np.array(percentile_paths[10]) * initial_value
    p30_values = np.array(percentile_paths[30]) * initial_value
    p50_values = np.array(percentile_paths[50]) * initial_value
    p70_values = np.array(percentile_paths[70]) * initial_value
    p90_values = np.array(percentile_paths[90]) * initial_value
    # Apply light smoothing for better visualization (but keep volatility visible)
    window_size = max(1, len(p10_values) // 50)  # About 2% smoothing
    
    def light_smooth(values):
        if window_size > 1:
            return pd.Series(values).rolling(window=window_size, center=True, min_periods=1).mean().values
        return values
    p10_smooth = light_smooth(p10_values)
    p30_smooth = light_smooth(p30_values)
    p50_smooth = light_smooth(p50_values)
    p70_smooth = light_smooth(p70_values)
    p90_smooth = light_smooth(p90_values)
    # Create the plot with professional styling
    plt.figure(figsize=(14, 8))
    # Plot the 5 percentile lines with distinct colors
    plt.plot(time_years, p10_smooth, label='P10: Pessimistic', 
             color='#d62728', linewidth=2.5, alpha=0.9)
    plt.plot(time_years, p30_smooth, label='P30: Conservative', 
             color='#ff7f0e', linewidth=2.5, alpha=0.9)
    plt.plot(time_years, p50_smooth, label='P50: Median', 
             color='#1f77b4', linewidth=3, alpha=0.95)
    plt.plot(time_years, p70_smooth, label='P70: Optimistic', 
             color='#7fdb7f', linewidth=2.5, alpha=0.9)  # Light green
    plt.plot(time_years, p90_smooth, label='P90: Very Optimistic', 
             color='#0b6623', linewidth=2.5, alpha=0.9)  # Dark green
    plt.xlabel('Years', fontsize=13, fontweight='bold')
    plt.ylabel('Portfolio Value ($)', fontsize=13, fontweight='bold')
    plt.title(f'Monte Carlo Simulation: Portfolio Growth Over {years} Years', 
              fontsize=15, fontweight='bold', pad=20)
    # Format y-axis as currency
    def currency_formatter(x, pos):
        if x >= 1_000_000:
            return f'${x/1_000_000:.1f}M'
        elif x >= 1000:
            return f'${x/1000:.0f}K'
        else:
            return f'${x:.0f}'
    plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    # Set y-axis to start at 0
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
    plt.legend(loc='upper left', fontsize=11, framealpha=0.95)
    plt.tight_layout()
    plt.show()
    # Display comprehensive statistics below the chart
    click.echo("\n" + "="*70)
    click.echo("📊 SIMULATION END VALUES - COMPLETE DISTRIBUTION")
    click.echo("="*70)
    # Calculate all percentiles
    percentile_1 = np.percentile(end_values, 1) * initial_value
    percentile_10 = np.percentile(end_values, 10) * initial_value
    percentile_30 = np.percentile(end_values, 30) * initial_value
    median_value = np.percentile(end_values, 50) * initial_value
    percentile_70 = np.percentile(end_values, 70) * initial_value
    percentile_90 = np.percentile(end_values, 90) * initial_value
    percentile_99 = np.percentile(end_values, 99) * initial_value
    min_value = np.min(end_values) * initial_value
    max_value = np.max(end_values) * initial_value
    click.echo(f"\n{'Percentile':<25} {'End Value':>20}")
    click.echo("-" * 45)
    click.echo(click.style(f"{'Absolute Worst Case':<25} ${min_value:>18,.2f}", fg='red', bold=True))
    click.echo(click.style(f"{'P1 (1st percentile)':<25} ${percentile_1:>18,.2f}", fg='red'))
    click.echo(f"{'P10 (10th percentile)':<25} ${percentile_10:>18,.2f}")
    click.echo(f"{'P30 (30th percentile)':<25} ${percentile_30:>18,.2f}")
    click.echo(click.style(f"{'P50 (Median)':<25} ${median_value:>18,.2f}", bold=True))
    click.echo(f"{'P70 (70th percentile)':<25} ${percentile_70:>18,.2f}")
    click.echo(f"{'P90 (90th percentile)':<25} ${percentile_90:>18,.2f}")
    click.echo(click.style(f"{'P99 (99th percentile)':<25} ${percentile_99:>18,.2f}", fg='green'))
    click.echo(click.style(f"{'Absolute Best Case':<25} ${max_value:>18,.2f}", fg='green', bold=True))
    click.echo("\n" + "="*70)
    click.echo("📈 INTERPRETATION GUIDE")
    click.echo("="*70)
    click.echo(f"• Starting Value:    ${initial_value:,.2f}")
    click.echo(f"• Median Outcome:    ${median_value:,.2f} (50% chance of more, 50% of less)")
    click.echo(f"• 80% Range:         ${percentile_10:,.2f} to ${percentile_90:,.2f}")
    click.echo(f"• 98% Range:         ${percentile_1:,.2f} to ${percentile_99:,.2f}")
    click.echo(f"• Absolute Range:    ${min_value:,.2f} to ${max_value:,.2f}")
    click.echo("="*70 + "\n")
# --- Asset Assumptions Display ---
def display_asset_assumptions(assumptions_data):
    """Displays expected returns and volatility assumptions for all assets."""
    click.echo("\n" + "="*80)
    click.echo(f"📊 ASSET RETURN & VOLATILITY ASSUMPTIONS")
    click.echo(f"Scenario: {assumptions_data['scenario_name']}")
    click.echo(f"Volatility Multiplier: {assumptions_data['vol_multiplier']}x")
    click.echo("="*80)
    # Display asset class expected returns
    click.echo("\n" + "="*80)
    click.echo("📈 EXPECTED ANNUAL RETURNS BY ASSET CLASS")
    click.echo("="*80)
    returns = assumptions_data['returns']
    click.echo(f"\n{'Asset Class':<25} {'Expected Return':>20}")
    click.echo("-" * 45)
    # Sort by return value for better readability
    sorted_returns = sorted(returns.items(), key=lambda x: x[1], reverse=True)
    for asset_class, return_val in sorted_returns:
        color = 'green' if return_val > 0.05 else ('red' if return_val < 0 else 'yellow')
        click.echo(click.style(f"  {asset_class.capitalize():<23} {return_val:>+7.1%}", fg=color))
    # Display base volatilities
    click.echo("\n" + "="*80)
    click.echo("📊 BASE VOLATILITY BY ASSET CLASS (Annual)")
    click.echo("="*80)
    base_vol = assumptions_data['base_volatility']
    vol_multiplier = assumptions_data['vol_multiplier']
    click.echo(f"\n{'Asset Class':<25} {'Base Vol':>15} {'Scenario Vol':>18}")
    click.echo("-" * 58)
    for asset_class, vol in sorted(base_vol.items()):
        scenario_vol = vol * vol_multiplier
        if vol > 0.5:
            color = 'red'
        elif vol > 0.25:
            color = 'yellow'
        else:
            color = 'green'
        click.echo(click.style(
            f"  {asset_class.capitalize():<23} {vol:>7.1%}        {scenario_vol:>7.1%} ({vol_multiplier}x)",
            fg=color
        ))
    # Display ticker-specific volatilities
    click.echo("\n" + "="*80)
    click.echo("📊 TICKER-SPECIFIC VOLATILITY ASSUMPTIONS")
    click.echo("="*80)
    # High Vol Tech
    if assumptions_data['high_vol_tech']:
        click.echo(click.style("\n🔴 High Volatility Tech Stocks:", bold=True))
        click.echo(f"{'Ticker':<10} {'Base Vol':>12} {'Scenario Vol':>18}")
        click.echo("-" * 40)
        for ticker, vol in sorted(assumptions_data['high_vol_tech'].items()):
            scenario_vol = vol * vol_multiplier
            click.echo(click.style(
                f"  {ticker:<8} {vol:>7.1%}     {scenario_vol:>7.1%}",
                fg='red'
            ))
    # Medium-High Vol
    if assumptions_data['medium_high_vol']:
        click.echo(click.style("\n🟠 Medium-High Volatility Growth Stocks:", bold=True))
        click.echo(f"{'Ticker':<10} {'Base Vol':>12} {'Scenario Vol':>18}")
        click.echo("-" * 40)
        for ticker, vol in sorted(assumptions_data['medium_high_vol'].items()):
            scenario_vol = vol * vol_multiplier
            click.echo(click.style(
                f"  {ticker:<8} {vol:>7.1%}     {scenario_vol:>7.1%}",
                fg='yellow'
            ))
    # Medium Vol
    if assumptions_data['medium_vol']:
        click.echo(click.style("\n🟡 Medium Volatility Established Stocks:", bold=True))
        click.echo(f"{'Ticker':<10} {'Base Vol':>12} {'Scenario Vol':>18}")
        click.echo("-" * 40)
        for ticker, vol in sorted(assumptions_data['medium_vol'].items()):
            scenario_vol = vol * vol_multiplier
            click.echo(f"  {ticker:<8} {vol:>7.1%}     {scenario_vol:>7.1%}")
    # Low Vol Defensive
    if assumptions_data['low_vol_defensive']:
        click.echo(click.style("\n🟢 Low Volatility Defensive Stocks:", bold=True))
        click.echo(f"{'Ticker':<10} {'Base Vol':>12} {'Scenario Vol':>18}")
        click.echo("-" * 40)
        for ticker, vol in sorted(assumptions_data['low_vol_defensive'].items()):
            scenario_vol = vol * vol_multiplier
            click.echo(click.style(
                f"  {ticker:<8} {vol:>7.1%}     {scenario_vol:>7.1%}",
                fg='green'
            ))
    click.echo("\n" + "="*80)
    click.echo("💡 INTERPRETATION")
    click.echo("="*80)
    click.echo("• Expected Returns: Used for calculating mean portfolio growth")
    click.echo("• Base Volatility: Annualized standard deviation of returns")
    click.echo("• Scenario Vol: Base volatility × scenario multiplier")
    click.echo("• Higher volatility = more uncertainty in returns")
    click.echo("• Scenario multiplier adjusts volatility based on economic conditions")
    click.echo("="*80 + "\n")
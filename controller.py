import model
import view
import pandas as pd
import click
import colorama
import inquirer

def initialize_colorama():
    """Initializes colorama for Windows."""
    colorama.init(autoreset=True)

# --- AANGEPASTE NAAM ---

def handle_add_buy_transaction():
    """Handles the 'Add Buy Transaction' workflow."""
    click.echo(click.style("\n--- Add New Buy Transaction ---", fg='green'))
    click.echo("Fetching live price data for menu...")
    rich_asset_database = model.get_rich_asset_database()
    # prompt_for_asset geeft nu type='BUY' terug
    transaction_data = view.prompt_for_asset(rich_asset_database)
    if transaction_data:
        portfolio = model.load_portfolio()
        new_transaction = model.Transaction(**transaction_data)
        portfolio.add_transaction(new_transaction)
        model.save_portfolio(portfolio)
        click.echo(click.style(f"\nSuccess! Buy transaction for {transaction_data['ticker']} added.", fg='green'))
    click.pause("\nPress any key to return to the main menu...")

# --- *** NIEUWE SELL FUNCTIE *** ---

def handle_add_sell_transaction():
    """Handles the 'Add Sell Transaction' workflow."""
    click.echo(click.style("\n--- Add New Sell Transaction ---", fg='yellow'))

    portfolio = model.load_portfolio()
    if portfolio.is_empty():
        click.echo("Portfolio is empty. Cannot sell.")
        click.pause("\nPress any key to return to the main menu...")
        return

    # Haal huidige holdings op om te tonen in het menu
    holdings_df, _ = model.get_dashboard_data(portfolio)

    # Vraag welke te verkopen
    sell_data = view.prompt_for_sell_details(holdings_df)

    if sell_data:
        try:
            # Maak het Transaction object
            sell_transaction = model.Transaction(**sell_data)
            # Voeg toe (deze checkt nu of er genoeg is)
            portfolio.add_transaction(sell_transaction)
            # Sla op
            model.save_portfolio(portfolio)
            click.echo(click.style(f"\nSuccess! Sell transaction for {sell_data['ticker']} added.", fg='green'))
        except ValueError as e: # Vang de 'niet genoeg quantity' fout op
             click.echo(click.style(f"\nError: {e}", fg='red'))
        except Exception as e:
             click.echo(click.style(f"\nAn unexpected error occurred: {e}", fg='red'))

    click.pause("\nPress any key to return to the main menu...")

# --- Dashboard Functie ---

def handle_show_dashboard(display_currency=None, eur_usd_rate=None):
    """
    Toont het volledige "Yahoo Finance" stijl dashboard.
    """
    click.echo(click.style("\n--- Portfolio Dashboard ---", fg='cyan'))
    # Use currency from main.py parameter
    currency_choice = display_currency if display_currency in ['EUR', 'USD'] else None
    click.echo("Loading portfolio...")
    portfolio = model.load_portfolio()

    if portfolio.is_empty():
        view.display_dashboard(pd.DataFrame(), currency=currency_choice)
        click.pause("\nPress any key to return to the main menu...")
        return

    click.echo("Fetching live prices for dashboard...")
    portfolio_df, total_value = model.get_dashboard_data(portfolio, target_currency=currency_choice)

    if portfolio_df.empty:
        click.echo(click.style("Could not fetch dashboard data.", fg='red'))
        click.echo("Possible reasons:")
        click.echo("  - No internet connection")
        click.echo("  - Invalid tickers in portfolio")
        click.echo("  - yfinance API issues")
        click.echo("\nTry again in a moment, or check your portfolio holdings.")
        click.pause("\nPress any key to return to the main menu...")
        return

    # Display with correct currency symbol
    if currency_choice == 'EUR':
        currency_symbol = '€'
    elif currency_choice == 'USD':
        currency_symbol = '$'
    else:
        currency_symbol = ''  # Mixed currencies
    if currency_symbol:
        click.echo(click.style(f"\nTotal Portfolio Value: {currency_symbol}{total_value:,.2f}", bold=True))
    else:
        click.echo(click.style(f"\nTotal Portfolio Value: ${total_value:,.2f} (mixed currencies)", bold=True))

    try:
        if inquirer.confirm("Show portfolio performance chart?", default=False):
            chart_type_q = [inquirer.List('type', message="Chart type", choices=['Value (vs. Cost)', 'Return (%)'], carousel=False)]
            chart_type_a = inquirer.prompt(chart_type_q)
            chart_type = 'Value' if chart_type_a['type'] == 'Value (vs. Cost)' else 'Return'
            period_q = [inquirer.List('period', message="Time interval", choices=['1 Month', '1 Year', '5 Years', 'All'], default='1 Year', carousel=False)]
            period_a = inquirer.prompt(period_q)
            period_map = {'1 Month': '1mo', '1 Year': '1y', '5 Years': '5y', 'All': 'all'}
            period = period_map[period_a['period']]
            click.echo("Calculating portfolio performance... (this may take a moment)")
            # 1. Model: Haal de performance data op
            performance_df = model.get_portfolio_performance(portfolio, period, target_currency=currency_choice)

            # 2. View: Toon de grafiek
            display_currency = currency_choice if currency_choice else 'Native'
            view.plot_portfolio_performance(performance_df, chart_type, currency=display_currency)

    except Exception as e:
        click.echo(click.style(f"Could not generate chart: {e}", fg='red'))
        import traceback
        traceback.print_exc()  # For debugging

    # --- Sorteer-loop ---
    sort_column = 'Market Value'
    sort_ascending = False

    while True:
        sorted_df = portfolio_df.sort_values(by=sort_column, ascending=sort_ascending)
        view.display_dashboard(sorted_df, currency=currency_choice)

        sector_summary_df = model.get_summary_by_group(portfolio_df, 'Sector')
        class_summary_df = model.get_summary_by_group(portfolio_df, 'Asset Class')
        view.display_summary(sector_summary_df, "Overview by Sector", currency=currency_choice)
        view.display_summary(class_summary_df, "Overview by Asset Class", currency=currency_choice)

        if sort_column in ['Ticker', 'Name']: sort_dir_str = "(A-Z)" if sort_ascending else "(Z-A)"
        else: sort_dir_str = "(Low-High)" if sort_ascending else "(High-Low)"

        sort_question = [
            inquirer.List(
                'sort_by',
                message=f"Currently sorting by: {sort_column} {sort_dir_str}. Sort by:",
                choices=[
                    ('Market Value', 'Market Value'), ("Day's Gain (%)", "Day's Gain (%)"),
                    ("Total Gain (%)", "Total Gain (%)"), ('Ticker', 'Ticker'),
                    ('Name', 'Name'), ('<- Back to Main Menu', 'BACK')
                ], carousel=False
            )
        ]
        answer = inquirer.prompt(sort_question)

        if not answer or answer['sort_by'] == 'BACK': break
        new_column = answer['sort_by']

        if new_column == sort_column: sort_ascending = not sort_ascending
        else:
            sort_column = new_column
            if sort_column in ['Ticker', 'Name']: sort_ascending = True
            else: sort_ascending = False

    # --- Einde Sorteer-loop ---

# --- RISK FUNCTIE ---
# --- Remove & View Functies ---

def handle_remove_transaction():
    click.echo(click.style("\n--- Remove Transaction ---", fg='red'))
    portfolio = model.load_portfolio()
    transactions_df = portfolio.get_all_transactions_df()
    if transactions_df.empty:
        click.echo("No transactions to remove.")
        click.pause("\nPress any key to return to the main menu...")
        return
    tx_id_to_remove = view.prompt_for_transaction_to_remove(transactions_df)
    if tx_id_to_remove:
        success = portfolio.remove_transaction_by_id(tx_id_to_remove)
        if success:
            model.save_portfolio(portfolio)
            click.echo(click.style("Transaction successfully removed.", fg='green'))
        else:
            click.echo(click.style("Error: Could not find transaction ID.", fg='red'))
    click.pause("\nPress any key to return to the main menu...")

def handle_view_transactions(display_currency=None, eur_usd_rate=None):
    click.echo(click.style("\n--- View All Transactions ---", fg='cyan'))
    portfolio = model.load_portfolio()
    transactions_df = portfolio.get_all_transactions_df()
    view.display_transactions_table(transactions_df)
    click.pause("\nPress any key to return to the main menu...")

# --- Simulatie Functie ---

def handle_run_simulation():
    """Handles the 'Run Simulation' workflow."""
    click.echo(click.style("\n--- Run Monte Carlo Simulation ---", fg='cyan'))

    click.echo("Loading portfolio and fetching current data...")
    portfolio = model.load_portfolio()
    if portfolio.is_empty():
        click.echo(click.style("Portfolio is empty. Add assets with 'add' to run simulation.", fg='red'))
        click.pause("\nPress any key to return to the main menu...")
        return

    dashboard_df, _ = model.get_dashboard_data(portfolio)
    if dashboard_df.empty:
        click.echo(click.style("Could not fetch portfolio data. Simulation stopped.", fg='red'))
        click.pause("\nPress any key to return to the main menu...")
        return

    sim_df, total_value = model.get_simple_portfolio_df(dashboard_df)

    if total_value == 0:
        click.echo(click.style("Portfolio value is zero. Simulation stopped.", fg='red'))
        click.pause("\nPress any key to return to the main menu...")
        return

    click.echo(f"Portfolio start value: ${total_value:,.2f}")
    # Scenario selection
    click.echo("\n" + "="*60)
    click.echo("SELECT ECONOMIC SCENARIO")
    click.echo("="*60)
    scenarios_list = [
        ('Base Case (Normal Economy)', 'base_case'),
        ('AI Bubble Bursts', 'ai_bubble_burst'),
        ('AGI Breakthrough', 'agi_success'),
        ('Global Trade War', 'trade_war'),
        ('Financial Crisis 2.0', 'financial_crisis'),
        ('Green Energy Revolution', 'green_transition'),
    ]
    questions = [
        inquirer.List('scenario',
                     message="Choose economic scenario",
                     choices=scenarios_list,
                     default='base_case')
    ]
    try:
        answers = inquirer.prompt(questions)
        if not answers:
            click.echo("Simulation canceled.")
            return
        scenario_key = answers['scenario']
    except (KeyboardInterrupt, EOFError):
        click.echo("\nSimulation canceled.")
        return
    try:
        years = click.prompt('Number of years to simulate', type=int, default=10)
        sims = click.prompt('Number of simulation paths', type=int, default=100_000)
    except click.Abort:
        click.echo("Simulation canceled.")
        return

    click.echo(f"Simulating {years} years with {sims} paths...")

    end_values, percentile_paths = model.run_monte_carlo_simulation(sim_df, years, sims, scenario=scenario_key)
    view.display_simulation_results(end_values, percentile_paths, years, total_value)
    # Auto-show the graph
    click.echo("\nGenerating performance chart...")
    view.plot_simulation_results(end_values, percentile_paths, years, total_value)
    

    click.pause("\nPress any key to return to the main menu...")
# --- View Asset Assumptions ---
def handle_view_asset_assumptions():
    """Displays expected returns and volatility for all assets."""
    click.echo(click.style("\n--- Asset Return & Volatility Assumptions ---", fg='cyan'))
    # Ask which scenario to display
    scenarios_list = [
        ('Base Case (Normal Economy)', 'base_case'),
        ('AI Bubble Bursts', 'ai_bubble_burst'),
        ('AGI Breakthrough', 'agi_success'),
        ('Global Trade War', 'trade_war'),
        ('Financial Crisis 2.0', 'financial_crisis'),
        ('Green Energy Revolution', 'green_transition'),
    ]
    questions = [
        inquirer.List('scenario',
                     message="Select scenario to view assumptions",
                     choices=scenarios_list,
                     default='base_case')
    ]
    try:
        answers = inquirer.prompt(questions)
        if not answers:
            click.pause("\nPress any key to return to the main menu...")
            return
        scenario_key = answers['scenario']
    except (KeyboardInterrupt, EOFError):
        click.pause("\nPress any key to return to the main menu...")
        return
    # Get assumptions data
    assumptions = model.get_asset_assumptions_overview(scenario=scenario_key)
    # Display the data
    view.display_asset_assumptions(assumptions)
    click.pause("\nPress any key to return to the main menu...")
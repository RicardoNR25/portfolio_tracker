import inquirer
import controller 
import sys
import click 
import model

def main_menu():
    """Displays the main menu and handles user selection."""
    controller.initialize_colorama()
    display_currency = 'USD'
    eur_usd_rate = None 
    while True:
        print("\n" + "="*30)
        print("    PORTFOLIO TRACKER MAIN MENU")
        print(f"     (Displaying in: {display_currency})")
        print("="*30)
        switch_currency_text = "Switch Currency (to EUR)" if display_currency == 'USD' else "Switch Currency (to USD)"
        questions = [
            inquirer.List(
                'choice',
                message="What would you like to do?",
                choices=[
                    ('View Portfolio Dashboard', 'DASHBOARD'),
                    ('Add Buy Transaction', 'ADD_BUY'),
                    ('Add Sell Transaction', 'ADD_SELL'),
                    ('View All Transactions', 'VIEW_TX'),
                    ('Remove Transaction', 'REMOVE_TX'), 
                    ('Run Simulation', 'SIMULATE'),
                    ('View Simulation Assumptions', 'VIEW_ASSUMPTIONS'),
                    (switch_currency_text, 'SWITCH_CURRENCY'), 
                    ('Quit', 'QUIT')
                ],
                carousel=True
            )
        ]
        try:
            answer = inquirer.prompt(questions)
            if not answer: break 
            choice = answer['choice']

            # --- Haal wisselkoers op indien nodig ---
            if choice in ['DASHBOARD', 'VIEW_TX', 'ANALYZE_RISK'] and eur_usd_rate is None:
                try:
                    click.echo("Fetching live exchange rate (EUR/USD)...")
                    eur_usd_rate = model.get_eur_usd_rate()
                    if eur_usd_rate:
                        click.echo(f"EUR/USD rate: {eur_usd_rate:.4f}")
                except Exception as e:
                    click.echo(click.style(f"Error fetching exchange rate: {e}. Defaulting to 1.0", fg='red'))
                    eur_usd_rate = 1.0  # Fallback
            if choice == 'DASHBOARD':
                controller.handle_show_dashboard(display_currency, eur_usd_rate)
            elif choice == 'ADD_BUY':
                controller.handle_add_buy_transaction()
            elif choice == 'ADD_SELL':
                controller.handle_add_sell_transaction()
            elif choice == 'VIEW_TX':
                controller.handle_view_transactions(display_currency, eur_usd_rate)
            elif choice == 'VIEW_ASSUMPTIONS':
                controller.handle_view_asset_assumptions()
            elif choice == 'REMOVE_TX':
                controller.handle_remove_transaction() 
            elif choice == 'SIMULATE':
                controller.handle_run_simulation()
            elif choice == 'SWITCH_CURRENCY': 
                display_currency = 'EUR' if display_currency == 'USD' else 'USD'
                eur_usd_rate = None  # Reset rate when switching
                click.echo(f"Display currency set to {display_currency}.")
                click.pause("Press any key to continue...")
            elif choice == 'QUIT':
                click.echo("Goodbye!")
                sys.exit(0)
        except KeyboardInterrupt:
            click.echo("\nGoodbye!")
            sys.exit(0)
        except ValueError as ve: 
            click.echo(click.style(f"\nInput Error: {ve}", fg='yellow'))
            click.pause("Press any key to return to the main menu...")
        except Exception as e:
            click.echo(click.style(f"\nAn unexpected error occurred: {e}", fg='red'))
            click.echo("Returning to main menu.")
            import traceback
            traceback.print_exc()  # Helpful for debugging
if __name__ == "__main__":
    main_menu()
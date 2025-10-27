# Portfolio Tracker

A command-line interface application for tracking and analyzing investment portfolios with Monte Carlo simulation capabilities.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup Steps

1. Extract or clone the project files to a directory

2. Create a virtual environment:
```bash
# On Windows:
py -m venv venv

# On Mac/Linux:
python3 -m venv venv
```

3. Activate the virtual environment:

**On Windows (PowerShell):**

If you get a security error, first run:
```bash
Set-ExecutionPolicy RemoteSigned -Scope Process
```
Type Y and press Enter when prompted.

Then activate:
```bash
.\venv\Scripts\activate
```

**On Mac/Linux:**
```bash
source venv/bin/activate
```

You should now see `(venv)` before your prompt.

4. Install required dependencies:
```bash
pip install -r requirements.txt
```

5. Start the application:
```bash
# On Windows:
py main.py

# On Mac/Linux:
python3 main.py
```

## Usage

### Main Menu

When you start the application, the main menu appears with the following options:

**View Portfolio Dashboard**
- Displays all holdings with current market values
- Shows daily and total gains/losses
- Calculates portfolio weights
- Provides summaries by sector and asset class
- Interactive sorting options
- Optional performance charts (Value vs Cost or Return percentage)

**Add Buy Transaction**
- Select from categorized asset menu (stocks, ETFs, bonds, commodities)
- Live prices are fetched automatically
- Enter quantity and transaction date
- Transaction is saved to portfolio

**Add Sell Transaction**
- View current holdings
- Select asset to sell
- Enter quantity (validated against holdings)
- Transaction is saved to portfolio

**View All Transactions**
- Complete history of all buy and sell transactions
- Shows dates, quantities, and prices

**Remove Transaction**
- Select transaction to delete
- Portfolio automatically recalculates

**Run Simulation**
- Select economic scenario:
  - Base Case (Normal Economy)
  - AI Bubble Bursts
  - AGI Breakthrough
  - Global Trade War
  - Financial Crisis 2.0
  - Green Energy Revolution
- Set simulation parameters (years and number of paths)
- View percentile outcomes (P1, P10, P30, P50, P70, P90, P99)
- Visualize five percentile paths showing portfolio evolution
- Review range statistics and interpretation guide

**View Simulation Assumptions**
- Display expected returns by asset class
- Show volatility assumptions for selected scenario
- Review ticker-specific risk parameters

**Switch Currency**
- Toggle between USD and EUR display
- Live exchange rates are fetched automatically

### Supported Assets

The application includes:
- US stocks: AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, JPM, BRK-B, SHEL
- European stocks: ASML.AS, ADYEN.AS, UNA.AS, ASRNL.AS
- Stock ETFs: VOO, QQQ, VTI, EEM
- Bond ETFs: TLT, IEF, AGG
- Commodity ETFs: GLD, SLV, USO

### Data Storage

Portfolio data is automatically saved to `portfolio.json` in the application directory. This file is created on first transaction and updated after each change.



- Internet connection required for live price data (Yahoo Finance API)
- Simulations use 252 trading days per year
- Historical performance calculations require at least one transaction with a past date

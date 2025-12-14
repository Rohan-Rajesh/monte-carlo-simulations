# Monte Carlo Portfolio & Option Simulation

A comprehensive Python library for pricing financial derivatives and evaluating portfolio risk using Monte Carlo simulation techniques. This project implements industry-standard quantitative finance methods for pricing European options, exotic options with barriers, and analyzing multi-asset portfolio performance under various market scenarios.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Mathematical Foundations](#mathematical-foundations)
- [Risk Metrics](#risk-metrics)
- [Examples](#examples)

## Overview

Monte Carlo simulation is a powerful computational technique used in quantitative finance to:

1. **Price derivatives** by simulating thousands of possible future asset price paths
2. **Assess portfolio risk** by understanding the distribution of potential outcomes
3. **Value exotic options** with path-dependent features like barriers
4. **Calculate risk metrics** such as Value at Risk (VaR), Expected Shortfall (CVaR), and maximum drawdown

This project provides three complete implementations tailored for different use cases:

- **Portfolio Simulation**: Multi-asset portfolios with correlated assets
- **Simple Option Pricing**: Vanilla European call/put options
- **Exotic Option Pricing**: Barrier options with advanced volatility modeling

## Key Features

### Portfolio Simulation

- ✅ Multi-asset portfolio support with custom weights
- ✅ Asset correlation modeling via Cholesky decomposition
- ✅ Geometric Brownian Motion (GBM) price path generation
- ✅ Comprehensive risk metrics: expected return, volatility, Sharpe ratio, max drawdown
- ✅ Tail risk analysis: Value at Risk (VaR) and Conditional Value at Risk (CVaR)
- ✅ Advanced visualization: price paths, distributions, confidence intervals

### Simple Option Pricing

- ✅ European call and put option pricing
- ✅ Historical volatility computation from Alpha Vantage API
- ✅ Risk-neutral pricing under GBM
- ✅ Discounted payoff calculation
- ✅ Probability of loss and CVaR analysis

### Exotic Option Simulation

- ✅ Barrier options (up-and-out, down-and-out)
- ✅ Antithetic variance reduction for improved accuracy
- ✅ GARCH volatility modeling (arch library)
- ✅ Support for multiple market calendars (NSE, etc.)
- ✅ Dynamic trading day calculation with holiday adjustments
- ✅ Advanced risk metrics: variance, standard deviation, VaR, CVaR

## Installation

### Requirements

- Python 3.7+
- NumPy: Numerical computations
- Matplotlib: Visualization
- Pandas: Data manipulation
- Requests: API integration
- arch: GARCH volatility modeling (for exotic options)
- pandas_market_calendars: Trading calendar support (for exotic options)

### Setup

```bash
pip install numpy matplotlib pandas requests arch pandas-market-calendars
```

## Project Structure

```
monte-carlo-portfolio-simulation/
├── monte-carlo-portfolio-simulation.py    # Multi-asset portfolio simulator
├── monte-carlo-simple-option-simulation.py # Vanilla option pricing
├── monte-carlo-option-simulation.py        # Exotic option simulator
└── README.md                              # This file
```

## Usage

### 1. Portfolio Monte Carlo Simulation

Simulate a 4-stock equal-weighted portfolio with correlations:

```python
from monte_carlo_portfolio_simulation import Stock, MonteCarloPortfolioSimulator
import numpy as np

# Define stocks
stocks = [
    Stock("AAPL", expected_return=0.12, volatility=0.25, initial_price=150.0),
    Stock("GOOGL", expected_return=0.15, volatility=0.30, initial_price=2500.0),
    Stock("MSFT", expected_return=0.10, volatility=0.22, initial_price=300.0),
    Stock("TSLA", expected_return=0.20, volatility=0.45, initial_price=200.0)
]

# Portfolio weights (must sum to 1.0)
weights = [0.25, 0.25, 0.25, 0.25]

# Correlation matrix between assets
correlation_matrix = np.array([
    [1.00, 0.60, 0.65, 0.30],
    [0.60, 1.00, 0.55, 0.25],
    [0.65, 0.55, 1.00, 0.35],
    [0.30, 0.25, 0.35, 1.00]
])

# Initialize simulator
simulator = MonteCarloPortfolioSimulator(stocks, weights, correlation_matrix)

# Run simulation
price_paths = simulator.simulate_paths(n_simulations=10000, n_days=252)
portfolio_values = simulator.calculate_portfolio_values(price_paths)

# Calculate risk metrics
risk_metrics = simulator.calculate_risk_metrics(portfolio_values)

print(f"Expected Annual Return: {risk_metrics['expected_return']:.2%}")
print(f"Annual Volatility: {risk_metrics['volatility']:.2%}")
print(f"Sharpe Ratio: {risk_metrics['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {risk_metrics['max_drawdown']:.2%}")
print(f"Value at Risk (VaR): {risk_metrics['var']:.2%}")

# Visualize results
simulator.plot_simulation_results(portfolio_values)
```

### 2. Simple Option Pricing

Price a European call option with historical volatility:

```python
from monte_carlo_simple_option_simulation import MonteCarloPortfolioSimulator

# Initialize simulator for SPY call option with $624 strike
simulator = MonteCarloPortfolioSimulator(
    strike=624,
    volatility=0.25,  # or compute_historical_volatility("SPY")
    option_type="C",   # "C" for call, "P" for put
    risk_free_rate=0.04
)

# Generate price paths and calculate option value
price_paths = simulator.simulate_paths(
    initial_stock_price=624,
    n_simulations=10000,
    n_days=252
)

option_values = simulator.calculate_option_value(price_paths)
risk_metrics = simulator.calculate_risk_metrics(624, option_values)

print(f"Option Fair Value: ${np.mean(option_values):.2f}")
print(f"Probability of Loss: {risk_metrics['probability_of_loss']:.2%}")
print(f"Value at Risk (VaR): ${risk_metrics['var']:.2f}")

# Plot results
simulator.plot_simulation_results(price_paths, option_values)
```

### 3. Exotic Option with Barriers

Price a barrier option with GARCH volatility:

```python
from monte_carlo_option_simulation import MonteCarloOptionSimulator, compute_volatility
from datetime import datetime

# Compute dynamic volatility using GARCH
volatility = compute_volatility(stock_ticker="TCS.BSE")

# Initialize barrier option simulator
simulator = MonteCarloOptionSimulator(
    strike=3140,
    volatility=volatility,
    option_type="C",
    risk_free_rate=0.04,
    barrier=3500,            # Barrier level
    barrier_type="up-and-out" # Option expires if price crosses barrier
)

# Generate paths with antithetic variance reduction
price_paths, barrier_breached = simulator.simulate_paths(
    initial_stock_price=3140,
    n_simulations=20000,
    n_days=126
)

option_values = simulator.calculate_option_value(price_paths, barrier_breached)
risk_metrics = simulator.calculate_risk_metrics(3140, option_values)

print(f"Barrier Option Value: ${np.mean(option_values):.2f}")
print(f"Volatility (GARCH): {volatility:.2%}")
print(f"Standard Deviation: ${risk_metrics['std']:.2f}")

simulator.plot_simulation_results(price_paths, option_values)
```

## Mathematical Foundations

### Geometric Brownian Motion (GBM)

All simulations use the GBM model, which assumes log-normal distribution of asset prices:

$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

**Discrete approximation:**
$$S_{t+1} = S_t \cdot \exp\left(\left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma\sqrt{\Delta t} \cdot Z\right)$$

Where:

- $S_t$ = Asset price at time $t$
- $\mu$ = Expected return (drift)
- $\sigma$ = Volatility (standard deviation)
- $Z \sim N(0,1)$ = Standard normal random variable
- $\Delta t$ = Time step (typically 1/252 for daily returns)

### Cholesky Decomposition for Correlated Assets

To introduce realistic correlations between assets:

1. Compute correlation matrix $\rho$
2. Calculate Cholesky factorization: $L = \text{cholesky}(\rho)$
3. Transform independent shocks: $\text{Correlated} = \text{Independent} @ L^T$

This ensures simulated asset prices maintain specified correlation structure.

### Risk-Neutral Valuation

For option pricing, drift uses risk-free rate under no-arbitrage conditions:

$$\mu = r \quad \text{(not the expected return)}$$

Final option values are discounted back to present value:

$$V_0 = E[\text{Payoff}] \cdot e^{-r \cdot T}$$

### Variance Reduction Techniques

**Antithetic Sampling** (in exotic option simulator):

- Generate $N/2$ standard normal variables
- Mirror them: if $Z \sim N(0,1)$, also use $-Z$
- Reduces variance of estimator by ~50%

**GARCH Volatility** (in exotic option simulator):

- Better captures time-varying volatility
- More accurate pricing for longer-dated options
- Uses arch library for autoregressive conditional heteroskedasticity

## Risk Metrics

### Expected Return

Mean of terminal returns across all simulations:
$$E[R] = \frac{1}{N}\sum_{i=1}^{N}\frac{S_T^i - S_0}{S_0}$$

### Volatility (Standard Deviation)

Standard deviation of returns:
$$\sigma = \sqrt{\text{Var}(R)}$$

### Sharpe Ratio

Risk-adjusted return metric:
$$\text{Sharpe} = \frac{E[R] - r_f}{\sigma}$$

Where $r_f$ is the risk-free rate (default 0%)

### Maximum Drawdown

Largest peak-to-trough decline in portfolio value:
$$\text{MDD} = \min_{t} \frac{S_t - \max(S_{0:t})}{\max(S_{0:t})}$$

### Value at Risk (VaR)

5th percentile of the return distribution. Represents the worst-case loss at 95% confidence level.

### Conditional Value at Risk (CVaR) / Expected Shortfall

Average loss among simulations worse than VaR:
$$\text{CVaR} = E[R | R \leq \text{VaR}]$$

### Probability of Loss

Percentage of simulations with negative returns:
$$P(\text{Loss}) = \frac{\text{Count}(R < 0)}{N}$$

## Examples

### Portfolio Risk Analysis

```python
risk_metrics = {
    'expected_return': 0.1243,      # 12.43% annual return
    'volatility': 0.1854,            # 18.54% annual volatility
    'sharpe_ratio': 0.670,           # 0.67 return per unit risk
    'max_drawdown': -0.2841,         # Maximum loss of 28.41%
    'probability_of_loss': 0.3421,   # 34.21% chance of loss
    'var': -0.1892,                  # 5% VaR: -18.92% return
    'cvar': -0.2634                  # Expected loss if VaR exceeded: -26.34%
}
```

### Option Pricing Results

```python
# European Call Option (SPY)
Option Fair Value: $12.34
Probability of Loss (worthless): 35.2%
Value at Risk (95%): $0.00
Conditional Value at Risk: $0.00
```

### Barrier Option Comparison

```python
# Regular European Call
Call Value: $15.67

# Up-and-Out Barrier Call (barrier at 110%)
Barrier Call Value: $8.92
```

The barrier option is cheaper because it expires if the stock price crosses the barrier level.

## Advanced Features

### Historical Volatility (Alpha Vantage API)

```python
from monte_carlo_simple_option_simulation import compute_historical_volatility

# Fetches 21-day and 5-day rolling volatility
# Uses weighted average: 70% long-term + 30% short-term
volatility = compute_historical_volatility(stock_ticker="SPY")
```

### GARCH Volatility Modeling

```python
from monte_carlo_option_simulation import compute_volatility

# More sophisticated volatility estimate
# Uses 1000 most recent returns for model fitting
volatility = compute_volatility(stock_ticker="TCS.BSE", trading_days=252)
```

### Market Calendar Support

```python
from monte_carlo_option_simulation import calculate_trading_days_with_holidays

# Calculate actual trading days between two dates (excludes weekends/holidays)
n_days = calculate_trading_days_with_holidays(
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 7, 31)
)
```

## Notes

- **Simulation Accuracy**: Use 10,000+ simulations for production use (accuracy scales with √N)
- **Time Steps**: Default 252 trading days per year; adjust for different markets
- **Random Seed**: Set `np.random.seed()` for reproducible results
- **Computational Cost**: Simulation time scales linearly with simulations × time steps × assets
- **API Rate Limits**: Alpha Vantage has rate limits; cache results when possible

## License

This project is provided as-is for educational and research purposes.

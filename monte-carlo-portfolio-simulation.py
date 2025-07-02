#!/usr/bin/env python3
"""
Monte Carlo Portfolio Simulation

This module provides functionality for Monte Carlo simulation of portfolio performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import requests
import time

def getData(stocks, start, end, api_key):
    stockData = pd.DataFrame()
    
    for symbol in stocks:
        # Alpha Vantage API endpoint for daily adjusted data
        url = f'https://www.alphavantage.co/query'
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'apikey': api_key,
            'outputsize': 'full',  # Get full historical data
            'datatype': 'json'
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                print(f"Error fetching {symbol}: {data['Error Message']}")
                continue
            elif 'Note' in data:
                print(f"API call frequency limit reached for {symbol}")
                time.sleep(60)  # Wait 1 minute and retry
                response = requests.get(url, params=params)
                data = response.json()
            
            # Extract time series data
            time_series = data.get('Time Series (Daily)', {})
            
            if not time_series:
                print(f"No data available for {symbol}")
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Use adjusted close price
            df['Close'] = df['4. close'].astype(float)
            
            # Filter by date range
            start_date = pd.to_datetime(start)
            end_date = pd.to_datetime(end)
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            stockData[symbol] = df['Close']
            
            time.sleep(12) 
            
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue
    
    if stockData.empty:
        raise ValueError("No stock data retrieved successfully")
    
    returns = stockData.pct_change().dropna()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    
    return meanReturns, covMatrix

def print_simulation_results(results, initial_value):
    """Print formatted simulation results"""
    print("\n" + "="*60)
    print("MONTE CARLO SIMULATION RESULTS")
    print("="*60)
    
    print(f"\nPortfolio Value Statistics:")
    print(f"Initial Value:           ${initial_value:,.2f}")
    print(f"Mean Final Value:        ${results['mean_final_value']:,.2f}")
    print(f"Median Final Value:      ${results['median_final_value']:,.2f}")
    print(f"Standard Deviation:      ${results['std_final_value']:,.2f}")
    print(f"Minimum Value:           ${results['min_final_value']:,.2f}")
    print(f"Maximum Value:           ${results['max_final_value']:,.2f}")
    
    print(f"\nRisk Metrics:")
    print(f"95% VaR:                 ${results['var_95']:,.2f}")
    print(f"99% VaR:                 ${results['var_99']:,.2f}")
    print(f"95% CVaR (Expected Shortfall): ${results['cvar_95']:,.2f}")
    
    print(f"\nProbability Analysis:")
    print(f"Probability of Loss:     {results['prob_loss']:.2%}")
    print(f"Probability of >10% Gain: {results['prob_gain_10pct']:.2%}")
    
    print(f"\nAnnualized Metrics:")
    print(f"Expected Return:         {results['annualized_return']:.2%}")
    print(f"Volatility:              {results['annualized_volatility']:.2%}")

def plot_simulation_results(results, initial_value):
    """Create visualization of simulation results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Sample portfolio paths
    paths_to_plot = min(100, results['portfolio_paths'].shape[0])
    for i in range(paths_to_plot):
        ax1.plot(results['portfolio_paths'][i], alpha=0.1, color='blue')
    ax1.axhline(y=initial_value, color='red', linestyle='--', label='Initial Value')
    ax1.set_title('Sample Portfolio Paths')
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()

    plt.tight_layout()
    plt.show()

def main():
	stockList  = ['AAPL', 'MSFT', 'GOOGL']
	endDate = dt.datetime.now()
	startDate = endDate - dt.timedelta(days = 250)

	meanReturns, covMatrix = getData(stockList, startDate, endDate, "VAIAU25I3ZOI1AHO")

	weights = np.random.random(len(meanReturns))
	weights /= np.sum(weights)

	numSims = 100
	T = 252
	initialPortfolioValue = 100000  

	simulations = np.zeros((numSims, T + 1))
	simulations[:, 0] = initialPortfolioValue  

	for i in range(numSims):
		randomReturns = np.random.multivariate_normal(meanReturns, covMatrix, T)
		weightedReturns = np.dot(randomReturns, weights)

		for t in range(T):
			simulations[i, t + 1] = simulations[i, t] * (1 + weightedReturns[t])

	finalPortfolio = simulations[:, -1]

	results = {
        'portfolio_paths': simulations,
        'final_values': finalPortfolio,
        'mean_final_value': np.mean(finalPortfolio),
        'median_final_value': np.median(finalPortfolio),
        'std_final_value': np.std(finalPortfolio),
        'min_final_value': np.min(finalPortfolio),
        'max_final_value': np.max(finalPortfolio),
        
        # Risk metrics
        'var_95': np.percentile(finalPortfolio, 5),  # 95% VaR
        'var_99': np.percentile(finalPortfolio, 1),  # 99% VaR
        'cvar_95': np.mean(finalPortfolio[finalPortfolio <= np.percentile(finalPortfolio, 5)]),
        
        # Return metrics
        'total_returns': (finalPortfolio - initialPortfolioValue) / initialPortfolioValue,
        'prob_loss': np.sum(finalPortfolio < initialPortfolioValue) / numSims,
        'prob_gain_10pct': np.sum(finalPortfolio > initialPortfolioValue * 1.1) / numSims,
        
        # Annualized metrics (assuming 252 trading days per year)
        'annualized_return': ((np.mean(finalPortfolio) / initialPortfolioValue) ** (252/T)) - 1,
        'annualized_volatility': np.std((finalPortfolio - initialPortfolioValue) / initialPortfolioValue) * np.sqrt(252/T)
    }

	print_simulation_results(results, initialPortfolioValue)
	plot_simulation_results(results, initialPortfolioValue)
    
	return results


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
from datetime import datetime
import pandas_market_calendars as mcal
import matplotlib.pyplot as plt
from arch import arch_model
import requests
from typing import Dict

class MonteCarloOptionSimulator:
    """
    Monte Carlo Simulation for an option on a stock.
    """
    
    def __init__(self, strike, volatility, option_type = "C" , risk_free_rate = 0.04, barrier: float = None, barrier_type: str = None):
        """
        Initialize the Monte Carlo option simulator.
        
        Args:
            strike: The strike price of the option being priced
            option_type: Either a call/put option, defaults to a call option
            risk_free_rate: The current risk free rate that all assets must earn under no-arbitrate conditions
        
        Raises:
            ValueError: If the option type isn't Call or Put
        """
        
        self.strike = strike
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.barrier = barrier
        self.barrier_type = barrier_type

        if (option_type not in ["C", "P"]):
            raise ValueError("Option Type must be either 'C' for call or 'P' for put")
        self.option_type = option_type
        
    def simulate_paths(self, initial_stock_price, n_simulations: int, n_days: int, 
                      trading_days: int = 252) -> np.ndarray:
        """
        Generate Monte Carlo price paths for prices that the stock can take.
        
        Args:
            n_simulations: Number of Monte Carlo simulations
            n_days: Number of trading days to simulate
            trading_days: Trading days per year (default 252)
        
        Returns:
            Dictionary with stock symbols as keys and price paths as values
        """
        dt = 1.0 / trading_days
        drift = (self.risk_free_rate - 0.5 * self.volatility**2) * dt
        
        # Initialize price paths with initial stock value
        price_paths = np.zeros((n_simulations, n_days + 1))
        price_paths[:, 0] = initial_stock_price
        # Track which paths have breached the barrier
        barrier_breached = np.zeros(n_simulations, dtype=bool)
        
        for day in range(n_days):
            # Generate random independent shocks
            base_shocks = np.random.standard_normal(n_simulations // 2)
            antithetic_shocks = -base_shocks
            independent_shocks = np.concatenate([base_shocks, antithetic_shocks])
            
            diffusion = self.volatility * np.sqrt(dt) * independent_shocks
            
            # Apply geometric Brownian motion
            price_paths[:, day + 1] = price_paths[:, day] * np.exp(drift + diffusion)

            if self.barrier is not None and self.barrier_type is not None:
                if self.barrier_type == "up-and-out":
                    barrier_breached |= (price_paths[:, day + 1] >= self.barrier)
                elif self.barrier_type == "down-and-out":
                    barrier_breached |= (price_paths[:, day + 1] <= self.barrier)
        
        return price_paths, barrier_breached
    
    def calculate_option_value(self, price_paths: np.ndarray, barrier_breached) -> np.ndarray:
        """
        Calculate final option value from stock price paths.
        
        Args:
            price_paths: Dictionary of stock price paths
        
        Returns:
            Final option payoffs of all simulations
        """
        final_values = price_paths[:, -1]
        
        # Calculate payoffs based on option type
        if self.option_type == "C":
            final_payoffs = np.maximum(final_values - self.strike, 0)
        else:  # option_type == "P"
            final_payoffs = np.maximum(self.strike - final_values, 0)

        if (barrier_breached is not None):
            final_payoffs = np.where(barrier_breached, 0, final_payoffs)
            
        trading_days = 252
        time_to_expiry = (len(price_paths[0]) - 1) / trading_days

        discounted_option_values = final_payoffs * np.exp(-self.risk_free_rate * time_to_expiry)
        return discounted_option_values
    
    def calculate_risk_metrics(self, initial_stock_price, final_option_values: np.ndarray) -> Dict:
        """
        Calculate risk metrics for option simulation results.
        
        Args:
            initial_stock_price: The initial price of the underlying stock
            final_option_values: Array of discounted final option values from simulation
        
        Returns:
            Dictionary containing risk metrics: probability of loss, VaR, CVaR, median return
        """
        probability_of_loss = np.mean(final_option_values == 0)
        variance = np.var(final_option_values)
        std = np.std(final_option_values)
        var = np.percentile(final_option_values, 5)
        cvar = final_option_values[final_option_values <= var].mean() if np.any(final_option_values <= var) else 0.0
        
        metrics = {
            'probability_of_loss': probability_of_loss,
            'variance': variance,
            'std': std,
            'var': var,
            'cvar': cvar,
        }

        return metrics
    
    def plot_simulation_results(self, price_paths, final_option_values: np.ndarray, 
                              n_paths_to_plot: int = 100) -> None:
        """
        Create comprehensive visualization of simulation results.
        
        Args:
            price_paths: Monte carlo simulation prices of the stock
            final_option_values: Final option payoffs of all simulations
            n_paths_to_plot: Number of individual paths to display
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        
        # Sample stock paths
        sample_indices = np.random.choice(
            price_paths.shape[0], 
            min(n_paths_to_plot, price_paths.shape[0]), 
            replace=False
        )
        
        for idx in sample_indices:
            axes[0].plot(price_paths[idx, :], alpha=0.3, color='blue', linewidth=0.5)
        
        # Plot mean path
        mean_path = np.mean(price_paths, axis=0)
        axes[0].plot(mean_path, color='red', linewidth=2, label='Mean Path')
        
        # Plot confidence intervals
        percentile_5 = np.percentile(price_paths, 5, axis=0)
        percentile_95 = np.percentile(price_paths, 95, axis=0)
        axes[0].fill_between(range(len(mean_path)), percentile_5, percentile_95, 
                      alpha=0.2, color='red', label='90% Confidence Interval')
        axes[0].set_title('Monte Carlo Price Paths')
        axes[0].set_xlabel('Trading Days')
        axes[0].set_ylabel('Stock Value ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Final option value distribution
        final_values = final_option_values
        axes[1].hist(final_values, bins=50, alpha=0.7, color='green', density=True)
        axes[1].axvline(np.mean(final_values), color='red', linestyle='--', 
              label=f'Mean: ${np.mean(final_values):.2f}')
        axes[1].set_title('Final Option Value Distribution')
        axes[1].set_xlabel('Option Value ($)')
        axes[1].set_ylabel('Density')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def compute_volatility(stock_ticker = "SPY", trading_days=252):
    """

    """
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock_ticker}&outputsize=full&apikey=VAIAU25I3ZOI1AHO'
    r = requests.get(url)
    data = r.json()
    data = data['Time Series (Daily)']
    df = pd.DataFrame.from_dict(data, orient='index')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    close_prices = df['4. close'].astype(float)
    returns = np.log(close_prices / close_prices.shift(1))
    returns = returns.dropna()

    if len(returns) > 2000:
        returns = returns[-1000:]

    # GARCH
    # Validated parameters
    model = arch_model(returns, mean='Zero', vol='GARCH', p=1, q=1)
    model_fit = model.fit(disp="off")
    predicted_values = model_fit.forecast(horizon=1)
    volatility = np.sqrt(predicted_values.variance.iloc[-1, 0] * trading_days)

    return volatility

def calculate_trading_days_with_holidays(end_date, start_date=datetime.now().date()):
    """
    Use pandas_market_calendars for NSE trading days
    """
    nse = mcal.get_calendar('NSE')
    trading_days = nse.valid_days(start_date=start_date, end_date=end_date)
    return len(trading_days) - 1

def main():
    risk_free_rate = 0.04
    volatility = compute_volatility(stock_ticker="TCS.BSE")
    n_days = calculate_trading_days_with_holidays(end_date="2025-07-31")
    strike = 3140
    initial_stock_price = 3140

    simulator = MonteCarloOptionSimulator(strike=strike, volatility=volatility, option_type="C", risk_free_rate=risk_free_rate)
    
    print("Running Monte Carlo simulation...")
    price_paths, barrier_breached = simulator.simulate_paths(initial_stock_price, n_simulations=20000, n_days=n_days, trading_days=252)
    final_option_values = simulator.calculate_option_value(price_paths, barrier_breached)
    
    risk_metrics = simulator.calculate_risk_metrics(initial_stock_price, final_option_values)
    
    print("="*50)
    print(f"Option Theoretical Value: {np.mean(final_option_values)}")
    print(f"Volatility: {volatility:.2%}")
    print(f"Probability of Loss: {risk_metrics['probability_of_loss']:.2%}")
    print(f"Variance: {risk_metrics['variance']:.2f}")
    print(f"Standard Deviation: {risk_metrics['std']:.2f}")
    print(f"Value at Risk (VaR): {risk_metrics['var']}")
    print(f"Conditional Value at Risk (CVaR): {risk_metrics['cvar']}")
    
    simulator.plot_simulation_results(price_paths, final_option_values)

if __name__ == "__main__":
    main()

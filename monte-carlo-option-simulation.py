import numpy as np
import matplotlib.pyplot as plt
import requests
from typing import Dict

class MonteCarloPortfolioSimulator:
    """
    Monte Carlo Simulation for a multi-asset weighted portfolio.
    """
    
    def __init__(self, strike, volatility, option_type = "C" , risk_free_rate = 0.04):
        """
        Initialize the Monte Carlo portfolio simulator.
        
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

        if (option_type not in ["C", "P"]):
            raise ValueError("Option Type must be either 'C' for call or 'P' for put")
        self.option_type = option_type
        
    def simulate_paths(self, initial_stock_price, n_simulations: int, n_days: int, 
                      trading_days: int = 252) -> np.ndarray:
        """
        Generate Monte Carlo price paths for all stocks in the portfolio.
        
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
        
        for day in range(n_days):
            # Generate random independent shocks
            base_shocks = np.random.standard_normal(n_simulations // 2)
            antithetic_shocks = -base_shocks
            independent_shocks = np.concatenate([base_shocks, antithetic_shocks])
            
            diffusion = self.volatility * np.sqrt(dt) * independent_shocks
            
            # Apply geometric Brownian motion
            price_paths[:, day + 1] = price_paths[:, day] * np.exp(drift + diffusion)
        
        return price_paths
    
    def calculate_option_value(self, price_paths: np.ndarray) -> np.ndarray:
        """
        Calculate portfolio value paths from individual stock price paths.
        
        Args:
            price_paths: Dictionary of stock price paths
        
        Returns:
            Portfolio value paths (n_simulations x n_days+1)
        """
        final_values = price_paths[:, -1]
        final_payoffs = np.maximum(final_values - self.strike, 0)
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
            portfolio_values: Portfolio value paths from simulation
            n_paths_to_plot: Number of individual paths to display
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        
        # Sample portfolio paths
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
        axes[0].set_title('Monte Carlo Portfolio Paths')
        axes[0].set_xlabel('Trading Days')
        axes[0].set_ylabel('Portfolio Value ($)')
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

def compute_historical_volatility(stock_ticker = "SPY"):
    """

    """
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock_ticker}&apikey=VAIAU25I3ZOI1AHO'
    r = requests.get(url)
    data = r.json()
    data = data['Time Series (Daily)']
    data = [float(data[date]['4. close']) for date in data.keys()]
    data = np.array(data)

    recent_21_prices = data[-21:]
    returns_21_day = np.log(recent_21_prices[1:] / recent_21_prices[:-1])
    vol_21_day = np.std(returns_21_day) * np.sqrt(252)
    recent_5_prices = data[-5:]
    returns_5_day = np.log(recent_5_prices[1:] / recent_5_prices[:-1])
    vol_5_day = np.std(returns_5_day) * np.sqrt(252)

    volatility = (0.7 * vol_21_day) + (0.3 * vol_5_day)

    return volatility



def main():
    risk_free_rate = 0.04
    volatility = compute_historical_volatility()
    strike = 624
    initial_stock_price = 624

    simulator = MonteCarloPortfolioSimulator(strike, volatility, "C", risk_free_rate)
    
    print("Running Monte Carlo simulation...")
    price_paths = simulator.simulate_paths(initial_stock_price, n_simulations=5000, n_days=252)
    final_option_values = simulator.calculate_option_value(price_paths)
    
    risk_metrics = simulator.calculate_risk_metrics(initial_stock_price, final_option_values)
    
    print("="*50)
    print(f"Option Theoretical Value: {np.mean(final_option_values)}")
    print(f"Probability of Loss: {risk_metrics['probability_of_loss']:.2%}")
    print(f"Variance: {risk_metrics['variance']:.2f}")
    print(f"Standard Deviation: {risk_metrics['std']:.2f}")
    print(f"Value at Risk (VaR): {risk_metrics['var']}")
    print(f"Conditional Value at Risk (CVaR): {risk_metrics['cvar']}")
    
    simulator.plot_simulation_results(price_paths, final_option_values)

if __name__ == "__main__":
    main()
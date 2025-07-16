import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class Stock:
    """
    Data class representing a stock with its statistical properties.
    
    Attributes:
        symbol: Stock ticker symbol
        expected_return: Annualized expected return (mu)
        volatility: Annualized volatility (sigma)
        initial_price: Starting price for simulation
    """
    symbol: str
    expected_return: float
    volatility: float
    initial_price: float

class MonteCarloPortfolioSimulator:
    """
    Monte Carlo Simulation for a multi-asset weighted portfolio.
    """
    
    def __init__(self, stocks: List[Stock], weights: List[float], 
                 correlation_matrix: Optional[np.ndarray] = None):
        """
        Initialize the Monte Carlo portfolio simulator.
        
        Args:
            stocks: List of Stock objects
            weights: Portfolio weights (must sum to 1.0)
            correlation_matrix: Asset correlation matrix
        
        Raises:
            ValueError: If weights don't sum to 1.0 or dimensions don't match
        """
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Portfolio weights must sum to 1.0")
        
        if len(stocks) != len(weights):
            raise ValueError("Number of stocks must match number of weights")
        
        self.stocks = stocks
        self.weights = np.array(weights)
        self.n_assets = len(stocks)
        
        # Validate correlation matrix
        if correlation_matrix is None:
            self.correlation_matrix = np.eye(self.n_assets)
        else:
            if correlation_matrix.shape != (self.n_assets, self.n_assets):
                raise ValueError("Correlation matrix dimensions don't match number of assets")
            self.correlation_matrix = correlation_matrix
        
        # Cholesky decomposition for correlated random variables
        self.cholesky_matrix = np.linalg.cholesky(self.correlation_matrix)
        
    def simulate_paths(self, n_simulations: int, n_days: int, 
                      trading_days: int = 252) -> Dict[str, np.ndarray]:
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
        
        # Initialize price paths with initial stock value
        price_paths = {}
        for stock in self.stocks:
            price_paths[stock.symbol] = np.zeros((n_simulations, n_days + 1))
            price_paths[stock.symbol][:, 0] = stock.initial_price
        
        for day in range(n_days):
            # Generate random independent shocks
            independent_shocks = np.random.standard_normal((n_simulations, self.n_assets))
            
            # Apply cholesky decomposition for correlated shocks
            correlated_shocks = independent_shocks @ self.cholesky_matrix.T
            
            for i, stock in enumerate(self.stocks):
                drift = (stock.expected_return - 0.5 * stock.volatility**2) * dt
                diffusion = stock.volatility * np.sqrt(dt) * correlated_shocks[:, i]
                
                # Apply geometric Brownian motion
                price_paths[stock.symbol][:, day + 1] = (
                    price_paths[stock.symbol][:, day] * np.exp(drift + diffusion)
                )
        
        return price_paths
    
    def calculate_portfolio_values(self, price_paths: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate portfolio value paths from individual stock price paths.
        
        Args:
            price_paths: Dictionary of stock price paths
        
        Returns:
            Portfolio value paths (n_simulations x n_days+1)
        """
        n_simulations, n_days_plus_one = next(iter(price_paths.values())).shape
        portfolio_values = np.zeros((n_simulations, n_days_plus_one))
        
        # Calculate initial portfolio value
        initial_portfolio_value = sum(
            weight * stock.initial_price 
            for weight, stock in zip(self.weights, self.stocks)
        )
        
        # Calculate portfolio values at each time step
        for day in range(n_days_plus_one):
            for i, stock in enumerate(self.stocks):
                portfolio_values[:, day] += (
                    self.weights[i] * initial_portfolio_value * 
                    price_paths[stock.symbol][:, day] / stock.initial_price
                )
        
        return portfolio_values
    
    def calculate_risk_metrics(self, portfolio_values: np.ndarray) -> Dict:
        """
        Calculate comprehensive risk metrics from portfolio simulation results.
        
        Args:
            portfolio_values: Portfolio value paths from simulation
        
        Returns:
            Dictionary containing various risk metrics
        """
        initial_value = portfolio_values[:, 0].mean()
        final_values = portfolio_values[:, -1]
        returns = (final_values - initial_value) / initial_value
        
        metrics = {
            'expected_return': np.mean(returns),
            'volatility': np.std(returns),
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'probability_of_loss': np.mean(returns < 0),
            'var': np.percentile(returns, 5),
            'cvar': returns[returns <= np.percentile(returns, 5)].mean()
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """
        Calculate maximum drawdown across all simulation paths.
        
        Args:
            portfolio_values: Portfolio value paths
        
        Returns:
            Maximum drawdown as a percentage
        """
        max_drawdowns = []
        
        for simulation in range(portfolio_values.shape[0]):
            path = portfolio_values[simulation, :]
            peak = np.maximum.accumulate(path)
            drawdown = (path - peak) / peak
            max_drawdowns.append(np.min(drawdown))
        
        return np.mean(max_drawdowns)
    
    def plot_simulation_results(self, portfolio_values: np.ndarray, 
                              n_paths_to_plot: int = 100) -> None:
        """
        Create comprehensive visualization of simulation results.
        
        Args:
            portfolio_values: Portfolio value paths from simulation
            n_paths_to_plot: Number of individual paths to display
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Sample portfolio paths
        sample_indices = np.random.choice(
            portfolio_values.shape[0], 
            min(n_paths_to_plot, portfolio_values.shape[0]), 
            replace=False
        )
        
        for idx in sample_indices:
            axes[0, 0].plot(portfolio_values[idx, :], alpha=0.3, color='blue', linewidth=0.5)
        
        # Plot mean path
        mean_path = np.mean(portfolio_values, axis=0)
        axes[0, 0].plot(mean_path, color='red', linewidth=2, label='Mean Path')
        
        # Plot confidence intervals
        percentile_5 = np.percentile(portfolio_values, 5, axis=0)
        percentile_95 = np.percentile(portfolio_values, 95, axis=0)
        axes[0, 0].fill_between(range(len(mean_path)), percentile_5, percentile_95, 
                               alpha=0.2, color='red', label='90% Confidence Interval')
        
        axes[0, 0].set_title('Monte Carlo Portfolio Paths')
        axes[0, 0].set_xlabel('Trading Days')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Final portfolio value distribution
        final_values = portfolio_values[:, -1]
        axes[0, 1].hist(final_values, bins=50, alpha=0.7, color='green', density=True)
        axes[0, 1].axvline(np.mean(final_values), color='red', linestyle='--', 
                          label=f'Mean: ${np.mean(final_values):.2f}')
        axes[0, 1].set_title('Final Portfolio Value Distribution')
        axes[0, 1].set_xlabel('Portfolio Value ($)')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Returns distribution
        initial_value = portfolio_values[:, 0].mean()
        returns = (final_values - initial_value) / initial_value
        axes[1, 0].hist(returns, bins=50, alpha=0.7, color='purple', density=True)
        axes[1, 0].axvline(np.mean(returns), color='red', linestyle='--', 
                          label=f'Mean Return: {np.mean(returns):.2%}')
        axes[1, 0].set_title('Portfolio Returns Distribution')
        axes[1, 0].set_xlabel('Return (%)')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Drawdown distribution
        drawdowns = []
        for simulation in range(portfolio_values.shape[0]):
            path = portfolio_values[simulation, :]
            peak = np.maximum.accumulate(path)
            drawdown = (path - peak) / peak
            drawdowns.append(np.min(drawdown))
        
        axes[1, 1].hist(drawdowns, bins=50, alpha=0.7, color='orange', density=True)
        axes[1, 1].axvline(np.mean(drawdowns), color='red', linestyle='--', 
                          label=f'Mean Max DD: {np.mean(drawdowns):.2%}')
        axes[1, 1].set_title('Maximum Drawdown Distribution')
        axes[1, 1].set_xlabel('Maximum Drawdown (%)')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    # Sample stocks (connect to API)
    stocks = [
        Stock("AAPL", expected_return=0.12, volatility=0.25, initial_price=150.0),
        Stock("GOOGL", expected_return=0.15, volatility=0.30, initial_price=2500.0),
        Stock("MSFT", expected_return=0.10, volatility=0.22, initial_price=300.0),
        Stock("TSLA", expected_return=0.20, volatility=0.45, initial_price=200.0)
    ]
    
    weights = [0.25, 0.25, 0.25, 0.25]
    correlation_matrix = np.array([
        [1.00, 0.60, 0.65, 0.30],
        [0.60, 1.00, 0.55, 0.25],
        [0.65, 0.55, 1.00, 0.35],
        [0.30, 0.25, 0.35, 1.00]
    ])
    
    simulator = MonteCarloPortfolioSimulator(stocks, weights, correlation_matrix)
    
    print("Running Monte Carlo simulation...")
    price_paths = simulator.simulate_paths(n_simulations=10000, n_days=252)
    portfolio_values = simulator.calculate_portfolio_values(price_paths)
    
    risk_metrics = simulator.calculate_risk_metrics(portfolio_values)
    
    print("\n" + "="*50)
    print("PORTFOLIO RISK ANALYSIS RESULTS")
    print("="*50)
    print(f"Expected Annual Return: {risk_metrics['expected_return']:.2%}")
    print(f"Annual Volatility: {risk_metrics['volatility']:.2%}")
    print(f"Sharpe Ratio: {risk_metrics['sharpe_ratio']:.3f}")
    print(f"Maximum Drawdown: {risk_metrics['max_drawdown']:.2%}")
    print(f"Probability of Loss: {risk_metrics['probability_of_loss']:.2%}")
    
    print(f"\nValue at Risk (VaR): {risk_metrics['var']}")
    print(f"\nConditional Value at Risk (CVaR): {risk_metrics['cvar']}")
    
    simulator.plot_simulation_results(portfolio_values)

if __name__ == "__main__":
    main()
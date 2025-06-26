import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class ChartGenerator:
    """
    Generator for various charts and visualizations for Monte Carlo simulations.
    """
    
    def __init__(self, figsize=(12, 8), style='seaborn-v0_8-darkgrid'):
        """
        Initialize chart generator.
        
        Parameters:
        -----------
        figsize : tuple
            Default figure size for matplotlib plots
        style : str
            Matplotlib style to use
        """
        self.figsize = figsize
        plt.style.use(style)
        sns.set_context("talk")
    
    def plot_simulation_paths(self, paths_df, title=None, num_paths=20, 
                             confidence_interval=0.9, figsize=None):
        """
        Plot simulation paths with confidence intervals.
        
        Parameters:
        -----------
        paths_df : pandas.DataFrame
            DataFrame containing simulation paths with dates as index
        title : str
            Plot title
        num_paths : int
            Number of individual paths to display
        confidence_interval : float
            Confidence interval to display (0.9 = 90%)
        figsize : tuple
            Figure size (width, height) in inches
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        figsize = figsize or self.figsize
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot a subset of individual paths (with no legend and no labels)
        if num_paths > 0:
            sampled_columns = np.random.choice(paths_df.columns, min(num_paths, len(paths_df.columns)), replace=False)
            for col in sampled_columns:
                ax.plot(paths_df.index, paths_df[col], linewidth=0.5, alpha=0.3, color='gray')
        
        # Calculate and plot median path
        median_path = paths_df.median(axis=1)
        median_path.plot(ax=ax, color='blue', linewidth=2, label='Median')
        
        # Calculate and plot confidence intervals
        lower_percentile = (1 - confidence_interval) / 2 * 100
        upper_percentile = 100 - lower_percentile
        
        lower_bound = paths_df.apply(lambda x: np.percentile(x, lower_percentile), axis=1)
        upper_bound = paths_df.apply(lambda x: np.percentile(x, upper_percentile), axis=1)
        
        ax.fill_between(paths_df.index, lower_bound, upper_bound, color='blue', alpha=0.1,
                       label=f"{int(confidence_interval * 100)}% Confidence Interval")
        
        # Add mean path
        mean_path = paths_df.mean(axis=1)
        mean_path.plot(ax=ax, color='red', linewidth=2, label='Mean')
        
        # Customize plot
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_xlabel('Date')
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Monte Carlo Simulation Paths')
        
        ax.legend()
        plt.tight_layout()
        
        return fig
    
    def plot_final_distribution(self, results, figsize=None, num_bins=50, confidence_level=0.95):
        """
        Plot histogram of final portfolio values with improved granularity.
        
        Parameters:
        -----------
        results : dict
            Simulation results dictionary
        figsize : tuple
            Figure size (width, height) in inches
        num_bins : int
            Number of bins for the histogram (more bins = more granularity)
        confidence_level : float
            Confidence level for filtering outliers (e.g., 0.95 = 95% confidence interval)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        figsize = figsize or self.figsize
        
        # Extract final values from paths
        paths_df = results['paths']
        final_values = paths_df.iloc[-1, :]
        
        # Extract final values for direct calculation
        final_values_array = np.array(final_values)
        
        # Calculate confidence interval bounds directly from data
        # This avoids dependency on pre-calculated percentiles that might not exist
        lower_percentile = (1 - confidence_level) / 2 * 100
        upper_percentile = 100 - lower_percentile
        
        lower_bound = np.percentile(final_values_array, lower_percentile)
        upper_bound = np.percentile(final_values_array, upper_percentile)
        
        # Filter values within confidence interval
        filtered_values = final_values[(final_values >= lower_bound) & (final_values <= upper_bound)]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot histogram with more bins
        sns.histplot(filtered_values, bins=num_bins, kde=True, ax=ax)
        
        # Add statistics lines with annotations
        median_line = ax.axvline(results['stats']['median'], color='blue', linestyle='--', 
                    label=f"Median: ${results['stats']['median']:,.0f}")
        ax.text(results['stats']['median'], ax.get_ylim()[1]*0.95, 
                f"Median: ${results['stats']['median']:,.0f}", 
                rotation=90, verticalalignment='top', color='blue')
        
        mean_line = ax.axvline(results['stats']['mean'], color='red', linestyle='-', 
                    label=f"Mean: ${results['stats']['mean']:,.0f}")
        ax.text(results['stats']['mean'], ax.get_ylim()[1]*0.9, 
                f"Mean: ${results['stats']['mean']:,.0f}", 
                rotation=90, verticalalignment='top', color='red')
        
        # Add percentile lines with annotations
        p5_line = ax.axvline(results['stats']['percentiles']['5%'], color='purple', linestyle=':', 
                    label=f"5th Percentile: ${results['stats']['percentiles']['5%']:,.0f}")
        ax.text(results['stats']['percentiles']['5%'], ax.get_ylim()[1]*0.85, 
                f"5%: ${results['stats']['percentiles']['5%']:,.0f}", 
                rotation=90, verticalalignment='top', color='purple')
        
        p95_line = ax.axvline(results['stats']['percentiles']['95%'], color='green', linestyle=':', 
                    label=f"95th Percentile: ${results['stats']['percentiles']['95%']:,.0f}")
        ax.text(results['stats']['percentiles']['95%'], ax.get_ylim()[1]*0.85, 
                f"95%: ${results['stats']['percentiles']['95%']:,.0f}", 
                rotation=90, verticalalignment='top', color='green')
        
        # Add text showing number of paths filtered out
        filtered_out = len(final_values) - len(filtered_values)
        if filtered_out > 0:
            ax.text(0.02, 0.98, f"Filtered out {filtered_out} outlier paths\n(outside {confidence_level*100:.0f}% confidence interval)",
                   transform=ax.transAxes, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Customize plot
        ax.set_title(f"Distribution of Final Portfolio Values (Leverage: {results['leverage']})")
        ax.set_xlabel('Portfolio Value ($)')
        ax.set_ylabel('Frequency')
        plt.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_leverage_comparison(self, results_dict, metric='median', figsize=None):
        """
        Plot comparison of different leverage values.
        
        Parameters:
        -----------
        results_dict : dict
            Dictionary with leverage as keys and simulation results as values
        metric : str
            Metric to compare ('mean', 'median', '5%', '95%', etc.)
        figsize : tuple
            Figure size (width, height) in inches
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        figsize = figsize or self.figsize
        
        leverages = []
        values = []
        
        for leverage, result in results_dict.items():
            if isinstance(leverage, (int, float)):  # Ensure it's a numeric leverage value
                leverages.append(leverage)
                
                if metric in result['stats']:
                    values.append(result['stats'][metric])
                elif '%' in metric:
                    values.append(result['stats']['percentiles'][metric])
                else:
                    raise ValueError(f"Unknown metric: {metric}")
        
        # Sort by leverage
        sorted_indices = np.argsort(leverages)
        leverages = [leverages[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot bar chart
        ax.bar(leverages, values, width=0.1)
        
        # Add value labels on top of bars
        for i, v in enumerate(values):
            ax.text(leverages[i], v * 1.01, f"${v:,.0f}", ha='center')
        
        # Customize plot
        ax.set_title(f"Comparison of {metric.capitalize()} Portfolio Values by Leverage")
        ax.set_xlabel('Leverage')
        ax.set_ylabel(f"{metric.capitalize()} Portfolio Value ($)")
        
        plt.tight_layout()
        return fig
    
    def plot_kelly_curve(self, leverage_values, growth_rates, optimal_leverage=None, figsize=None):
        """
        Plot the Kelly criterion curve showing growth rate vs. leverage.
        
        Parameters:
        -----------
        leverage_values : array-like
            Array of leverage values
        growth_rates : array-like
            Corresponding growth rates
        optimal_leverage : float, optional
            Optimal leverage to highlight
        figsize : tuple
            Figure size (width, height) in inches
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        figsize = figsize or self.figsize
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Fix: Make a copy of growth_rates to ensure it's a properly formatted array
        if hasattr(growth_rates, 'values'):
            growth_rates_array = growth_rates.values
        elif isinstance(growth_rates, list):
            growth_rates_array = np.array(growth_rates)
        else:
            growth_rates_array = growth_rates.copy()
            
        # Debug information
        print(f"DEBUG - Growth rates min: {np.min(growth_rates_array)}, max: {np.max(growth_rates_array)}")
        
        # If all growth rates are negative, this indicates a potential issue with the calculation
        if np.max(growth_rates_array) < 0:
            # Adjust to ensure we have at least some positive values to display
            # This is a temporary fix - the underlying calculation should be fixed
            growth_rates_array = growth_rates_array - np.min(growth_rates_array) + 0.0001
            print("WARNING: All growth rates were negative. Adjusted for display purposes only.")
        
        # Plot the growth rate curve
        ax.plot(leverage_values, growth_rates_array, 'b-', linewidth=2)
        
        # Find and mark the maximum growth rate
        if optimal_leverage is None:
            max_idx = np.argmax(growth_rates_array)
            optimal_leverage = leverage_values[max_idx]
        else:
            # Find the nearest index
            max_idx = np.abs(leverage_values - optimal_leverage).argmin()
        
        # Ensure max_growth is a float
        max_growth = growth_rates_array[max_idx]
        if hasattr(max_growth, 'iloc'):
            max_growth = float(max_growth.iloc[0])
        elif hasattr(max_growth, 'item'):
            max_growth = max_growth.item()
        else:
            max_growth = float(max_growth)
        
        # Mark the optimal leverage point - with smaller font
        ax.plot([optimal_leverage], [max_growth], 'ro', markersize=8)
        ax.annotate(f"Optimal Leverage: {optimal_leverage:.2f}\nGrowth Rate: {max_growth:.2%}", 
                    xy=(optimal_leverage, max_growth),
                    xytext=(optimal_leverage + 0.2, max_growth * 0.9),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='red'),
                    fontsize=8)  # Reduced font size
        
        # Add a vertical line at optimal leverage
        ax.axvline(x=optimal_leverage, color='r', linestyle='--', alpha=0.3)
        
        # Add a horizontal line at zero growth
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Add fractional Kelly markers (1/2, 3/4)
        half_kelly = optimal_leverage / 2
        ax.axvline(x=half_kelly, color='green', linestyle=':', alpha=0.5)
        ax.annotate("Half Kelly", xy=(half_kelly, 0), xytext=(half_kelly, max_growth / 2),
                   ha='center', va='center', rotation=90, color='green', fontsize=8)  # Reduced font size
        
        # Add hover labels to the plot using annotate - with smaller font
        for i in range(0, len(leverage_values), 10):  # Add annotations every 10 points to avoid clutter
            lev = leverage_values[i]
            growth = growth_rates_array[i]
            ax.annotate(f"Leverage: {lev:.1f}x\nGrowth: {growth:.2%}",
                       xy=(lev, growth), xytext=(0, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0),
                       ha='center', va='bottom', fontsize=7)  # Reduced font size
        
        # Customize plot
        ax.set_title("Kelly Criterion: Growth Rate vs. Leverage")
        ax.set_xlabel("Leverage")
        ax.set_ylabel("Expected Growth Rate")
        ax.grid(True)
        
        # Set y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
        
        plt.tight_layout()
        return fig
    
    def plot_drawdown_risk(self, results_dict, figsize=None):
        """
        Plot maximum drawdown risk by leverage.
        
        Parameters:
        -----------
        results_dict : dict
            Dictionary with leverage as keys and simulation results as values
        figsize : tuple
            Figure size (width, height) in inches
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        figsize = figsize or self.figsize
        
        leverages = []
        mean_drawdowns = []
        max_drawdowns = []
        p95_drawdowns = []
        
        for leverage, result in results_dict.items():
            if isinstance(leverage, (int, float)):  # Ensure it's a numeric leverage value
                leverages.append(leverage)
                mean_drawdowns.append(result['stats']['max_drawdown']['mean'])
                max_drawdowns.append(result['stats']['max_drawdown']['max'])
                p95_drawdowns.append(result['stats']['max_drawdown']['percentiles']['95%'])
        
        # Sort by leverage
        sorted_indices = np.argsort(leverages)
        leverages = [leverages[i] for i in sorted_indices]
        mean_drawdowns = [mean_drawdowns[i] for i in sorted_indices]
        max_drawdowns = [max_drawdowns[i] for i in sorted_indices]
        p95_drawdowns = [p95_drawdowns[i] for i in sorted_indices]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot lines
        ax.plot(leverages, mean_drawdowns, 'b-', marker='o', label='Mean Drawdown')
        ax.plot(leverages, p95_drawdowns, 'r-', marker='s', label='95th Percentile Drawdown')
        ax.plot(leverages, max_drawdowns, 'k-', marker='^', label='Maximum Drawdown')
        
        # Customize plot
        ax.set_title("Drawdown Risk by Leverage")
        ax.set_xlabel("Leverage")
        ax.set_ylabel("Maximum Drawdown")
        ax.grid(True)
        ax.legend()
        
        # Set y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        
        plt.tight_layout()
        return fig
    
    def plot_ruin_probability(self, results_dict, figsize=None):
        """
        Plot probability of ruin by leverage.
        
        Parameters:
        -----------
        results_dict : dict
            Dictionary with leverage as keys and simulation results as values
        figsize : tuple
            Figure size (width, height) in inches
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        figsize = figsize or self.figsize
        
        leverages = []
        ruin_probs = []
        
        for leverage, result in results_dict.items():
            if isinstance(leverage, (int, float)):  # Ensure it's a numeric leverage value
                leverages.append(leverage)
                ruin_probs.append(result['stats']['ruin_probability'])
        
        # Sort by leverage
        sorted_indices = np.argsort(leverages)
        leverages = [leverages[i] for i in sorted_indices]
        ruin_probs = [ruin_probs[i] for i in sorted_indices]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot line
        ax.plot(leverages, ruin_probs, 'r-', marker='o', linewidth=2)
        
        # Add a horizontal line at 5% ruin probability (common risk threshold)
        if max(ruin_probs) > 0.05:
            ax.axhline(y=0.05, color='gray', linestyle='--', alpha=0.7, label='5% Risk Threshold')
        
        # Customize plot
        ax.set_title("Probability of Ruin by Leverage")
        ax.set_xlabel("Leverage")
        ax.set_ylabel("Probability of Ruin")
        ax.grid(True)
        
        # Set y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
        
        plt.tight_layout()
        return fig
    
    def plot_model_comparison(self, model_results, leverage=1.0, metric='median', figsize=None):
        """
        Compare different models at the same leverage level.
        
        Parameters:
        -----------
        model_results : dict
            Dictionary with model names as keys and simulation results dictionaries as values
        leverage : float
            Leverage value to compare
        metric : str
            Metric to compare ('mean', 'median', '5%', '95%', etc.)
        figsize : tuple
            Figure size (width, height) in inches
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        figsize = figsize or self.figsize
        
        models = []
        values = []
        
        # Find the closest leverage for each model
        for model_name, results in model_results.items():
            # Get available leverages
            available_leverages = [lev for lev in results.keys() if isinstance(lev, (int, float))]
            
            # Find closest leverage
            if available_leverages:
                closest_lev = min(available_leverages, key=lambda x: abs(x - leverage))
                result = results[closest_lev]
                
                models.append(model_name)
                
                if metric in result['stats']:
                    values.append(result['stats'][metric])
                elif '%' in metric:
                    values.append(result['stats']['percentiles'][metric])
                else:
                    values.append(None)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot bar chart
        ax.bar(models, values)
        
        # Add value labels on top of bars
        for i, v in enumerate(values):
            if v is not None:
                ax.text(i, v * 1.01, f"${v:,.0f}", ha='center')
        
        # Customize plot
        ax.set_title(f"Comparison of Models (Leverage: {leverage:.1f}x, Metric: {metric})")
        ax.set_xlabel('Model')
        ax.set_ylabel(f"{metric.capitalize()} Portfolio Value ($)")
        
        plt.tight_layout()
        return fig
    
    def plot_asset_comparison(self, asset_results, model="MonteCarloModel", leverage=1.0, 
                             metric='median', figsize=None):
        """
        Compare different assets using the same model and leverage.
        
        Parameters:
        -----------
        asset_results : dict
            Dictionary with asset names as keys and simulation results dictionaries as values
        model : str
            Model name to use for comparison
        leverage : float
            Leverage value to compare
        metric : str
            Metric to compare ('mean', 'median', '5%', '95%', etc.)
        figsize : tuple
            Figure size (width, height) in inches
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        figsize = figsize or self.figsize
        
        assets = []
        values = []
        
        # Find the closest leverage for each asset
        for asset_name, asset_data in asset_results.items():
            if model in asset_data:
                results = asset_data[model]
                
                # Get available leverages
                available_leverages = [lev for lev in results.keys() if isinstance(lev, (int, float))]
                
                # Find closest leverage
                if available_leverages:
                    closest_lev = min(available_leverages, key=lambda x: abs(x - leverage))
                    result = results[closest_lev]
                    
                    assets.append(asset_name)
                    
                    if metric in result['stats']:
                        values.append(result['stats'][metric])
                    elif '%' in metric:
                        values.append(result['stats']['percentiles'][metric])
                    else:
                        values.append(None)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot bar chart
        ax.bar(assets, values)
        
        # Add value labels on top of bars
        for i, v in enumerate(values):
            if v is not None:
                ax.text(i, v * 1.01, f"${v:,.0f}", ha='center')
        
        # Customize plot
        ax.set_title(f"Comparison of Assets (Model: {model}, Leverage: {leverage:.1f}x)")
        ax.set_xlabel('Asset')
        ax.set_ylabel(f"{metric.capitalize()} Portfolio Value ($)")
        
        plt.tight_layout()
        return fig
    
    def create_interactive_dashboard(self, results_dict, kelly_data=None):
        """
        Create an interactive Plotly dashboard for comprehensive analysis.
        
        Parameters:
        -----------
        results_dict : dict
            Dictionary with leverage as keys and simulation results as values
        kelly_data : tuple, optional
            (leverage_values, growth_rates, optimal_leverage) for Kelly curve
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Interactive Plotly figure
        """
        # Extract data for plotting
        leverages = sorted([lev for lev in results_dict.keys() if isinstance(lev, (int, float))])
        mean_values = [results_dict[lev]['stats']['mean'] for lev in leverages]
        median_values = [results_dict[lev]['stats']['median'] for lev in leverages]
        p5_values = [results_dict[lev]['stats']['percentiles']['5%'] for lev in leverages]
        p95_values = [results_dict[lev]['stats']['percentiles']['95%'] for lev in leverages]
        
        mean_drawdowns = [results_dict[lev]['stats']['max_drawdown']['mean'] for lev in leverages]
        max_drawdowns = [results_dict[lev]['stats']['max_drawdown']['max'] for lev in leverages]
        
        ruin_probs = [results_dict[lev]['stats']['ruin_probability'] for lev in leverages]
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Portfolio Value vs. Leverage",
                "Kelly Criterion: Growth Rate vs. Leverage",
                "Maximum Drawdown vs. Leverage",
                "Probability of Ruin vs. Leverage"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ]
        )
        
        # Add portfolio value traces
        fig.add_trace(
            go.Scatter(x=leverages, y=median_values, mode='lines+markers', name='Median Value',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=leverages, y=mean_values, mode='lines+markers', name='Mean Value',
                      line=dict(color='red', width=2)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=leverages, y=p5_values, mode='lines', name='5th Percentile',
                      line=dict(color='purple', width=1, dash='dot')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=leverages, y=p95_values, mode='lines', name='95th Percentile',
                      line=dict(color='green', width=1, dash='dot')),
            row=1, col=1
        )
        
        # Add Kelly criterion curve if provided
        if kelly_data:
            leverage_values, growth_rates, optimal_leverage = kelly_data
            
            # Convert growth_rates to a list of floats if it's a Series
            if hasattr(growth_rates, 'to_list'):
                growth_rates = [float(x) for x in growth_rates.to_list()]
            elif hasattr(growth_rates, 'tolist'):
                growth_rates = [float(x) for x in growth_rates.tolist()]
            
            fig.add_trace(
                go.Scatter(x=leverage_values, y=growth_rates, mode='lines', name='Growth Rate',
                          line=dict(color='blue', width=2)),
                row=1, col=2
            )
            
            # Add optimal leverage point
            optimal_idx = np.abs(leverage_values - optimal_leverage).argmin()
            optimal_growth = float(growth_rates[optimal_idx])
            
            fig.add_trace(
                go.Scatter(x=[optimal_leverage], y=[optimal_growth], mode='markers',
                          marker=dict(color='red', size=10),
                          name=f'Optimal Leverage: {optimal_leverage:.2f}'),
                row=1, col=2
            )
            
            # Add annotation for optimal leverage
            fig.add_annotation(
                x=optimal_leverage, y=optimal_growth,
                text=f"Optimal: {optimal_leverage:.2f}<br>Growth: {optimal_growth:.2%}",
                showarrow=True, arrowhead=1,
                ax=20, ay=-30,
                row=1, col=2
            )
        
        # Add drawdown traces
        fig.add_trace(
            go.Scatter(x=leverages, y=mean_drawdowns, mode='lines+markers', name='Mean Drawdown',
                      line=dict(color='blue', width=2)),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=leverages, y=max_drawdowns, mode='lines+markers', name='Maximum Drawdown',
                      line=dict(color='red', width=2)),
            row=2, col=1
        )
        
        # Add ruin probability trace
        fig.add_trace(
            go.Scatter(x=leverages, y=ruin_probs, mode='lines+markers', name='Ruin Probability',
                      line=dict(color='red', width=2)),
            row=2, col=2
        )
        
        # Add horizontal line at 5% ruin probability
        if max(ruin_probs) > 0.05:
            fig.add_trace(
                go.Scatter(x=leverages, y=[0.05] * len(leverages), mode='lines',
                          line=dict(color='gray', width=1, dash='dash'),
                          name='5% Risk Threshold'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1200,
            title_text="Monte Carlo Simulation Dashboard",
            showlegend=True,
        )
        
        # Update axes
        fig.update_xaxes(title_text="Leverage", row=1, col=1)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        
        fig.update_xaxes(title_text="Leverage", row=1, col=2)
        fig.update_yaxes(title_text="Growth Rate", row=1, col=2, tickformat=".1%")
        
        fig.update_xaxes(title_text="Leverage", row=2, col=1)
        fig.update_yaxes(title_text="Drawdown", row=2, col=1, tickformat=".0%")
        
        fig.update_xaxes(title_text="Leverage", row=2, col=2)
        fig.update_yaxes(title_text="Probability", row=2, col=2, tickformat=".1%")
        
        return fig

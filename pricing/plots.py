import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import seaborn as sns
import numpy as np
import random
import string
import json
import plotly

import matplotlib
matplotlib.use('Agg')

def generate_random_filename(length=6):
    """Generate a random string for the file name"""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def plot_mc_paths_density(paths):
    """
    Plots Monte Carlo paths and the density of final stock prices with a rotated KDE plot.
    
    Parameters:
    paths (numpy.ndarray): A 2D array where each row represents a simulated stock price path.
    """
    # Limit the number of paths to 1000
    max_paths = 1000
    paths_to_plot = paths[:min(max_paths, len(paths))]

    # Generate a random filename
    plot_filename = f'mc_plot_path_{generate_random_filename()}.png'
    plot_path = os.path.join('static', 'images', plot_filename)
    
    # Create the plot with shared Y-axis
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)

    # Left subplot - Monte Carlo paths
    axes[0].plot(paths_to_plot.T, alpha=0.5, lw=1, color='steelblue')
    axes[0].axhline(paths_to_plot[:, -1].mean(), color='red', linestyle='--', lw=2, label=f'Reference Price: ${paths_to_plot[:, -1].mean():.2f}')
    axes[0].set_title('Monte Carlo Paths', fontsize=14)
    axes[0].set_xlabel('Time Steps', fontsize=12)
    axes[0].set_ylabel('Stock Price', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(fontsize=10)

    # Right subplot - Rotated KDE plot
    final_prices = paths_to_plot[:, -1]
    sns.kdeplot(y=final_prices, fill=True, color='blue', alpha=0.6, label='Final Price Density', ax=axes[1])
    axes[1].axhline(paths_to_plot[:, -1].mean(), color='red', linestyle='--', lw=2, label=f'Reference Price: ${paths_to_plot[:, -1].mean():.2f}')
    axes[1].set_title('Density of Final Stock Prices', fontsize=14)
    axes[1].set_ylabel('Stock Price', fontsize=12)  # Swapped from xlabel
    axes[1].set_xlabel('Density', fontsize=12)      # Swapped from ylabel
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot with the random filename
    plt.savefig(plot_path)
    plt.close()
    
    return plot_filename  # Return the random filename


def plot_vol_paths_density(paths):
    """
    Plots Volatility paths and the density of final stock prices with a rotated KDE plot.
    
    Parameters:
    paths (numpy.ndarray): A 2D array where each row represents a simulated stock price path.
    """
    # Limit the number of paths to 1000
    max_paths = 1000
    paths_to_plot = paths[:min(max_paths, len(paths))]

    # Generate a random filename
    plot_filename = f'mc_plot_path_{generate_random_filename()}.png'
    plot_path = os.path.join('static', 'images', plot_filename)
    
    # Create the plot with shared Y-axis
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)

    # Left subplot - Monte Carlo paths
    axes[0].plot(paths_to_plot.T, alpha=0.5, lw=1, color='steelblue')
    axes[0].axhline(paths_to_plot[:, -1].mean(), color='red', linestyle='--', lw=2, label=f'Reference Volatility: ${paths_to_plot[:, -1].mean():.2f}')
    axes[0].set_title('Volatility Paths', fontsize=14)
    axes[0].set_xlabel('Time Steps', fontsize=12)
    axes[0].set_ylabel('Volatility Value', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(fontsize=10)

    # Right subplot - Rotated KDE plot
    final_prices = paths_to_plot[:, -1]
    sns.kdeplot(y=final_prices, fill=True, color='blue', alpha=0.6, label='Final Volatility Density', ax=axes[1])
    axes[1].axhline(paths_to_plot[:, -1].mean(), color='red', linestyle='--', lw=2, label=f'Reference Volatility: ${paths_to_plot[:, -1].mean():.2f}')
    axes[1].set_title('Density of Final Volatilities', fontsize=14)
    axes[1].set_ylabel('Volatility', fontsize=12)  # Swapped from xlabel
    axes[1].set_xlabel('Density', fontsize=12)      # Swapped from ylabel
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot with the random filename
    plt.savefig(plot_path)
    plt.close()
    
    return plot_filename  # Return the random filename


def plot_MCMC_density(S_T_samples):
    """
    Plots the density of terminal stock prices using a KDE plot.
    
    Parameters:
    S_T_samples (numpy.ndarray): An array of terminal stock prices.
    """
    # Generate a random filename
    plot_filename = f'mcmc_plot_density_{generate_random_filename()}.png'
    plot_path = os.path.join('static', 'images', plot_filename)
    
    # Create the KDE plot
    sns.kdeplot(x=S_T_samples, fill=True, color='blue', alpha=0.6)
    plt.axvline(x=S_T_samples.mean(), color='red', linestyle='--', lw=2, label=f'Mean: {S_T_samples.mean():.2f}')
    plt.title('Density of Terminal Stock Prices', fontsize=14)
    plt.legend(loc='best')
    plt.xlabel('Terminal Stock Price', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save the plot with the random filename
    plt.savefig(plot_path)
    plt.close()
    
    return plot_filename

def option_price(option):
    """
    Generates a static Matplotlib plot for the option price vs. stock price, 
    as well as a plot for the option's payoff.

    Parameters:
    option: An option object with attributes (S, K, T, r, sigma, option_type).
    
    Returns:
    str: The filename of the saved plot image.
    """
    S = option.S
    S_range = np.linspace(0.1, S * 2, 1000)
    option_price = np.zeros_like(S_range)
    option_payoff = np.zeros_like(S_range)

    for i in range(len(S_range)):
        temp_option = option.__class__(S=S_range[i], K=option.K, T=option.T, 
                                       r=option.r, sigma=option.sigma, 
                                       option_type=option.option_type)
        option_price[i] = temp_option.black_scholes()
        
        # Payoff calculation based on the option type
        if option.option_type == 'call':
            option_payoff[i] = max(S_range[i] - option.K, 0)
        elif option.option_type == 'put':
            option_payoff[i] = max(option.K - S_range[i], 0)

    # Generate a random filename
    plot_filename = f'option_plot_{generate_random_filename()}.png'
    plot_path = os.path.join('static', 'plots', plot_filename)

    # Create the Matplotlib plot
    plt.figure(figsize=(8, 5))
    plt.plot(S_range, option_price, label=f'{option.option_type.capitalize()} Option Price', color='b')
    plt.axvline(x=option.K, color='black', linestyle='dashed', label='Strike Price')
    plt.xlabel('Stock Price')
    plt.ylabel('Option Price')
    plt.title('Option Price vs. Stock Price')
    plt.legend()
    plt.grid(True)
    
    # Plot the Payoff
    plt.plot(S_range, option_payoff, label=f'{option.option_type.capitalize()} Payoff', color='r', linestyle='--')
    plt.legend()

    # Save the plot as a static image
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    return plot_filename


def plot_price_surface(option):
    S = option.S
    T = option.T

    S_range = np.linspace(0.1, S * 2, 100)
    T_range = np.linspace(0.1, T, 100)
    price_matrix = np.zeros((len(S_range), len(T_range)))

    for s_idx, S_val in enumerate(S_range):
        for t_idx, T_val in enumerate(T_range):
            temp_option = option.__class__(
                S=S_val, K=option.K, T=T_val,
                r=option.r, sigma=option.sigma, option_type=option.option_type
            )
            price_matrix[s_idx, t_idx] = temp_option.black_scholes()

    # Create Plotly figure
    figure = go.Figure(data=[go.Surface(z=price_matrix, y=S_range, x=T_range)])
    figure.update_layout(
        title='Option Price Surface',
        scene=dict(
            yaxis_title='Stock Price',
            xaxis_title='Time to Expiry',
            zaxis_title='Option Price'
        ),
        template='plotly_white'
    )

    # Convert to JSON
    figure_json = json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)

    # Save JSON to a file in the "static/plots" folder
    filename = f'price_surface_{generate_random_filename()}.json'
    json_path = os.path.join('static', 'plotly', filename)
    
    with open(json_path, 'w') as f:
        f.write(figure_json)

    return filename  # Return the filename instead of storing JSON in session

def plot_delta_surface(delta_surface, S_range, K_range):
    
    figure = go.Figure(data=[go.Surface(z=delta_surface, y=S_range, x=K_range)])
    figure.update_layout(
        title='Delta Surface',
        scene=dict(
            yaxis_title='Stock Price',
            xaxis_title='Strike Price',
            zaxis_title='Delta'
        ),
        template='plotly_white'
    )

    # Convert to JSON
    figure_json = json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)

    # Save JSON to a file in the "static/plots" folder
    filename = f'delta_surface_{generate_random_filename()}.json'
    json_path = os.path.join('static', 'plotly', filename)
    
    with open(json_path, 'w') as f:
        f.write(figure_json)

    return filename

def generate_random_filename():
    return np.random.randint(100000, 999999)

def plot_convergence_speed(results, error, time_dict):
    # Create price convergence plot
    price_figure = go.Figure()

    # Access true price directly (it's a scalar value)
    true_price = results['true_price']

    # Create the range of x-axis (steps)
    steps = np.arange(1, len(results['mc_price']) + 1)

    # Plot Monte Carlo price
    price_figure.add_trace(go.Scatter(
        x=steps,
        y=results['mc_price'],
        mode='lines',
        name='Monte Carlo',
        line=dict(color='blue')
    ))

    # Plot Antithetic Monte Carlo price
    price_figure.add_trace(go.Scatter(
        x=steps,
        y=results['antithetic_mc_price'],
        mode='lines',
        name='Antithetic Monte Carlo',
        line=dict(color='red')
    ))
    
    # Plot Control Variate price
    price_figure.add_trace(go.Scatter(
        x=steps,
        y=results['cv_mc_price'],
        mode='lines',
        name='Control Variate',
        line=dict(color='orange')
    ))

    # Plot Binomial Tree price
    price_figure.add_trace(go.Scatter(
        x=steps,
        y=results['bt_price'],
        mode='lines',
        name='Binomial Tree',
        line=dict(color='green')
    ))

    # Add horizontal line for True Price
    price_figure.add_trace(go.Scatter(
        x=steps,
        y=[true_price] * len(steps),
        mode='lines',
        name='True Price',
        line=dict(color='black', dash='dash')
    ))

    price_figure.update_layout(
        title='Option Price Convergence',
        xaxis_title='Number of Steps',
        yaxis_title='Price',
        template='plotly_white'
    )

    # Create error convergence plot
    error_figure = go.Figure()

    # Plot Monte Carlo error
    error_figure.add_trace(go.Scatter(
        x=steps,
        y=error['mc_error'],
        mode='lines',
        name='Monte Carlo',
        line=dict(color='blue')
    ))

    # Plot Antithetic Monte Carlo error
    error_figure.add_trace(go.Scatter(
        x=steps,
        y=error['antithetic_mc_error'],
        mode='lines',
        name='Antithetic Monte Carlo',
        line=dict(color='red')
    ))
    
    # Plot Control Variate error
    error_figure.add_trace(go.Scatter(
        x=steps,
        y=error['cv_mc_error'],
        mode='lines',
        name='Control Variate',
        line=dict(color='orange')
    ))

    # Plot Binomial Tree error
    error_figure.add_trace(go.Scatter(
        x=steps,
        y=error['bt_error'],
        mode='lines',
        name='Binomial Tree',
        line=dict(color='green')
    ))

    error_figure.update_layout(
        title='Option Price Error Convergence',
        xaxis_title='Number of Steps',
        yaxis_title='Error',
        template='plotly_white'
    )

    # Create execution time plot
    time_figure = go.Figure()
    time_figure.add_trace(go.Bar(
        x=['Monte Carlo', 'Antithetic Monte Carlo', 'Control Variate Monte Carlo', 'Binomial Tree'],
        y=[time_dict['mc_time'], time_dict['antithetic_mc_time'], time_dict['cv_mc_time'], time_dict['bt_time']],
        marker=dict(color=['blue', 'red', 'orange', 'green']),
        name='Execution Time'
    ))
    
    time_figure.update_layout(
        title='Execution Time Comparison',
        xaxis_title='Method',
        yaxis_title='Time (seconds)',
        template='plotly_white'
    )

    # Convert figures to JSON
    price_figure_json = json.dumps(price_figure, cls=plotly.utils.PlotlyJSONEncoder)
    error_figure_json = json.dumps(error_figure, cls=plotly.utils.PlotlyJSONEncoder)
    time_figure_json = json.dumps(time_figure, cls=plotly.utils.PlotlyJSONEncoder)

    # Generate random filenames for saving
    price_filename = f'price_convergence_{generate_random_filename()}.json'
    error_filename = f'error_convergence_{generate_random_filename()}.json'
    time_filename = f'time_comparison_{generate_random_filename()}.json'

    # Define file paths
    price_json_path = os.path.join('static', 'plotly', price_filename)
    error_json_path = os.path.join('static', 'plotly', error_filename)
    time_json_path = os.path.join('static', 'plotly', time_filename)

    # Save JSON data to files
    with open(price_json_path, 'w') as f:
        f.write(price_figure_json)

    with open(error_json_path, 'w') as f:
        f.write(error_figure_json)

    with open(time_json_path, 'w') as f:
        f.write(time_figure_json)

    # Return the filenames of the saved figures
    return price_filename, error_filename, time_filename


def bs_plot_convergence_speed(results, error, time_dict):
    """
    Plot the convergence speed of Monte Carlo, Euler-Maruyama, and Milstein schemes using Plotly.
    Save the plots as JSON files and return their filenames.
    """
    price_figure = go.Figure()
    bs_price = results['bs_price'][0]
    steps = np.arange(1, len(results['mc_price']) + 1)

    # Plot Monte Carlo price
    price_figure.add_trace(go.Scatter(
        x=steps, y=results['mc_price'],
        mode='lines', name='Monte Carlo',
        line=dict(color='blue')
    ))

    # Plot Euler-Maruyama price
    price_figure.add_trace(go.Scatter(
        x=steps, y=results['euler_maruyama_price'],
        mode='lines', name='Euler-Maruyama',
        line=dict(color='red')
    ))
    
    # Plot Implicit Euler price
    price_figure.add_trace(go.Scatter(
        x=steps, y=results['euler_implicit_price'],
        mode='lines', name='Implicit Euler',
        line=dict(color='green')
    ))

    # Plot Milstein Scheme price
    price_figure.add_trace(go.Scatter(
        x=steps, y=results['milstein_scheme_price'],
        mode='lines', name='Milstein Scheme',
        line=dict(color='orange')
    ))

    # Add horizontal line for True Price
    price_figure.add_trace(go.Scatter(
        x=steps, y=[bs_price] * len(steps),
        mode='lines', name='True Price',
        line=dict(color='black', dash='dash')
    ))

    price_figure.update_layout(
        title='Option Price Convergence',
        xaxis_title='Number of Steps',
        yaxis_title='Price',
        template='plotly_white'
    )

    # Create error convergence plot
    error_figure = go.Figure()
    error_figure.add_trace(go.Scatter(
        x=steps, y=error['mc_error'],
        mode='lines', name='Monte Carlo',
        line=dict(color='blue')
    ))

    error_figure.add_trace(go.Scatter(
        x=steps, y=error['euler_maruyama_error'],
        mode='lines', name='Euler-Maruyama',
        line=dict(color='red')
    ))
    
    error_figure.add_trace(go.Scatter(
        x=steps, y=error['euler_implicit_error'],
        mode='lines', name='Implicit Euler',
        line=dict(color='green')
    ))

    error_figure.add_trace(go.Scatter(
        x=steps, y=error['milstein_scheme_error'],
        mode='lines', name='Milstein Scheme',
        line=dict(color='orange')
    ))

    error_figure.update_layout(
        title='Option Price Error Convergence',
        xaxis_title='Number of Steps',
        yaxis_title='Error',
        template='plotly_white'
    )

    time_figure = go.Figure()
    time_figure.add_trace(go.Bar(
        x=['Monte Carlo', 'Euler-Maruyama', 'Implicit Euler', 'Milstein Scheme'],
        y=[time_dict['mc_time'], time_dict['euler_maruyama_time'], time_dict['euler_implicit_time'],
           time_dict['milstein_scheme_time']],
        marker=dict(color=['blue', 'red', 'green', 'orange']),
        name='Execution Time'
    ))

    time_figure.update_layout(
        title='Execution Time Comparison',
        xaxis_title='Method',
        yaxis_title='Time (seconds)',
        template='plotly_white'
    )

    # Ensure directory exists
    os.makedirs('static/plotly', exist_ok=True)

    price_filename = f'bs_price_convergence_{generate_random_filename()}.json'
    error_filename = f'bs_error_convergence_{generate_random_filename()}.json'
    time_filename = f'bs_time_comparison_{generate_random_filename()}.json'

    price_json_path = os.path.join('static', 'plotly', price_filename)
    error_json_path = os.path.join('static', 'plotly', error_filename)
    time_json_path = os.path.join('static', 'plotly', time_filename)

    with open(price_json_path, 'w') as f:
        f.write(json.dumps(price_figure, cls=plotly.utils.PlotlyJSONEncoder))

    with open(error_json_path, 'w') as f:
        f.write(json.dumps(error_figure, cls=plotly.utils.PlotlyJSONEncoder))

    with open(time_json_path, 'w') as f:
        f.write(json.dumps(time_figure, cls=plotly.utils.PlotlyJSONEncoder))

    return price_filename, error_filename, time_filename


def heston_plot_convergence_speed(results, error, time_dict):
    """
    Plot the convergence speed of Euler-Maruyama and Milstein schemes for the Heston model.
    Save the plots as JSON files and return their filenames.

    Parameters:
        results (dict): Dictionary containing price results for Euler-Maruyama, Milstein, and Black-Scholes.
        error (dict): Dictionary containing error results for Euler-Maruyama and Milstein.
        time_dict (dict): Dictionary containing execution times for Euler-Maruyama and Milstein.

    Returns:
        tuple: Filenames of the saved JSON files (price_filename, error_filename, time_filename).
    """


    # Extract true price (Black-Scholes price)
    true_price = results['bs_price'][0]

    # Steps for x-axis
    steps = np.arange(1, len(results['euler_maruyama_price']) + 1)

    # Create price convergence plot
    price_figure = go.Figure()

    # Add Euler-Maruyama price
    price_figure.add_trace(go.Scatter(
        x=steps,
        y=results['euler_maruyama_price'],
        mode='lines',
        name='Euler-Maruyama',
        line=dict(color='blue')
    ))
    
    # Add Implicit Euler price
    price_figure.add_trace(go.Scatter(
        x=steps,
        y=results['euler_implicit_price'],
        mode='lines',
        name='Implicit Euler',
        line=dict(color='green')
    ))

    # Add Milstein Scheme price
    price_figure.add_trace(go.Scatter(
        x=steps,
        y=results['milstein_scheme_price'],
        mode='lines',
        name='Milstein Scheme',
        line=dict(color='red')
    ))

    # Add true price (Black-Scholes)
    price_figure.add_trace(go.Scatter(
        x=steps,
        y=[true_price] * len(steps),
        mode='lines',
        name='Black-Scholes Price',
        line=dict(color='black', dash='dash')
    ))

    # Update layout for price convergence plot
    price_figure.update_layout(
        title='Option Price Convergence (Heston Model)',
        xaxis_title='Number of Steps',
        yaxis_title='Price',
        template='plotly_white'
    )

    # Create error convergence plot
    error_figure = go.Figure()

    # Add Euler-Maruyama error
    error_figure.add_trace(go.Scatter(
        x=steps,
        y=error['euler_maruyama_error'],
        mode='lines',
        name='Euler-Maruyama',
        line=dict(color='blue')
    ))
    
    # Add Implicit Euler error
    error_figure.add_trace(go.Scatter(
        x=steps,
        y=error['euler_implicit_error'],
        mode='lines',
        name='Implicit Euler',
        line=dict(color='green')
    ))

    # Add Milstein Scheme error
    error_figure.add_trace(go.Scatter(
        x=steps,
        y=error['milstein_scheme_error'],
        mode='lines',
        name='Milstein Scheme',
        line=dict(color='red')
    ))

    # Update layout for error convergence plot
    error_figure.update_layout(
        title='Option Price Error Convergence (Heston Model)',
        xaxis_title='Number of Steps',
        yaxis_title='Error',
        template='plotly_white'
    )

    # Create execution time comparison plot
    time_figure = go.Figure()

    # Add execution times
    time_figure.add_trace(go.Bar(
        x=['Euler-Maruyama', 'Implicit Euler', 'Milstein Scheme'],
        y=[time_dict['euler_maruyama_time'], time_dict['euler_implicit_time'],
           time_dict['milstein_scheme_time']],
        marker=dict(color=['blue', 'green', 'red']),
        name='Execution Time'
    ))

    # Update layout for execution time plot
    time_figure.update_layout(
        title='Execution Time Comparison (Heston Model)',
        xaxis_title='Method',
        yaxis_title='Time (seconds)',
        template='plotly_white'
    )

    price_figure_json = json.dumps(price_figure, cls=plotly.utils.PlotlyJSONEncoder)
    error_figure_json = json.dumps(error_figure, cls=plotly.utils.PlotlyJSONEncoder)
    time_figure_json = json.dumps(time_figure, cls=plotly.utils.PlotlyJSONEncoder)

    price_filename = f'heston_price_convergence_{generate_random_filename()}.json'
    error_filename = f'heston_error_convergence_{generate_random_filename()}.json'
    time_filename = f'heston_time_comparison_{generate_random_filename()}.json'

    price_json_path = os.path.join('static', 'plotly', price_filename)
    error_json_path = os.path.join('static', 'plotly', error_filename)
    time_json_path = os.path.join('static', 'plotly', time_filename)

    with open(price_json_path, 'w') as f:
        f.write(price_figure_json)

    with open(error_json_path, 'w') as f:
        f.write(error_figure_json)

    with open(time_json_path, 'w') as f:
        f.write(time_figure_json)

    # Return the filenames of the saved figures
    return price_filename, error_filename, time_filename





    
    
# if __name__ == "__main__":
#     option = option_class.Option(239, 220, 1, 0.05, 0.24, 'call')
#     price, S_T_samples = option.MCMC(num_steps=10000)
#     figure = plot_MCMC_density(S_T_samples)
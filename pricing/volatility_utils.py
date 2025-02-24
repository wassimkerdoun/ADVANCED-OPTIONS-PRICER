import numpy as np
import plotly.graph_objects as go
import plotly
import json
import os
from scipy.optimize import brentq
from .heston_monte_carlo import heston_milstein_scheme
from .option_class import Option

def brentq_implied_volatility(option, market_price):
    
    def objective(sigma):
        option.sigma = sigma
        return option.black_scholes() - market_price
    
    try:
        return brentq(objective, 1e-8, 2)
    except ValueError:
        return np.nan
    
def volatility_smile(option, kappa, theta, vol_of_vol, rho, v0, num_paths, num_steps):
    
    S0 = option.S
    K = option.K
    T = option.T
    r = option.r
    sigma = option.sigma
    option_type = option.option_type
    
    K_range = np.linspace(0.50 * K, 1.50 * K, 100)
    ivs = np.full(len(K_range), np.nan)
    paths = heston_milstein_scheme(option, kappa, theta, vol_of_vol, rho, v0, num_paths, num_steps)[1]
    
    for i, strike in enumerate(K_range):
        if option_type == 'call':
            payoffs = np.maximum(paths[:, -1] - strike, 0)
        elif option_type == 'put':
            payoffs = np.maximum(strike - paths[:, -1], 0)
            
        market_price = np.exp(-r * T) * np.mean(payoffs)
    
        try:
            temp_option = Option(S0, strike, T, r, option.sigma, option_type)
            ivs[i] = brentq_implied_volatility(temp_option, market_price)
        except ValueError:
            ivs[i] = np.nan
        
    return ivs, K_range

def volatility_surface(option, kappa, theta, vol_of_vol, rho, v0, num_paths, num_steps):
    
    S0, K, T, r, sigma, option_type = option.S, option.K, option.T, option.r, option.sigma, option.option_type
    
    K_range = np.linspace(0.50 * K, 1.50 * K, 50)
    T_range = np.linspace(0.25 * T, 1.50 * T, 50)
    iv_surface = np.full((len(K_range), len(T_range)), np.nan)
    
    # Simulate the Heston model for the longest maturity
    max_maturity = np.max(T_range)
    paths = heston_milstein_scheme(option, kappa, theta, vol_of_vol, rho, v0, num_paths, num_steps)[1]
    
    for i, strike in enumerate(K_range):
        for j, maturity in enumerate(T_range):
            # Determine the number of steps for this maturity
            steps = max(1, int((maturity / max_maturity) * num_steps))
            
            # Get terminal asset price
            terminal_prices = paths[:, steps]
            
            # Compute payoffs
            payoffs = np.maximum(terminal_prices - strike, 0) if option_type == 'call' else np.maximum(strike - terminal_prices, 0)
            
            # Compute discounted market price
            market_price = np.exp(-r * maturity) * np.mean(payoffs)
            
            # Compute implied volatility
            try:
                temp_option = Option(S0, strike, T, r, sigma, option_type)
                iv_surface[i, j] = brentq_implied_volatility(temp_option, market_price)
            except ValueError:
                iv_surface[i, j] = np.nan  # Set NaN if implied vol fails to converge

    return iv_surface, K_range, T_range


def generate_random_filename():
    return np.random.randint(100000, 999999)

def plot_volatility_smile(implied_volatility, K_range):
    """
    Plot the volatility smile and save the figure as a JSON file.

    Parameters:
        implied_volatility (list or np.array): Implied volatilities.
        K_range (list or np.array): Strike prices.

    Returns:
        str: Filename of the saved JSON file.
    """
    # Generate random filename
    filename = f'volatility_smile_{generate_random_filename()}.json'

    K = K_range[-1] / 1.5
    
    # Create figure
    fig = go.Figure()

    # Add implied volatility plot
    fig.add_trace(go.Scatter(
        x=K_range,
        y=implied_volatility,
        mode='lines',
        name='Implied Volatility',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=[K, K],
        y=[np.min(implied_volatility), np.max(implied_volatility)],
        mode='lines',
        name='Strike Price',
        line=dict(color='red', dash='dash')
    ))

    # Update layout
    fig.update_layout(
        title='Volatility Smile',
        xaxis_title='Strike Price',
        yaxis_title='Implied Volatility',
        template='plotly_white'
    )

    # Convert figure to JSON and save
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    file_path = os.path.join('static', 'plotly', filename)

    with open(file_path, 'w') as f:
        f.write(fig_json)

    # Return the filename of the saved figure
    return filename

def plot_volatility_surface(ivsurface, K_range, T_range):
    """
    Plot the volatility surface and save the figure as a JSON file.

    Parameters:
        ivsurface (2D array): Implied volatility surface.
        K_range (list or np.array): Strike prices.
        T_range (list or np.array): Maturities.

    Returns:
        str: Filename of the saved JSON file.
    """
    # Generate random filename
    filename = f'volatility_surface_{generate_random_filename()}.json'
    
    K = K_range[-1] / 1.5
    T = T_range[-1] / 1.5

    # Create figure
    fig = go.Figure()

    # Add surface plot
    fig.add_trace(go.Surface(
        z=ivsurface,
        x=K_range,
        y=T_range,
        colorscale='Viridis'
    ))

    # Update layout
    fig.update_layout(
        title='Volatility Surface',
        scene=dict(
            xaxis_title='Strike Price',
            yaxis_title='Maturity',
            zaxis_title='Implied Volatility'
        ),
        template='plotly_white'
    )

    # Convert figure to JSON and save
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    file_path = os.path.join('static', 'plotly', filename)

    with open(file_path, 'w') as f:
        f.write(fig_json)

    # Return the filename of the saved figure
    return filename
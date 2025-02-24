from .option_class import Option
import numpy as np
from multiprocessing import Pool
import time


def gbm_discrete(option, num_paths, num_steps):
    
    """
    Calculate the price of European options using the Geometric Brownian Motion model.

    Parameters:
    option : Option
        The option for which to calculate the price.
    num_paths : int
        Number of paths to simulate.
    num_steps : int
        Number of time steps for each path.

    Returns:
    tuple
        A tuple containing the option price and a 2D numpy array of generated paths.
    """
    S0 = option.S
    K = option.K
    T = option.T
    r = option.r
    sigma = option.sigma
    option_type = option.option_type
    
    dt = T / num_steps
    
    Z = np.random.normal(size=(num_paths, num_steps))
    
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    cumulative_log_returns = np.cumsum(log_returns, axis=1)
    
    paths = S0 * np.exp(cumulative_log_returns)
    paths = np.insert(paths, 0, S0, axis=1)
    
    if option_type == 'call':
        payoffs = np.maximum(paths[:, -1] - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - paths[:, -1], 0)
        
    price = np.exp(-r * T) * np.mean(payoffs)
    
    return price, paths

def bs_euler_maruyama(option, num_paths, num_steps):
    
    """
    Calculate the price of European options using the Euler-Maruyama method.

    Parameters:
    option : Option
        The option for which to calculate the price.
    num_paths : int
        Number of paths to simulate.
    num_steps : int
        Number of time steps for each path.

    Returns:
    tuple
        A tuple containing the option price and a 2D numpy array of generated paths.
    """
    
    S0 = option.S
    K = option.K
    T = option.T
    r = option.r
    sigma = option.sigma
    option_type = option.option_type
    
    dt = T / num_steps
    
    Z = np.random.normal(size=(num_paths, num_steps))
    
    paths = np.zeros((num_paths, num_steps + 1))
    paths[:, 0] = S0
    
    for i in range(1, num_steps + 1):
        paths[:, i] = paths[:, i - 1] * (1 + r * dt + sigma * np.sqrt(dt) * Z[:, i - 1])
        
    if option_type == 'call':
        payoffs = np.maximum(paths[:, -1] - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - paths[:, -1], 0)
        
    price = np.exp(-r * T) * np.mean(payoffs)
    
    return price, paths

def bs_milstein_scheme(option, num_paths, num_steps):
    
    """
    Calculate the price of European options using the Milstein scheme.

    Parameters:
    option : Option
        The option for which to calculate the price.
    num_paths : int
        Number of paths to simulate.
    num_steps : int
        Number of time steps for each path.

    Returns:
    tuple
        A tuple containing the option price and a 2D numpy array of generated paths.
    """
    S0 = option.S
    K = option.K
    T = option.T
    r = option.r
    sigma = option.sigma
    option_type = option.option_type
    
    dt = T / num_steps
    
    Z = np.random.normal(size=(num_paths, num_steps))
    
    paths = np.zeros((num_paths, num_steps + 1))
    paths[:, 0] = S0
    
    for i in range(1, num_steps + 1):
        paths[:, i] = paths[:, i - 1] * (1 + r * dt + sigma * np.sqrt(dt) * Z[:, i - 1] + 0.5 * sigma**2 * dt * (Z[:, i - 1]**2 - 1))
        
    if option_type == 'call':
        payoffs = np.maximum(paths[:, - 1] - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - paths[:, -1], 0)
        
    price = np.exp(-r * T) * np.mean(payoffs)
    
    return price, paths


def calculate_monte_carlo_price(S, K, T, r, sigma, num_paths, num_steps, option_type):
    option = Option(S, K, T, r, sigma, option_type)
    true_price = option.black_scholes()
    mc_price, _ = gbm_discrete(option, num_paths, num_steps)
    return mc_price, abs(mc_price - true_price)

def calculate_euler_maruyama_price(S, K, T, r, sigma, num_paths, num_steps, option_type):
    option = Option(S, K, T, r, sigma, option_type)
    true_price = option.black_scholes()
    mc_price, _ = bs_euler_maruyama(option, num_paths, num_steps)
    return mc_price, abs(mc_price - true_price)

def calculate_milstein_scheme_price(S, K, T, r, sigma, num_paths, num_steps, option_type):
    option = Option(S, K, T, r, sigma, option_type)
    true_price = option.black_scholes()
    mc_price, _ = bs_milstein_scheme(option, num_paths, num_steps)
    return mc_price, abs(mc_price - true_price)

def bs_convergence_speed(option, num_paths, num_steps):
    """
    Calculate the prices of European options using the Black-Scholes, Monte Carlo, Euler Maruyama, and Milstein Scheme methods
    and their respective errors compared to the Black-Scholes price.
    
    Parameters:
    option : Option
        The option for which to calculate the prices.
    num_paths : int
        Number of paths for each Monte Carlo simulation.
    num_steps : int
        Number of time steps for each simulation.

    Returns:
    tuple
        A tuple containing three dictionaries: results, error, and time_dict. 
        - results: Contains the prices of the option calculated using the Black-Scholes, Monte Carlo, Euler Maruyama, and Milstein Scheme methods.
        - error: Contains the errors of the respective methods compared to the Black-Scholes price.
        - time_dict: Contains the computation time for each method.
    """
    
    results = {
        'bs_price': [],
        'mc_price': [],
        'euler_maruyama_price': [],
        'milstein_scheme_price': [],
    }
    
    error = {
        'mc_error': [],
        'euler_maruyama_error': [],
        'milstein_scheme_error': []
    }
    
    time_dict = {
        'mc_time': [],
        'euler_maruyama_time': [],
        'milstein_scheme_time': [],
    }
    
    true_price = option.black_scholes()
    results['bs_price'] = [true_price] * num_steps
    
    steps = np.arange(1, num_steps + 1)
    
    
    S, K, T, r, sigma, option_type = option.S, option.K, option.T, option.r, option.sigma, option.option_type
    mc_args = [(S, K, T, r, sigma, num_paths, step, option_type) for step in steps]
    euler_maruyama_args = [(S, K, T, r, sigma, num_paths, step, option_type) for step in steps]
    milstein_scheme_args = [(S, K, T, r, sigma, num_paths, step, option_type) for step in steps]
    
    with Pool() as pool:
        
        # Warm-up phase for both calculations (dummy tasks to initialize workers)
        dummy_args = [(S, K, T, r, sigma, num_paths, 1, option_type)]
        pool.starmap(calculate_monte_carlo_price, dummy_args)  # Warm-up Monte Carlo
        pool.starmap(calculate_euler_maruyama_price, dummy_args)  # Warm-up Euler Maruyama
        pool.starmap(calculate_milstein_scheme_price, dummy_args)  # Warm-up Milstein Scheme

        # Monte Carlo calculations
        
        start_time = time.time()
        mc_results = pool.starmap(calculate_monte_carlo_price, mc_args)
        time_dict['mc_time'] = time.time() - start_time
        results['mc_price'] = [res[0] for res in mc_results]
        error['mc_error'] = [res[1] for res in mc_results]

        # Euler Maruyama calculations
        
        start_time = time.time()
        euler_maruyama_results = pool.starmap(calculate_euler_maruyama_price, euler_maruyama_args)
        time_dict['euler_maruyama_time'] = time.time() - start_time 
        results['euler_maruyama_price'] = [res[0] for res in euler_maruyama_results]  
        error['euler_maruyama_error'] = [res[1] for res in euler_maruyama_results] 
        
        # Milstein Scheme calculations
        
        start_time = time.time()
        milstein_scheme_results = pool.starmap(calculate_milstein_scheme_price, milstein_scheme_args)
        time_dict['milstein_scheme_time'] = time.time() - start_time  
        results['milstein_scheme_price'] = [res[0] for res in milstein_scheme_results]  
        error['milstein_scheme_error'] = [res[1] for res in milstein_scheme_results]  

    return results, error, time_dict







# if __name__ == '__main__':
    
#     option = Option(100, 100, 1, 0.05, 0.2, 'call')
    
#     num_paths = 1000
#     num_steps = 70
    
#     results, error, time_dict = bs_convergence_speed(option, num_paths, num_steps)
    
#     print(error)
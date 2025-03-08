from .option_class import Option
import numpy as np
from multiprocessing import Pool
import time

def volatility_euler_maruyama(option, kappa, theta, vol_of_vol, v0, num_paths, num_steps):
    
    T = option.T
    dt = T / num_steps
    Z1 = np.random.normal(size=(num_paths, num_steps))
    
    v = np.zeros((num_paths, num_steps + 1))
    v[:, 0] = v0
    
    for i in range(1, num_steps + 1):
        v_prev = np.maximum(v[:, i - 1], 0)
        v[:, i] = v_prev + kappa * (theta - v_prev) * dt + vol_of_vol * np.sqrt(v_prev * dt) * Z1[:, i - 1]
        v[:, i] = np.maximum(v[:, i], 1e-8)
        
    return v

def volatility_euler_implicit(option, kappa, theta, vol_of_vol, v0, num_paths, num_steps):
    
    T = option.T
    dt = T / num_steps
    Z1 = np.random.normal(size=(num_paths, num_steps))
    
    v = np.zeros((num_paths, num_steps + 1))
    v[:, 0] = v0
    
    for i in range(1, num_steps + 1):
        v_prev = np.maximum(v[:, i - 1], 0)
        v[:, i] = (v_prev + kappa * (theta - v_prev) * dt + vol_of_vol * np.sqrt(v_prev * dt) * Z1[:, i - 1]) / (1 + kappa * dt)
        v[:, i] = np.maximum(v[:, i], 1e-8)
        
    return v

def volatility_milstein_scheme(option, kappa, theta, vol_of_vol, v0, num_paths, num_steps):
    
    T = option.T
    dt = T / num_steps
    Z1 = np.random.normal(size=(num_paths, num_steps))
    
    v = np.zeros((num_paths, num_steps + 1))
    v[:, 0] = v0
    
    for i in range(1, num_steps + 1):
        v_prev = np.maximum(v[:, i - 1], 0)
        v[:, i] = v_prev + kappa * (theta - v_prev) * dt + vol_of_vol * np.sqrt(v_prev * dt) * Z1[:, i - 1] + 0.25 * vol_of_vol**2 * dt * (Z1[:, i - 1]**2 - 1)
        v[:, i] = np.maximum(v[:, i], 1e-8)
        
    return v

def heston_euler_maruyama(option, kappa, theta, vol_of_vol, rho, v0, num_paths, num_steps):
    
    S0 = option.S
    K = option.K
    T = option.T
    r = option.r
    option_type = option.option_type
    
    dt = T / num_steps
    
    Z1 = np.random.normal(size=(num_paths, num_steps))
    Z2 = rho * Z1 + np.sqrt(1 - rho ** 2) * np.random.normal(size=(num_paths, num_steps))
    
    v = volatility_euler_maruyama(option, kappa, theta, vol_of_vol, v0, num_paths, num_steps)
    
    paths = np.zeros((num_paths, num_steps + 1))
    paths[:, 0] = S0
    
    for i in range(1, num_steps + 1):
        paths[:, i] = paths[:, i - 1] * (1 + r * dt + np.sqrt(v[:, i - 1] * dt) * Z2[:, i - 1])
        
    if option_type == 'call':
        payoffs = np.maximum(paths[:, -1] - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - paths[:, -1], 0)
        
    price = np.exp(-r * T) * np.mean(payoffs)
    
    return price, paths

def heston_euler_implicit(option, kappa, theta, vol_of_vol, rho, v0, num_paths, num_steps):
    
    S0 = option.S
    K = option.K
    T = option.T
    r = option.r
    option_type = option.option_type
    
    dt = T / num_steps
    
    Z1 = np.random.normal(size=(num_paths, num_steps))
    Z2 = rho * Z1 + np.sqrt(1 - rho ** 2) * np.random.normal(size=(num_paths, num_steps))
    
    v = volatility_euler_implicit(option, kappa, theta, vol_of_vol, v0, num_paths, num_steps)
    
    paths = np.zeros((num_paths, num_steps + 1))
    paths[:, 0] = S0
    
    for i in range(1, num_steps + 1):
        paths[:, i] = (paths[:, i - 1] + paths[:, i - 1] * np.sqrt(v[:, i - 1] * dt) * Z2[:, i - 1]) / (1 - r * dt)
        
    if option_type == 'call':
        payoffs = np.maximum(paths[:, -1] - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - paths[:, -1], 0)
        
    price = np.exp(-r * T) * np.mean(payoffs)
    
    return price, paths

def heston_milstein_scheme(option, kappa, theta, vol_of_vol, rho, v0, num_paths, num_steps):
    """
    Simulate the Heston model using the Milstein scheme for the stock price
    and the provided `volatility_milstein_scheme` for the volatility process.
    """
    S0 = option.S
    K = option.K
    T = option.T
    r = option.r
    option_type = option.option_type
    
    dt = T / num_steps 
    
    Z1 = np.random.normal(size=(num_paths, num_steps))
    Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal(size=(num_paths, num_steps))
    
    v = volatility_milstein_scheme(option, kappa, theta, vol_of_vol, v0, num_paths, num_steps)
    
    paths = np.zeros((num_paths, num_steps + 1))
    paths[:, 0] = S0
    
    for i in range(1, num_steps + 1):
        paths[:, i] = paths[:, i - 1] * (1 + r * dt+ np.sqrt(v[:, i - 1] * dt) * Z2[:, i - 1] + 0.25 * v[:, i - 1] * dt * (Z2[:, i - 1]**2 - 1))
    
    if option_type == 'call':
        payoffs = np.maximum(paths[:, -1] - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - paths[:, -1], 0)
    
    price = np.exp(-r * T) * np.mean(payoffs)
    
    return price, paths


def calculate_euler_maruyama_price(S, K, T, r, sigma, kappa, theta, vol_of_vol, rho, v0, num_paths, num_steps, option_type):
    option = Option(S, K, T, r, sigma, option_type)
    true_price = option.black_scholes()
    mc_price, _ = heston_euler_maruyama(option, kappa, theta, vol_of_vol, rho, v0, num_paths, num_steps)
    return mc_price, abs(mc_price - true_price)

def calculate_euler_implicit_price(S, K, T, r, sigma, kappa, theta, vol_of_vol, rho, v0, num_paths, num_steps, option_type):
    option = Option(S, K, T, r, sigma, option_type)
    true_price = option.black_scholes()
    mc_price, _ = heston_euler_implicit(option, kappa, theta, vol_of_vol, rho, v0, num_paths, num_steps)
    return mc_price, abs(mc_price - true_price)

def calculate_milstein_scheme_price(S, K, T, r, sigma, kappa, theta, vol_of_vol, rho, v0, num_paths, num_steps, option_type):
    option = Option(S, K, T, r, sigma, option_type)
    true_price = option.black_scholes()
    mc_price, _ = heston_milstein_scheme(option, kappa, theta, vol_of_vol, rho, v0, num_paths, num_steps)
    return mc_price, abs(mc_price - true_price)

def heston_convergence_speed(option, kappa, theta, vol_of_vol, rho, v0, num_paths, num_steps):
    results = {
        'bs_price': [],
        'euler_maruyama_price': [],
        'euler_implicit_price': [],
        'milstein_scheme_price': [],
    }
    
    error = {
        'euler_maruyama_error': [],
        'euler_implicit_error': [],
        'milstein_scheme_error': []
    }
    
    time_dict = {
        'euler_maruyama_time': [],
        'euler_implicit_time': [],
        'milstein_scheme_time': [],
    }
    
    true_price = option.black_scholes()
    results['bs_price'] = [true_price] * num_steps
    
    steps = np.arange(1, num_steps + 1)
    
    # Define arguments for both functions
    S, K, T, r, sigma, option_type = option.S, option.K, option.T, option.r, option.sigma, option.option_type
    euler_maruyama_args = [(S, K, T, r, sigma, kappa, theta, vol_of_vol, rho, v0, num_paths, step, option_type) for step in steps]
    euler_implicit_args = [(S, K, T, r, sigma, kappa, theta, vol_of_vol, rho, v0, num_paths, step, option_type) for step in steps]
    milstein_scheme_args = [(S, K, T, r, sigma, kappa, theta, vol_of_vol, rho, v0, num_paths, step, option_type) for step in steps]
    
    # Use a single pool for both calculations
    with Pool() as pool:
        # Warm-up phase for both calculations (dummy tasks to initialize workers)
        dummy_args = [(S, K, T, r, sigma, kappa, theta, vol_of_vol, rho, v0, num_paths, 1, option_type)]
        pool.starmap(calculate_euler_maruyama_price, dummy_args)  # Warm-up Euler Maruyama
        pool.starmap(calculate_euler_implicit_price, dummy_args)  # Warm-up Euler Implicit
        pool.starmap(calculate_milstein_scheme_price, dummy_args)  # Warm-up Milstein Scheme
        
        
        # Main calculation for Milstein Scheme
        start_time = time.time()
        milstein_scheme_results = pool.starmap(calculate_milstein_scheme_price, milstein_scheme_args)
        time_dict['milstein_scheme_time'] = time.time() - start_time
        results['milstein_scheme_price'] = [res[0] for res in milstein_scheme_results]
        error['milstein_scheme_error'] = [res[1] for res in milstein_scheme_results]
        
        
        # Main calculation for Euler Implicit
        start_time = time.time()
        euler_implicit_results = pool.starmap(calculate_euler_implicit_price, euler_implicit_args)
        time_dict['euler_implicit_time'] = time.time() - start_time
        results['euler_implicit_price'] = [res[0] for res in euler_implicit_results]
        error['euler_implicit_error'] = [res[1] for res in euler_implicit_results]
        
        # Main calculation for Euler Maruyama
        start_time = time.time()
        euler_maruyama_results = pool.starmap(calculate_euler_maruyama_price, euler_maruyama_args)
        time_dict['euler_maruyama_time'] = time.time() - start_time
        results['euler_maruyama_price'] = [res[0] for res in euler_maruyama_results]
        error['euler_maruyama_error'] = [res[1] for res in euler_maruyama_results]

    return results, error, time_dict




# if __name__ == '__main__':
    
#     S0 = 228
#     K = 230
#     T = 2
#     r = 0.05
#     sigma = 0.23
#     kappa = 2
#     theta = 0.05
#     vol_of_vol = 0.25
#     rho = -0.75
#     v0 = 0.1
#     num_paths = 10000
#     num_steps = 100
#     option_type = 'put'
    
#     option = Option(S0, K, T, r, sigma, option_type)
    
#     results, error, time_dict = heston_convergence_speed(option, kappa, theta, vol_of_vol, rho, v0, num_paths, num_steps)
#     print(time_dict)
from .option_class import Option
import time
import numpy as np
from multiprocessing import Pool


def calculate_monte_carlo_price(S, K, T, r, sigma, num_paths, num_steps, option_type):
    option = Option(S, K, T, r, sigma, option_type)
    true_price = option.black_scholes()
    mc_price, _ = option.monte_carlo(num_paths, num_steps)
    return mc_price, abs(mc_price - true_price)

def calculate_antithetic_monte_carlo_price(S, K, T, r, sigma, num_paths, num_steps, option_type):
    option = Option(S, K, T, r, sigma, option_type)
    true_price = option.black_scholes()
    mc_price, _ = option.antithetic_monte_carlo(num_paths, num_steps)
    return mc_price, abs(mc_price - true_price)

def calculate_control_variate_monte_carlo_price(S, K, T, r, sigma, num_paths, num_steps, option_type):
    option = Option(S, K, T, r, sigma, option_type)
    true_price = option.black_scholes()
    mc_price, _ = option.control_variates_monte_carlo(num_paths, num_steps)
    return mc_price, abs(mc_price - true_price)

def calculate_binomial_tree_price(S, K, T, r, sigma, num_paths, option_type):
    option = Option(S, K, T, r, sigma, option_type)
    true_price = option.black_scholes()
    bt_price = option.binomial_tree(num_paths)
    return bt_price, abs(bt_price - true_price)

def convergence_speed(option, num_paths, num_steps):
    results = {
        'true_price': [],
        'mc_price': [],
        'antithetic_mc_price': [],
        'cv_mc_price': [],
        'bt_price': []
    }

    error = {
        'mc_error': [],
        'antithetic_mc_error': [],
        'cv_mc_error': [],
        'bt_error': []
    }

    time_dict = {
        'mc_time': [],
        'antithetic_mc_time': [],
        'cv_mc_time': [],
        'bt_time': []
    }

    # True price from Black-Scholes
    true_price = option.black_scholes()
    results['bs_price'] = [true_price] * num_steps

    steps = np.arange(1, num_steps + 1)

    # Define arguments for both functions
    S, K, T, r, sigma, option_type = option.S, option.K, option.T, option.r, option.sigma, option.option_type
    mc_args = [(S, K, T, r, sigma, num_paths, step, option_type) for step in steps]
    antithetic_args = [(S, K, T, r, sigma, num_paths, step, option_type) for step in steps]
    cv_args = [(S, K, T, r, sigma, num_paths, step, option_type) for step in steps]
    bt_args = [(S, K, T, r, sigma, step, option_type) for step in steps]

    # Use a single pool for all calculations
    with Pool() as pool:
        # Warm-up phase for all calculations (small task to initialize workers)
        warm_up_args = [(S, K, T, r, sigma, num_paths, 1, option_type)]  # Use 1 step for warm-up
        pool.starmap(calculate_monte_carlo_price, warm_up_args)  # Warm-up Monte Carlo
        pool.starmap(calculate_antithetic_monte_carlo_price, warm_up_args)  # Warm-up Antithetic Monte Carlo
        pool.starmap(calculate_control_variate_monte_carlo_price, warm_up_args)  # Warm-up Control Variate Monte Carlo

        # Main calculation for Monte Carlo
        start_time = time.time()
        mc_results = pool.starmap(calculate_monte_carlo_price, mc_args)
        time_dict['mc_time'] = time.time() - start_time
        results['mc_price'] = [res[0] for res in mc_results]
        error['mc_error'] = [res[1] for res in mc_results]

        # Main calculation for Antithetic Monte Carlo
        start_time = time.time()
        antithetic_results = pool.starmap(calculate_antithetic_monte_carlo_price, antithetic_args)
        time_dict['antithetic_mc_time'] = time.time() - start_time
        results['antithetic_mc_price'] = [res[0] for res in antithetic_results]
        error['antithetic_mc_error'] = [res[1] for res in antithetic_results]

        # Main calculation for Control Variate Monte Carlo
        start_time = time.time()
        cv_results = pool.starmap(calculate_control_variate_monte_carlo_price, cv_args)
        time_dict['cv_mc_time'] = time.time() - start_time
        results['cv_mc_price'] = [res[0] for res in cv_results]
        error['cv_mc_error'] = [res[1] for res in cv_results]

        # Main calculation for Binomial Tree
        start_time = time.time()
        bt_results = pool.starmap(calculate_binomial_tree_price, bt_args)
        time_dict['bt_time'] = time.time() - start_time
        results['bt_price'] = [res[0] for res in bt_results]
        error['bt_error'] = [res[1] for res in bt_results]

    return results, error, time_dict

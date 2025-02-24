import numpy as np
from math import exp, sqrt, log
from scipy.stats import norm
from scipy.optimize import brentq


class Option:
  def __init__(self, S, K, T, r, sigma, option_type='call'):
    self.S = S
    self.K = K
    self.T = T
    self.r = r
    self.sigma = sigma
    self.option_type = option_type
    
  def d1_d2(self):
    
    """
    Calculate d1 and d2 used in the Black-Scholes model.
    
    Parameters:
    None
    
    Returns:
    tuple
        A tuple containing the values of d1 and d2.
    """
    d1 = (log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * sqrt(self.T))
    d2 = d1 - self.sigma * sqrt(self.T)
    
    return d1, d2
  
  def black_scholes(self):
    
    """
    Calculate the price of European options using the Black-Scholes model.

    Parameters:
    None

    Returns:
    float
        The price of the option.
    """
    d1, d2 = self.d1_d2()
    
    if self.option_type == 'call':
      price = self.S * norm.cdf(d1) - self.K * exp(-self.r * self.T) * norm.cdf(d2)
    else:
      price = self.K * exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
      
    return price
  
  def monte_carlo(self, num_paths, num_steps):
      """
      Calculate the price of a European option using Monte Carlo simulation.

      Parameters:
      num_paths : int
          Number of Monte Carlo paths to simulate.
      num_steps : int
          Number of time steps for each path.

      Returns:
      tuple
          A tuple containing the option price and a 2D numpy array of generated paths.
      """
      dt = self.T / num_steps
      paths = np.zeros((num_paths, num_steps + 1))
      paths[:, 0] = self.S

      Z = np.random.normal(size=(num_paths, num_steps))

      drift = (self.r - 0.5 * self.sigma**2) * dt
      diffusion = self.sigma * np.sqrt(dt)
      for j in range(1, num_steps + 1):
          paths[:, j] = paths[:, j - 1] * np.exp(drift + diffusion * Z[:, j - 1])

      if self.option_type == 'call':
          payoff = np.maximum(paths[:, -1] - self.K, 0)
      else:
          payoff = np.maximum(self.K - paths[:, -1], 0)

      price = np.exp(-self.r * self.T) * np.mean(payoff)

      return price, paths
  
  def antithetic_monte_carlo(self, num_paths, num_steps):

    """
    Calculate the price of European options using the antithetic Monte Carlo method.

    Parameters:
    num_paths : int
        Number of Monte Carlo paths to simulate.
    num_steps : int
        Number of time steps for each path.

    Returns:
    tuple
        A tuple containing the option price and a 2D numpy array of generated paths.
    """
    
    dt = self.T / num_steps
    paths = np.zeros((num_paths, num_steps+1))
    antithetic_paths = np.zeros((num_paths, num_steps+1))
    paths[:, 0] = self.S
    antithetic_paths[:, 0] = self.S
    
    for i in range(num_paths):
      for j in range(1, num_steps+1):
        Z = np.random.normal()
        paths[i, j] = paths[i, j-1] * exp((self.r - 0.5 * self.sigma**2) * dt + self.sigma * sqrt(dt) * Z)
        antithetic_paths[i, j] = antithetic_paths[i, j-1] * exp((self.r - 0.5 * self.sigma**2) * dt + self.sigma * sqrt(dt) * -Z)
        
    if self.option_type == 'call':
      payoff = np.maximum(paths[:, -1] - self.K, 0)
      anthetic_payoff = np.maximum(antithetic_paths[:, -1] - self.K, 0)
    else:
      payoff = np.maximum(self.K - paths[:, -1], 0)
      anthetic_payoff = np.maximum(self.K - antithetic_paths[:, -1], 0)
    
    payoff = 0.5 * (payoff + anthetic_payoff)
    price = np.exp(-self.r * self.T) * np.mean(payoff)
    
    return price, paths
  
  def control_variates_monte_carlo(self, num_paths, num_steps):
    
    """
    Calculate the price of European options using the Monte Carlo method with control variates.

    Parameters:
    num_paths : int
        Number of Monte Carlo paths to simulate.
    num_steps : int
        Number of time steps for each path.

    Returns:
    tuple
        A tuple containing the option price and a 2D numpy array of generated paths.
    """

    dt = self.T / num_steps
    paths = np.zeros((num_paths, num_steps+1))
    paths[:, 0] = self.S
    
    for i in range(num_paths):
      for j in range(1, num_steps+1):
        Z = np.random.normal()
        paths[i, j] = paths[i, j-1] * exp((self.r - 0.5 * self.sigma**2) * dt + self.sigma * sqrt(dt) * Z)
        
    ST = paths[:, -1]

    if self.option_type == "call":
      discounted_payoff = np.exp(-self.r * self.T) * np.maximum(ST - self.K, 0)
    else:
      discounted_payoff = np.exp(-self.r * self.T) * np.maximum(self.K - ST, 0)

    ST_BS = self.S * np.exp(self.r * self.T) 

    lambda_star = np.cov(discounted_payoff, ST)[0, 1] / np.var(ST)

    MC_control_variate_estimate = np.mean(discounted_payoff - lambda_star * (ST - ST_BS))

    return MC_control_variate_estimate, paths
    
  
  def MCMC(self, num_steps):
    def target_distribution(S_T):
      mean = log(self.S) + (self.r - 0.5 * self.sigma**2) * self.T
      std_dev = self.sigma * sqrt(self.T)
      return norm.pdf(log(S_T), mean, std_dev)
    
    S_T_samples = []
    S_T_current = self.S * exp((self.r - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(self.T) * np.random.normal()) #Initialize S_T
    
    proposal_std = 10.0
    
    for _ in range(num_steps):
      S_T_new = S_T_current + proposal_std * np.random.normal()
      
      if S_T_new <=0:
        S_T_samples.append(S_T_current)
        continue
      
      acceptance_ratio = min(1, target_distribution(S_T_new) / target_distribution(S_T_current))
      
      if np.random.rand() < acceptance_ratio:
        S_T_current = S_T_new # Accept the move
        
      S_T_samples.append(S_T_current)
      
    S_T_samples = np.array(S_T_samples)
      
    if self.option_type.lower() == 'call':
      payoffs = np.maximum(S_T_samples - self.K, 0)
    else:
      payoffs = np.maximum(self.K - S_T_samples, 0)
      
    price = exp(-self.r * self.T) * np.mean(payoffs)
    
    return price, S_T_samples
  
  def binomial_tree(self, num_steps):
    
    """
    Calculate the price of a European option using the binomial tree model.

    Parameters:
    num_steps : int
        Number of time steps in the tree.

    Returns:
    float
        The price of the option.
    """
    dt = self.T / num_steps
    u = exp(self.sigma * sqrt(dt))
    d = 1 / u
    p = (exp(self.r * dt) - d) / (u - d)
    
    asset_prices = np.zeros(num_steps + 1)
    for i in range(num_steps + 1):
      asset_prices[i] = self.S * (u ** (num_steps - i)) * (d ** i)
      
    option_values = np.zeros(num_steps + 1)
    for i in range(num_steps + 1):
      if self.option_type == 'call':
        option_values[i] = max(0, asset_prices[i] - self.K)
      else:
        option_values[i] = max(0, self.K - asset_prices[i])
        
    for i in range(num_steps - 1, -1, -1):
      for j in range(i + 1):
        option_values[j] = np.exp(-self.r * dt) * (p * option_values[j] + (1 - p) * option_values[j + 1])
        
    return option_values[0]
  
  
  def brentq_implied_volatility(option, market_price):
    
    """
    Calculate the implied volatility of an option using the brentq root-finding method.

    Parameters:
    option : Option
        The option for which to calculate the implied volatility.
    market_price : float
        The market price of the option.

    Returns:
    float
        The implied volatility of the option.
    """
    def objective(sigma):
        option.sigma = sigma
        return option.black_scholes() - market_price
    
    try:
        return brentq(objective, 1e-8, 2)
    except ValueError:
        return np.nan
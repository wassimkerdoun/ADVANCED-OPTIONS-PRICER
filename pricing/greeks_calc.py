from scipy.stats import norm
from math import log, sqrt, exp
import numpy as np
import json
from .option_class import Option

class GREEKS:
  
    def __init__(self, option):
        self.option = option
    
    def calculate_greeks(self):
        """
        Calculate the Greeks for European options using the Black-Scholes model.

        Parameters:
        S : float
            Current stock price.
        K : float
            Option strike price.
        T : float
            Time to expiration in years.
        r : float
            Risk-free interest rate.
        sigma : float
            Volatility of the stock.
        option_type : str, optional
            Type of the option, either 'call' or 'put'. Default is 'call'.

        Returns:
        dict
            Dictionary containing the calculated values of delta, gamma, vega, theta, and rho.
        """
        
        d1 = (log(self.option.S / self.option.K) + (self.option.r + 0.5 * self.option.sigma**2) * self.option.T) / (self.option.sigma * sqrt(self.option.T))
        d2 = d1 - self.option.sigma * sqrt(self.option.T)
        
        if self.option.option_type == 'call':
            delta = norm.cdf(d1)
            gamma = norm.pdf(d1) / (self.option.S * self.option.sigma * sqrt(self.option.T))
            vega = self.option.S * norm.pdf(d1) * sqrt(self.option.T) / 100  # Convert to decimal
            theta = (- (self.option.S * norm.pdf(d1) * self.option.sigma) / (2 * sqrt(self.option.T)) - self.option.r * self.option.K * exp(-self.option.r * self.option.T) * norm.cdf(d2)) / 365  # Per day
            rho = self.option.K * self.option.T * exp(-self.option.r * self.option.T) * norm.cdf(d2) / 100  # Convert to decimal
        else:
            delta = -norm.cdf(-d1)
            gamma = norm.pdf(d1) / (self.option.S * self.option.sigma * sqrt(self.option.T))  # Gamma is the same for call and put
            vega = self.option.S * norm.pdf(d1) * sqrt(self.option.T) / 100  # Convert to decimal
            theta = (- (self.option.S * norm.pdf(d1) * self.option.sigma) / (2 * sqrt(self.option.T)) + self.option.r * self.option.K * exp(-self.option.r * self.option.T) * norm.cdf(-d2)) / 365  # Per day
            rho = -self.option.K * self.option.T * exp(-self.option.r * self.option.T) * norm.cdf(-d2) / 100  # Convert to decimal
        
        # Create a dictionary of the results
        results_dict = {
            'Delta': delta,
            'Gamma': gamma,
            'Vega': vega,
            'Theta': theta,
            'Rho': rho
        }
        
        # Return as JSON
        return json.dumps(results_dict)
    
    def simulate_delta_surface(self):
        
        option = self.option
        S = option.S
        K = option.K
        r = option.r
        T = option.T
        sigma = option.sigma
        option_type = option.option_type
        
        S_range = np.linspace(0.50 * S, 1.50 * S, 100)
        K_range = np.linspace(0.50 * K, 1.50 * K, 100)
        delta_surface = np.full((len(S_range), len(K_range)), np.nan)
        
        for i, s in enumerate(S_range):
            for j, k in enumerate(K_range):
                
                temp_option = Option(s, k, T, r, sigma, option_type)
                greeks = GREEKS(temp_option).calculate_greeks()
                delta_surface[i, j] = json.loads(greeks)['Delta']
                
        return delta_surface, S_range, K_range
        

# if __name__ == '__main__':  
#   S = 239
#   K = 230
#   r = 0.05
#   T = 1
#   sigma = 0.23
#   option_type = 'call' 
#   option = Option(S, K, r, T, sigma, option_type)
#   delta_surface = GREEKS(option).simulate_delta_surface()
#   print(delta_surface)
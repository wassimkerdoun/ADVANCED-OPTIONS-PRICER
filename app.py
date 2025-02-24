from flask import Flask, render_template, session, request, redirect, url_for, flash
from datetime import datetime, timedelta
import yfinance as yf
from pricing import greeks_calc, option_class, utils, volatility_utils
from pricing.bs_monte_carlo import *
from pricing.heston_monte_carlo import *
from pricing import plots
import json
import logging

app = Flask(__name__)
app.secret_key = '13a9aad9f23bc281ca3aa07c'

def validate_form_data_home(form_data):
    """Validate form inputs and return error message if any"""
    required_fields = {
        'ticker': 'Stock ticker',
        'strike_price': 'Strike price',
        'maturity': 'Time to maturity',
        'risk_free_rate': 'Risk-free rate',
        'sigma': 'Volatility (sigma)',
        'method': 'Pricing method',
        'option_type': 'Option type'
    }

    for field, name in required_fields.items():
        if not form_data.get(field):
            return f"{name} is required"
        
    try:
        float(form_data.get('strike_price'))
        float(form_data.get('maturity'))
        float(form_data.get('risk_free_rate'))
        float(form_data.get('sigma'))
    except ValueError:
        return "All numeric fields must contain valid numbers"
    
    return None

def validate_form_data_comparison(form_data):
    """Validate form inputs and return error message if any"""
    required_fields = {
        'ticker': 'Stock ticker',
        'strike_price': 'Strike price',
        'maturity': 'Time to maturity',
        'risk_free_rate': 'Risk-free rate',
        'sigma': 'Volatility (sigma)',
        'option_type': 'Option type'
    }

    # Check if all required fields are provided in the form or session
    for field, name in required_fields.items():
        # If the field is missing in the form, try to get it from session
        if not form_data.get(field) and not session.get(field):
            return f"{name} is required"
    
    # Validate numeric fields
    try:
        float(form_data.get('strike_price', session.get('strike_price', 0)))
        float(form_data.get('maturity', session.get('maturity', 0)))
        float(form_data.get('risk_free_rate', session.get('risk_free_rate', 0)))
        float(form_data.get('sigma', session.get('sigma', 0)))
    except ValueError:
        return "All numeric fields must contain valid numbers"

    return None


def save_form_data_home(form_data):
    """Save validated form data to session"""
    session['ticker'] = form_data.get('ticker').upper()
    session['strike_price'] = float(form_data.get('strike_price'))
    session['maturity'] = float(form_data.get('maturity'))
    session['risk_free_rate'] = float(form_data.get('risk_free_rate'))
    session['sigma'] = float(form_data.get('sigma'))
    session['method'] = form_data.get('method')
    session['option_type'] = form_data.get('option_type')

    # Save Monte Carlo parameters if applicable
    if form_data.get('method') in ["Monte Carlo", "Antithetic MC", "CV Monte Carlo"]:
        session['num_paths'] = int(form_data.get('num_paths'))
        session['num_steps'] = int(form_data.get('num_steps'))
        
    elif form_data.get('method') in ["Gbm Discrete", "Euler Maruyama", "Milstein Scheme"]:
        session['num_paths'] = int(form_data.get('num_paths'))
        session['num_steps'] = int(form_data.get('num_steps'))
        
    elif form_data.get('method') in ["Euler Maruyama", "Milstein Scheme"]:
        session['kappa'] = float(form_data.get('kappa'))
        session['theta'] = float(form_data.get('theta'))
        session['vol_of_vol'] = float(form_data.get('vol_of_vol'))
        session['rho'] = float(form_data.get('rho'))
        session['v0'] = float(form_data.get('v0'))
        session['num_paths'] = int(form_data.get('num_paths'))
        session['num_steps'] = int(form_data.get('num_steps'))
        
    elif form_data.get('method') == "MCMC":
        session['num_steps'] = int(form_data.get('num_steps'))
        
        
def save_form_data_comparison(form_data):
    """Save validated form data to session"""
    session['ticker'] = form_data.get('ticker').upper()
    session['strike_price'] = float(form_data.get('strike_price'))
    session['maturity'] = float(form_data.get('maturity'))
    session['risk_free_rate'] = float(form_data.get('risk_free_rate'))
    session['sigma'] = float(form_data.get('sigma'))
    session['option_type'] = form_data.get('option_type')
    session['num_paths'] = int(form_data.get('num_paths'))
    session['num_steps'] = int(form_data.get('num_steps'))
    


def validate_form_data_home_heston(form_data):
    """Validate form inputs and return error message if any"""
    required_fields = {
        'ticker': 'Stock ticker',
        'strike_price': 'Strike price',
        'maturity': 'Time to maturity',
        'risk_free_rate': 'Risk-free rate',
        'sigma': 'Volatility (sigma)',
        'method': 'Pricing method',
        'option_type': 'Option type',
        'kappa': 'Kappa',
        'theta': 'Theta',
        'vol_of_vol': 'Vol of vol',
        'rho': 'Rho',
        'v0': 'V0',
        'num_paths': 'Number of paths',
        'num_steps': 'Number of steps'
    }

    for field, name in required_fields.items():
        if not form_data.get(field):
            return f"{name} is required"
        
    try:
        float(form_data.get('strike_price'))
        float(form_data.get('maturity'))
        float(form_data.get('risk_free_rate'))
        float(form_data.get('sigma'))
        float(form_data.get('kappa'))
        float(form_data.get('theta'))
        float(form_data.get('vol_of_vol'))
        float(form_data.get('rho'))
        float(form_data.get('v0'))
        int(form_data.get('num_paths'))
        int(form_data.get('num_steps'))
    except ValueError:
        return "All numeric fields must contain valid numbers"
    
    return None

def validate_form_data_comparison_heston(form_data):
    """Validate form inputs and return error message if any"""
    required_fields = {
        'ticker': 'Stock ticker',
        'strike_price': 'Strike price',
        'maturity': 'Time to maturity',
        'risk_free_rate': 'Risk-free rate',
        'sigma': 'Volatility (sigma)',
        'option_type': 'Option type',
        'kappa': 'Kappa',
        'theta': 'Theta',
        'vol_of_vol': 'Vol of vol',
        'rho': 'Rho',
        'v0': 'V0',
        'num_paths': 'Number of paths',
        'num_steps': 'Number of steps'
    }

    for field, name in required_fields.items():
        if not form_data.get(field):
            return f"{name} is required"
        
    try:
        float(form_data.get('strike_price'))
        float(form_data.get('maturity'))
        float(form_data.get('risk_free_rate'))
        float(form_data.get('sigma'))
        float(form_data.get('kappa'))
        float(form_data.get('theta'))
        float(form_data.get('vol_of_vol'))
        float(form_data.get('rho'))
        float(form_data.get('v0'))
        int(form_data.get('num_paths'))
        int(form_data.get('num_steps'))
    except ValueError:
        return "All numeric fields must contain valid numbers"
    
    return None



def save_form_data_home_heston(form_data):
    """Save validated form data to session"""
    session['ticker'] = form_data.get('ticker').upper()
    session['strike_price'] = float(form_data.get('strike_price'))
    session['maturity'] = float(form_data.get('maturity'))
    session['risk_free_rate'] = float(form_data.get('risk_free_rate'))
    session['sigma'] = float(form_data.get('sigma'))
    session['method'] = form_data.get('method')
    session['option_type'] = form_data.get('option_type')
    session['kappa'] = float(form_data.get('kappa'))
    session['theta'] = float(form_data.get('theta'))
    session['vol_of_vol'] = float(form_data.get('vol_of_vol'))
    session['rho'] = float(form_data.get('rho'))
    session['v0'] = float(form_data.get('v0'))
    session['num_paths'] = int(form_data.get('num_paths'))
    session['num_steps'] = int(form_data.get('num_steps'))
    
    
def save_form_data_comparison_heston(form_data):
    """Save validated form data to session"""
    session['ticker'] = form_data.get('ticker').upper()
    session['strike_price'] = float(form_data.get('strike_price'))
    session['maturity'] = float(form_data.get('maturity'))
    session['risk_free_rate'] = float(form_data.get('risk_free_rate'))
    session['sigma'] = float(form_data.get('sigma'))
    session['option_type'] = form_data.get('option_type')
    session['kappa'] = float(form_data.get('kappa'))
    session['theta'] = float(form_data.get('theta'))
    session['vol_of_vol'] = float(form_data.get('vol_of_vol'))
    session['rho'] = float(form_data.get('rho'))
    session['v0'] = float(form_data.get('v0'))
    session['num_paths'] = int(form_data.get('num_paths'))
    session['num_steps'] = int(form_data.get('num_steps'))


def fetch_option_price_and_closest_date(ticker, years_to_maturity, strike_price, option_type):
    """Fetch option price and closest expiration date for the given ticker."""
    today = datetime.now()
    target_date = today + timedelta(days=int(365 * years_to_maturity))

    stock = yf.Ticker(ticker)
    available_expirations = stock.options
    if not available_expirations:
        raise ValueError(f"No options available for ticker {ticker}")

    # Find closest expiration date
    available_dates = [datetime.strptime(date, "%Y-%m-%d") for date in available_expirations]
    closest_date = min(available_dates, key=lambda x: abs(x - target_date))
    closest_expiration_date = closest_date.strftime("%Y-%m-%d")

    # Fetch option chain for closest expiration date
    option_chain = stock.option_chain(closest_expiration_date)
    options = option_chain.calls if option_type.lower() == "call" else option_chain.puts
    option_row = options[options['strike'] == strike_price]

    if option_row.empty:
        raise ValueError(f"No option found with strike price {strike_price} for expiration {closest_expiration_date}")

    option_price = option_row['lastPrice'].values[0]
    underlying_price = stock.history(period="1d")['Close'].iloc[-1]

    return underlying_price, closest_expiration_date, option_price

# ------------------------------------------------------------------------------
# -------------------------------- Pages --------------------------------------- 

@app.route("/", methods=["GET", "POST"])
@app.route("/home", methods=["GET", "POST"])
def home_page():
    return render_template("index.html")
    


@app.route("/home1", methods=["GET", "POST"])
def home_page1():
    if request.method == "POST":
        error = validate_form_data_home(request.form)
        if error:
            flash(error, 'danger')
            return redirect(url_for('home_page1'))

        try:
            save_form_data_home(request.form)
            ticker = session['ticker']
            S, closest_expiration_date, market_price = fetch_option_price_and_closest_date(
                ticker, session['maturity'], session['strike_price'], session['option_type']
            )

            if S is None:
                flash(f"Could not fetch price for ticker {ticker}", 'danger')
                return redirect(url_for('home_page1'))
            

            option = option_class.Option(
                S=S, K=session['strike_price'], T=session['maturity'],
                r=session['risk_free_rate'], sigma=session['sigma'],
                option_type=session['option_type']
            )

            session['underlying_price'] = S
            

            # Calculate Greeks
            greeks_values = greeks_calc.GREEKS(option).calculate_greeks()
            session['greeks'] = json.loads(greeks_values)

            method = session['method']
            price = None

            if method == "Black-Scholes":
                price = option.black_scholes()
                delta_surface, S_range, K_range = greeks_calc.GREEKS(option).simulate_delta_surface()
                delta_surface_filename = plots.plot_delta_surface(delta_surface, S_range, K_range)
                session['delta_surface_filename'] = delta_surface_filename
            elif method == "MCMC":
                price, S_T_samples = option.MCMC(session['num_steps'])
                mcmc_plot_filename = plots.plot_MCMC_density(S_T_samples)
                session['mcmc_plotfilename'] = mcmc_plot_filename
            elif method in ["Monte Carlo", "Antithetic MC", "CV Monte Carlo"]:
                if method == 'Monte Carlo':
                    price, paths = option.monte_carlo(session['num_paths'], session['num_steps'])
                elif method == 'Antithetic MC':
                    price, paths = option.antithetic_monte_carlo(session['num_paths'], session['num_steps'])
                elif method == 'CV Monte Carlo':
                    price, paths = option.control_variates_monte_carlo(session['num_paths'], session['num_steps'])
                    
                plot_filename = plots.plot_mc_paths_density(paths)
                session['plot_filename'] = plot_filename
            else:
                flash("Invalid pricing method selected", 'danger')
                return redirect(url_for('home_page1'))

            # Calculate implied volatility
            implied_vol = option_class.Option.brentq_implied_volatility(option, market_price)

            # Option pricing logic
            filename = plots.plot_price_surface(option)
            session['plotly_json_path'] = url_for('static', filename=f'plotly/{filename}')
            plot_filename2 = plots.option_price(option)
            session['plot_filename2'] = plot_filename2
            
            # Store results in session
            session['expiration_date'] = (datetime.now() + timedelta(days=int(365 * session['maturity']))).strftime('%Y-%m-%d')
            session['price'] = price
            session['true_price'] = market_price
            session['closest_expiration_date'] = closest_expiration_date
            session['implied_volatility'] = implied_vol
            flash("Calculation completed successfully", 'success')

        except Exception as e:
            logging.error(f"Error in home_page: {str(e)}", exc_info=True)
            flash(f"Calculation error: {str(e)}", 'danger')

        return redirect(url_for('home_page1'))

    # Context dictionary for rendering
    context = {
        'price': session.get('price', None),
        'true_price': session.get('true_price', None),
        'maturity': session.get('maturity', None),
        'closest_expiration_date': session.get('closest_expiration_date', None),
        'greeks': session.get('greeks', None),
        'expiration_date': session.get('expiration_date', None),
        'implied_volatility': session.get('implied_volatility', None),
        'delta_surface_filename': session.get('delta_surface_filename', None),
        'plot_filename': session.get('plot_filename', None),
        'plot_filename2': session.get('plot_filename2', None),
        'mcmc_plotfilename': session.get('mcmc_plotfilename', None),
        'plotly_json_path': session.get('plotly_json_path', None),
        'messages': {'danger': session.get('_flashes', None)},
        'current_route': request.endpoint
    }
    
    print(context)

    return render_template("home1.html", **context)



@app.route('/comparison1', methods=['GET', 'POST'])
def comparison_page1():
    if request.method == "POST":
        # Validate form data
        error = validate_form_data_comparison(request.form)
        if error:
            flash(error, 'danger')
            return redirect(url_for('comparison_page1'))

        try:
            # Save form data to session
            save_form_data_comparison(request.form)

            # Fetch option data
            ticker = session['ticker']
            S, closest_expiration_date, market_price = fetch_option_price_and_closest_date(
                ticker, session['maturity'], session['strike_price'], session['option_type']
            )

            if S is None:
                flash(f"Could not fetch price for ticker {ticker}", 'danger')
                return redirect(url_for('home_page'))

            # Create Option object
            option = option_class.Option(
                S=S, K=session['strike_price'], T=session['maturity'],
                r=session['risk_free_rate'], sigma=session['sigma'],
                option_type=session['option_type']
            )


            # Analyze convergence speed
            convergence, errors, time_dict = utils.convergence_speed(option, session['num_paths'], session['num_steps'])
            
            # Generate convergence and error plots
            convergence_filename, errors_filename, time_dict_filename = plots.plot_convergence_speed(convergence, errors, time_dict)
            session['convergence_filename'] = convergence_filename
            session['errors_filename'] = errors_filename
            session['time_dict_filename'] = time_dict_filename

            flash("Calculation completed successfully", 'success')

        except Exception as e:
            logging.error(f"Error in comparison_page: {str(e)}", exc_info=True)
            flash(f"Calculation error: {str(e)}", 'danger')

        return redirect(url_for('comparison_page1'))  # Redirect to prevent re-submission on refresh

    print(session)
    
    # Retrieve session data for GET request
    context = {
        'convergence_filename': session.get('convergence_filename', None),
        'errors_filename': session.get('errors_filename', None),
        'time_dict_filename': session.get('time_dict_filename', None),
        'messages': {'danger': session.get('_flashes', None)},
        'current_route': request.endpoint  # Pass the current route to the template
    }
    
    print('Context', context)

    return render_template('comparison1.html', **context)


@app.route("/home2", methods=["GET", "POST"])
def home_page2():
    if request.method == "POST":
        error = validate_form_data_home(request.form)
        if error:
            flash(error, 'danger')
            return redirect(url_for('home_page2'))

        try:
            save_form_data_home(request.form)
            ticker = session['ticker']
            S, closest_expiration_date, market_price = fetch_option_price_and_closest_date(
                ticker, session['maturity'], session['strike_price'], session['option_type']
            )

            if S is None:
                flash(f"Could not fetch price for ticker {ticker}", 'danger')
                return redirect(url_for('home_page2'))
            
            session['underlying_price'] = S

            option = option_class.Option(
                S=S, K=session['strike_price'], T=session['maturity'],
                r=session['risk_free_rate'], sigma=session['sigma'],
                option_type=session['option_type']
            )

            # Option pricing logic
            filename = plots.plot_price_surface(option)
            session['plotly_json_path'] = url_for('static', filename=f'plotly/{filename}')
            plot_filename2 = plots.option_price(option)
            session['plot_filename2'] = plot_filename2

            # Calculate Greeks
            greeks_values = greeks_calc.GREEKS(option).calculate_greeks()
            session['greeks'] = json.loads(greeks_values)

            method = session['method']
            price = None

            if method == "Black-Scholes":
                price = option.black_scholes()
                delta_surface, S_range, K_range = greeks_calc.GREEKS(option).simulate_delta_surface()
                delta_surface_filename = plots.plot_delta_surface(delta_surface, S_range, K_range)
                session['delta_surface_filename'] = delta_surface_filename
            elif method in ["Gbm Discrete", "Euler Maruyama", "Milstein Scheme"]:
                if method == 'Gbm Discrete':
                    price, paths = gbm_discrete(option, session['num_paths'], session['num_steps'])
                elif method == 'Euler Maruyama':
                    price, paths = bs_euler_maruyama(option, session['num_paths'], session['num_steps'])
                elif method == 'Milstein Scheme':
                    price, paths = bs_milstein_scheme(option, session['num_paths'], session['num_steps'])
                    
                plot_filename = plots.plot_mc_paths_density(paths)
                session['plot_filename'] = plot_filename
            else:
                flash("Invalid pricing method selected", 'danger')
                return redirect(url_for('home_page2'))

            # Calculate implied volatility
            implied_vol = option_class.Option.brentq_implied_volatility(option, market_price)

            # Store results in session
            session['expiration_date'] = (datetime.now() + timedelta(days=int(365 * session['maturity']))).strftime('%Y-%m-%d')
            session['price'] = price
            session['true_price'] = market_price
            session['closest_expiration_date'] = closest_expiration_date
            session['implied_volatility'] = implied_vol
            flash("Calculation completed successfully", 'success')

        except Exception as e:
            logging.error(f"Error in home_page: {str(e)}", exc_info=True)
            flash(f"Calculation error: {str(e)}", 'danger')

        return redirect(url_for('home_page2'))

    # Context dictionary for rendering
    context = {
        'price': session.get('price', None),
        'true_price': session.get('true_price', None),
        'maturity': session.get('maturity', None),
        'closest_expiration_date': session.get('closest_expiration_date', None),
        'greeks': session.get('greeks', None),
        'expiration_date': session.get('expiration_date', None),
        'implied_volatility': session.get('implied_volatility', None),
        'delta_surface_filename': session.pop('delta_surface_filename', None),
        'plot_filename': session.get('plot_filename', None),
        'plot_filename2': session.get('plot_filename2', None),
        'mcmc_plotfilename': session.get('mcmc_plotfilename', None),
        'plotly_json_path': session.get('plotly_json_path', None),
        'messages': {'danger': session.get('_flashes', None)},
        'current_route': request.endpoint
    }

    return render_template("home2.html", **context)


@app.route('/comparison2', methods=['GET', 'POST'])
def comparison_page2():
    if request.method == "POST":
        # Validate form data
        error = validate_form_data_comparison(request.form)
        if error:
            flash(error, 'danger')
            return redirect(url_for('comparison_page2'))

        try:
            # Save form data to session
            save_form_data_comparison(request.form)

            # Fetch option data
            ticker = session['ticker']
            S, closest_expiration_date, market_price = fetch_option_price_and_closest_date(
                ticker, session['maturity'], session['strike_price'], session['option_type']
            )

            if S is None:
                flash(f"Could not fetch price for ticker {ticker}", 'danger')
                return redirect(url_for('home_page'))

            # Create Option object
            option = option_class.Option(
                S=S, K=session['strike_price'], T=session['maturity'],
                r=session['risk_free_rate'], sigma=session['sigma'],
                option_type=session['option_type']
            )

            # Analyze convergence speed
            results, errors, time_dict = bs_convergence_speed(option, session['num_paths'], session['num_steps'])
            
            
            # Generate convergence and error plots
            results_filename, errors_filename, time_dict_filename = plots.bs_plot_convergence_speed(results, errors, time_dict)
            session['results_filename'] = results_filename
            session['errors_filename'] = errors_filename
            session['time_dict_filename'] = time_dict_filename

            flash("Calculation completed successfully", 'success')

        except Exception as e:
            logging.error(f"Error in comparison_page: {str(e)}", exc_info=True)
            flash(f"Calculation error: {str(e)}", 'danger')

        return redirect(url_for('comparison_page2'))  # Redirect to prevent re-submission on refresh

    print(session)
    
    # Retrieve session data for GET request
    context = {
        'results_filename': session.get('results_filename', None),
        'errors_filename': session.get('errors_filename', None),
        'time_dict_filename': session.get('time_dict_filename', None),
        'messages': {'danger': session.get('_flashes', None)},
        'current_route': request.endpoint  # Pass the current route to the template
    }
    
    print('Context', context)

    return render_template('comparison2.html', **context)



@app.route("/home3", methods=["GET", "POST"])
def home_page3():
    if request.method == "POST":
        error = validate_form_data_home_heston(request.form)
        if error:
            flash(error, 'danger')
            return redirect(url_for('home_page3'))

        try:
            save_form_data_home_heston(request.form)
            ticker = session['ticker']
            S, closest_expiration_date, market_price = fetch_option_price_and_closest_date(
                ticker, session['maturity'], session['strike_price'], session['option_type']
            )

            if S is None:
                flash(f"Could not fetch price for ticker {ticker}", 'danger')
                return redirect(url_for('home_page3'))
            
            session['underlying_price'] = S

            option = option_class.Option(
                S=S, K=session['strike_price'], T=session['maturity'],
                r=session['risk_free_rate'], sigma=session['sigma'],
                option_type=session['option_type']
            )

            # Option pricing logic
            filename = plots.plot_price_surface(option)
            session['plotly_json_path'] = url_for('static', filename=f'plotly/{filename}')
            plot_filename2 = plots.option_price(option)
            session['plot_filename2'] = plot_filename2

            # Calculate Greeks
            greeks_values = greeks_calc.GREEKS(option).calculate_greeks()
            session['greeks'] = json.loads(greeks_values)

            method = session['method']
            price = None
            
            kappa = session['kappa']
            theta = session['theta']
            vol_of_vol = session['vol_of_vol']
            rho = session['rho']
            v0 = session['v0']
            num_paths = session['num_paths']
            num_steps = session['num_steps']

            if method in ["Euler Maruyama", "Milstein Scheme"]:
                if method == 'Euler Maruyama':
                    price, price_paths = heston_euler_maruyama(option, kappa, theta, vol_of_vol, rho, v0, num_paths, num_steps)
                    v_paths = volatility_euler_maruyama(option, kappa, theta, vol_of_vol, v0, num_paths, num_steps)
                elif method == 'Milstein Scheme':
                    price, price_paths = heston_milstein_scheme(option, kappa, theta, vol_of_vol, rho, v0, num_paths, num_steps)
                    v_paths = volatility_milstein_scheme(option, kappa, theta, vol_of_vol, v0, num_paths, num_steps)
                    
                price_plot_filename = plots.plot_mc_paths_density(price_paths)
                session['price_plot_filename'] = price_plot_filename
                volatility_filename = plots.plot_vol_paths_density(v_paths)
                session['volatility_filename'] = volatility_filename
                
            else:
                flash("Invalid pricing method selected", 'danger')
                return redirect(url_for('home_page'))

            # Calculate implied volatility
            implied_vol = option_class.Option.brentq_implied_volatility(option, market_price)

            # Store results in session
            session['expiration_date'] = (datetime.now() + timedelta(days=int(365 * session['maturity']))).strftime('%Y-%m-%d')
            session['price'] = price
            session['true_price'] = market_price
            session['closest_expiration_date'] = closest_expiration_date
            session['implied_volatility'] = implied_vol

        except Exception as e:
            logging.error(f"Error in home_page: {str(e)}", exc_info=True)
            flash(f"Calculation error: {str(e)}", 'danger')

        return redirect(url_for('home_page3'))

    # Context dictionary for rendering
    context = {
        'price': session.get('price', None),
        'true_price': session.get('true_price', None),
        'maturity': session.get('maturity', None),
        'closest_expiration_date': session.get('closest_expiration_date', None),
        'kaapa': session.get('kappa', None),
        'theta': session.get('theta', None),
        'vol_of_vol': session.get('vol_of_vol', None),
        'rho': session.get('rho', None),
        'v0': session.get('v0', None),
        'greeks': session.get('greeks', None),
        'expiration_date': session.get('expiration_date', None),
        'implied_volatility': session.get('implied_volatility', None),
        'price_plot_filename': session.get('price_plot_filename', None),
        'volatility_filename': session.get('volatility_filename', None),
        'plot_filename2': session.get('plot_filename2', None),
        'plotly_json_path': session.get('plotly_json_path', None),
        'messages': {'danger': session.get('_flashes', None)},
        'current_route': request.endpoint
    }

    return render_template("home3.html", **context)


@app.route('/comparison3', methods=['GET', 'POST'])
def comparison_page3():
    if request.method == "POST":
        # Validate form data
        error = validate_form_data_comparison_heston(request.form)
        if error:
            flash(error, 'danger')
            return redirect(url_for('comparison_page3'))

        try:
            # Save form data to session
            save_form_data_comparison_heston(request.form)

            # Fetch option data
            ticker = session['ticker']
            S, closest_expiration_date, market_price = fetch_option_price_and_closest_date(
                ticker, session['maturity'], session['strike_price'], session['option_type']
            )

            if S is None:
                flash(f"Could not fetch price for ticker {ticker}", 'danger')
                return redirect(url_for('home_page'))

            # Create Option object
            option = option_class.Option(
                S=S, K=session['strike_price'], T=session['maturity'],
                r=session['risk_free_rate'], sigma=session['sigma'],
                option_type=session['option_type']
            )
            
            kappa = session['kappa']
            theta = session['theta']
            vol_of_vol = session['vol_of_vol']            
            rho = session['rho']
            v0 = session['v0']
            num_paths = session['num_paths']
            num_steps = session['num_steps']
            
            # Analyze convergence speed
            results, errors, time_dict = heston_convergence_speed(option, kappa, theta, vol_of_vol, rho, v0, num_paths, num_steps)
            
            
            # Generate convergence and error plots
            results_filename, errors_filename, time_dict_filename = plots.heston_plot_convergence_speed(results, errors, time_dict)
            session['results_filename'] = results_filename
            session['errors_filename'] = errors_filename
            session['time_dict_filename'] = time_dict_filename

            flash("Calculation completed successfully", 'success')

        except Exception as e:
            logging.error(f"Error in comparison_page: {str(e)}", exc_info=True)
            flash(f"Calculation error: {str(e)}", 'danger')

        return redirect(url_for('comparison_page3'))  # Redirect to prevent re-submission on refresh

    print(session)
    
    # Retrieve session data for GET request
    context = {
        'results_filename': session.get('results_filename', None),
        'errors_filename': session.get('errors_filename', None),
        'time_dict_filename': session.get('time_dict_filename', None),
        'messages': {'danger': session.get('_flashes', None)},
        'current_route': request.endpoint  # Pass the current route to the template
    }
    
    print('Context', context)

    return render_template('comparison3.html', **context)


@app.route("/volatility", methods=["GET", "POST"])
def volatility_page():
    
    if request.method == "POST":
        # Validate form data
        error = validate_form_data_comparison_heston(request.form)
        if error:
            flash(error, 'danger')
            return redirect(url_for('volatility_page'))

        try:
            # Save form data to session
            save_form_data_comparison_heston(request.form)

            # Fetch option data
            ticker = session['ticker']
            S, closest_expiration_date, market_price = fetch_option_price_and_closest_date(
                ticker, session['maturity'], session['strike_price'], session['option_type']
            )

            if S is None:
                flash(f"Could not fetch price for ticker {ticker}", 'danger')
                return redirect(url_for('home_page'))

            # Create Option object
            option = option_class.Option(
                S=S, K=session['strike_price'], T=session['maturity'],
                r=session['risk_free_rate'], sigma=session['sigma'],
                option_type=session['option_type']
            )
            
            kappa = session['kappa']
            theta = session['theta']
            vol_of_vol = session['vol_of_vol']            
            rho = session['rho']
            v0 = session['v0']
            num_paths = session['num_paths']
            num_steps = session['num_steps']
            
            ivs, K_range_smile = volatility_utils.volatility_smile(option, kappa, theta, vol_of_vol, rho, v0, num_paths, num_steps)
            volatility_smile_filename = volatility_utils.plot_volatility_smile(ivs, K_range_smile)
            session['volatility_smile_filename'] = volatility_smile_filename
            
            iv_surface, K_range_surface, T_range = volatility_utils.volatility_surface(option, kappa, theta, vol_of_vol, rho, v0, num_paths, num_steps)
            volatility_surface_filename = volatility_utils.plot_volatility_surface(iv_surface, K_range_surface, T_range)
            session['volatility_surface_filename'] = volatility_surface_filename

            flash("Calculation completed successfully", 'success')

        except Exception as e:
            logging.error(f"Error in comparison_page: {str(e)}", exc_info=True)
            flash(f"Calculation error: {str(e)}", 'danger')

        return redirect(url_for('volatility_page'))  # Redirect to prevent re-submission on refresh

    print(session)
    
    # Retrieve session data for GET request
    context = {
        'volatility_smile_filename': session.get('volatility_smile_filename', None),
        'volatility_surface_filename': session.get('volatility_surface_filename', None),
        'messages': {'danger': session.get('_flashes', None)},
        'current_route': request.endpoint  # Pass the current route to the template
    }
    
    print('Context', context)
    
    
    return render_template("volatility_page.html", **context)


@app.route("/clear_session")
def clear_session():
    session.clear()
    return redirect(url_for("home_page"))



if __name__ == "__main__":
    app.run(debug=True)

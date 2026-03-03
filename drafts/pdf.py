"""" Estimating the risk-neutral probability density function from the option prices.
        - This file uses a heston model that is calibrated from historical stock prices, to obtain volatility meassures. 
        - Because we use historical stock prices, we miss out on actual implied volatility measures observed in actual option contracts. 
        - This will lead to a less accurate rnd. 
 """
from pathlib import Path
import sys 
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from bin.dbm import DBManager
from bin.main import Manager 
from bin.options.bsm.bs import bs_df


manager = Manager()
stock = manager.get_stock_data('spy')
# use_cache = False, returns the full option chain 
cdf = stock.options.option_chain_on(chain_date = '2026-02-18', use_cache = False)
cdf = bs_df(cdf)
expirys = cdf.sort_values(by = 'expiry').expiry.unique()

# keep otm options only
otm_calls = cdf[(cdf.strike > cdf.stk_price) & (cdf.type == 'Call')]
otm_puts = cdf[(cdf.strike < cdf.stk_price) & (cdf.type == 'Put')]
cdf = pd.concat([otm_calls, otm_puts])
cdf = cdf[cdf.gatherdate == cdf.gatherdate.max()]

gcdf = cdf[cdf.expiry == expirys[0]].copy() # Front month 

from models.densityEstimation.iter1.curve_fitting import bspline, smoothing_spline
x_name = 'strike'
y_name = 'lastprice'

print(gcdf.shape, gcdf.expiry.unique())
gcdf = gcdf.groupby(x_name).agg({y_name: 'sum', 'timevalue': 'mean'}).reset_index()

x = gcdf.sort_values(by = x_name)[x_name].values
y = gcdf.sort_values(by = x_name)[y_name].values

xb, yb = bspline(x, y, k = 3)
xx, yy = smoothing_spline(x, y)

fig, ax = plt.subplots(figsize = (10, 5))
ax.plot(x, y, color = 'black', label = 'Original')
ax.plot(xb, yb, color = 'red', label = 'B-Spline')
ax.plot(xx, yy, color = 'green', label = 'Smoothing Spline')
max_y = max(y)
ax.vlines(stock.price_data.current_price, 0, max_y, color = 'blue', label = 'Current Price')
ax.set_title('Gamma Profile')
ax.set_xlabel('Strike')
ax.set_ylabel('Gamma')
ax.legend()
plt.show()


### Get Optimized Parameters from Heston Model 
from bin.price.sims.priceSims import simulated_prices
pdf = stock.price_data.daily_df
stock_prices, params = simulated_prices(stock.stock, pdf, method = 'heston_path', days = 10, return_params = True)

# Initialise parameters
S0 = stock.price_data.current_price     # initial stock price
# K = 150.0      # strike price
tau = gcdf.timevalue.values        # time to maturity in years
r = 0.06       # annual risk-free rate (We should add an endpoint, to get the most current 10 year yield)

# Heston dependent parameters
kappa = params.kappa             # rate of mean reversion of variance under risk-neutral dynamics
theta = params.theta        # long-term mean of variance under risk-neutral dynamics
v0 = params.v_0           # initial variance under risk-neutral dynamics
rho = params.rho              # correlation between returns and variances under risk-neutral dynamics
sigma = params.xi            # volatility of volatility
lambd = params.lam              # risk premium of variance


# Heston Characteristic Function
def heston_charfunc(phi, S0, v0, kappa, theta, sigma, rho, lambd, tau, r):

    # constants
    a = kappa*theta
    b = kappa+lambd

    # common terms w.r.t phi
    rspi = rho*sigma*phi*1j

    # define d parameter given phi and b
    d = np.sqrt( (rspi - b)**2 + (phi*1j+phi**2)*sigma**2 )

    # define g parameter given phi, b and d
    g = (b-rspi+d)/(b-rspi-d)

    # calculate characteristic function by components
    exp1 = np.exp(r*phi*1j*tau)
    term2 = S0**(phi*1j) * ( (1-g*np.exp(d*tau))/(1-g) )**(-2*a/sigma**2)
    exp2 = np.exp(a*tau*(b-rspi+d)/sigma**2 + v0*(b-rspi+d)*( (1-np.exp(d*tau))/(1-g*np.exp(d*tau)) )/sigma**2)

    return exp1*term2*exp2

def heston_price_rec(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r):
    args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)

    P, umax, N = 0, 100, 650
    dphi=umax/N #dphi is width
    for j in range(1,N):
        # rectangular integration
        phi = dphi * (2*j + 1)/2 # midpoint to calculate height
        numerator = heston_charfunc(phi-1j,*args) - K * heston_charfunc(phi,*args)
        denominator = 1j*phi*K**(1j*phi)

        P += dphi * numerator/denominator

    return np.real((S0 - K*np.exp(-r*tau))/2 + P/np.pi)



strikes = x 
option_prices = heston_price_rec(S0, strikes, v0, kappa, theta, sigma, rho, lambd, tau, r)

prices = pd.DataFrame([strikes, option_prices]).transpose()
prices.columns = ['strike', 'price']
prices['curvature'] = (-2 * prices['price'] +prices['price'].shift(1) + prices['price'].shift(-1)) / 1**2


# And plotting...
def plot_pdf(strikes, option_prices, prices, fig, ax, stock_price = None ):
    ax2 = ax.twinx()

    ax.plot(strikes, option_prices, label='Option Prices')
    ax2.plot(prices['strike'], prices['curvature'], label='$d^2C/dK^2 (\sim pdf)$', color='orange')
    # Scatter point at the max curvature
    ax.scatter(prices['strike'][prices['curvature'].idxmax()], prices['price'][prices['curvature'].idxmax()], color = 'red', label = f'{prices["strike"][prices["curvature"].idxmax()]:.2f}')
    # Vertical line at last stock price 
    if stock_price is not None:
        ax.vlines(stock_price, 0, prices['curvature'].max(), color = 'blue', label = 'Current Price')
    ax.legend(loc="center right")
    ax2.legend(loc="upper right")
    plt.xlabel('Strikes (K)')
    plt.ylabel('$f_\\tau(K)$')
    plt.title('Risk-neutral PDF, $f_\mathbb{Q}(K, \\tau)$')
    # Mu, sigma, skew and kurtosis for the curvature

    mu = prices['curvature'].mean()
    sigma = prices['curvature'].std()
    skew = prices['curvature'].skew()
    kurt = prices['curvature'].kurt()

    print(f"Mu: {mu}, Sigma: {sigma}, Skew: {skew}, Kurtosis: {kurt}")
    ax.grid(True)
    return ax

def plot_interpolation(prices, fig, ax ):

    inter = prices.dropna()

    pdf = sc.interpolate.interp1d(inter.strike, np.exp(r*tau[0])*inter.curvature, kind = 'linear')
    pdfc = sc.interpolate.interp1d(inter.strike, np.exp(r*tau[0])*inter.curvature, kind = 'cubic')

    strikes = inter.strike

    ax.plot(strikes, pdfc(strikes), '-+', label='cubic')
    ax.plot(strikes, pdf(strikes), label='linear')
    ax.fill_between(strikes, pdf(strikes), color='yellow', alpha=0.2)
    ax.set_xlabel('Strikes (K)')
    ax.set_ylabel('$f_\\tau(K)$')
    ax.set_title('Risk-neutral PDF: $f_\mathbb{Q}(K, \\tau)$')
    ax.legend()
    ax.grid(True)
    return ax




lastprice = stock.price_data.current_price
fig, ax = plt.subplots(1, 2, figsize = (12, 6))
ax = ax.flatten()
plot_pdf(strikes, option_prices, prices, fig, ax[0], stock_price = lastprice)
plot_interpolation(prices, fig, ax[1])
plt.show()
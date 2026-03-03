import sys 
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from bin.dbm import DBManager
from bin.main import Manager
from bin.options.bsm.bs import bs_df


from anderson_huge import agg_by_strike, get_front_month_chain, andreasen_huge_fit, ah_fit_vega, andreasen_huge_fit_exposure, filter_otm_options
from density.heston_calibrator import HestonCalibrator, FlowAwarePDF, FlowData, NUMBA_AVAILABLE
from density.pipe import preprocess_option_chain, get_front_month_chain


dbm = DBManager().for_notebook()
manager = Manager(dbm = dbm)
stock = manager.get_stock_data('spy')
price_data = stock.price_data.daily_df
# option_chain =  stock.options.option_chain_on(chain_date = '2026-02-20', use_cache = False)
option_chain = stock.options.chain_df


ochain = bs_df(option_chain)
ochain = ochain[ochain.gatherdate == ochain.gatherdate.max()]
gcdf = preprocess_option_chain(ochain, expiry = 2)
print(gcdf.columns)
r = 0.0405
q = 0.00
S0 = gcdf['stk_price'].iloc[-1]
tau = gcdf['timevalue'].mean()
strikes = gcdf['strike'].values
market_prices = gcdf['lastprice'].values
bs_model_prices = gcdf['fairvalue'].values
actual_ivs = gcdf['impliedvolatility'].values
gamma_exposure = gcdf['gexp'].values
charm_exposure = gcdf['cexp'].values
vanna_exposure = gcdf['vexp'].values
volume = gcdf['volume'].values
openinterest = gcdf['openinterest'].values
oi_chg = gcdf['oi_chg'].values


N, umax = HestonCalibrator.auto_select_params(tau, strikes, S0)
calibrator = HestonCalibrator(N = N, umax = umax)
if NUMBA_AVAILABLE:
    result = calibrator.calibrate_numba(
        strikes = strikes,
        market_prices = market_prices,
        S0 = S0,
        tau = tau,
        r = r,
        weight_type = 'atm', # 'tail', 'uniform'
        flow_weights = {'gamma': gamma_exposure, 'charm': charm_exposure},
        pop_size = 30,
        max_iter = 100,
    )

else:
    print('Numba not available, falling back to standard calibration')
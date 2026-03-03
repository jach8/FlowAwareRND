import pandas as pd


def agg_by_strike(option_chain, x = 'strike', y = 'lastprice'):
    gcdf = option_chain.groupby(['timevalue',x]).agg({
        'gexp': 'sum', 
        'cexp': 'sum', 
        'vexp': 'sum',
        'vega': 'mean',
        'impliedvolatility': 'mean',
        'volume': 'sum',
        'openinterest': 'sum',
        'oi_chg': 'sum', ## This will be net oi change per strike, so if one strike goes down, the other goes up, the net oi change will be 0.
         y: 'mean',
         'fairvalue': 'mean',
        'stk_price': 'last'})
    return gcdf.reset_index()

def get_front_month_chain(option_chain, expiry = None):
    expiry_dates = sorted(option_chain.expiry.unique())
    gather_dates = sorted(option_chain.gatherdate.unique())
    print(f'Found {len(gather_dates)} gather dates: {gather_dates}')
    if expiry is not None:
        if type(expiry) == str:
            expiry = pd.to_datetime(expiry)
        if expiry in expiry_dates:
            print(f"Using specified expiry: {expiry}")
            return option_chain[option_chain.expiry == expiry]
        if type(expiry) == int:
            if 0 <= expiry < len(expiry_dates):
                print(f"Using expiry at index {expiry}: {expiry_dates[expiry]}")
                return option_chain[option_chain.expiry == expiry_dates[expiry]]
            else:
                raise ValueError(f"Expiry index {expiry} out of range.")
        else:
            current_datetime = pd.Timestamp.now()
            if current_datetime.hour > 15: 
                expiry = expiry_dates[1]
                print(f"Current time is after 3 PM. Using next expiry: {expiry}")
            else:
                expiry = expiry_dates[0]
            
            return option_chain[option_chain.expiry == expiry]

def filter_otm_options(cdf):
    # keep otm options only
    otm_calls = cdf[(cdf.strike > cdf.stk_price) & (cdf.type == 'Call')]
    otm_puts = cdf[(cdf.strike < cdf.stk_price) & (cdf.type == 'Put')]

    cdf = pd.concat([otm_calls, otm_puts])
    return cdf

def preprocess_option_chain(option_chain, expiry = 0):
    option_chain = filter_otm_options(option_chain)
    option_chain = get_front_month_chain(option_chain, expiry = expiry)
    gcdf = agg_by_strike(option_chain, x = 'strike', y = 'lastprice')
    return gcdf
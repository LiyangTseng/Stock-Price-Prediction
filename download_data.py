from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
import os
import pandas as pd
import time

# api_key = 'R3LDTDSO4EMBKX0W'
# api_key = 'V5W5P31ODSAKRIZL'
api_key = '3E1WQDGBDJDBRDGR'
company_list = ['INTC', 'AMD', 'CSCO', 'AAPL', 'MU', 'NVDA', 'QCOM', 'AMZN', 'NFLX', 'FB',
                 'GOOG', 'BABA', 'EBAY', 'IBM', 'XLNX', 'TXN', 'NOK', 'TSLA', 'MSFT', 'SNPS']

data_dir = 'Company_Data/'
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

print('========== download latest stock data via download_data.py ==========')
for company in company_list:
    company_data_path = os.path.join(data_dir, company+'.csv')
    sma, _ = TechIndicators(key=api_key).get_sma(company, interval='daily', series_type='close')
    sma_df = pd.DataFrame(sma).T
    time.sleep(12) # standard API call frequency is 5 calls per minute
    ema, _ = TechIndicators(key=api_key).get_ema(company, interval='daily', series_type='close')
    ema_df = pd.DataFrame(ema).T
    time.sleep(12) # standard API call frequency is 5 calls per minute
    rsi, _ = TechIndicators(key=api_key).get_rsi(company, interval='daily', series_type='close')
    rsi_df = pd.DataFrame(rsi).T
    time.sleep(12) # standard API call frequency is 5 calls per minute
    stoch, _ = TechIndicators(key=api_key).get_stoch(company, interval='daily')
    stoch_df = pd.DataFrame(stoch).T
    time.sleep(12) # standard API call frequency is 5 calls per minute
    macd, _ = TechIndicators(key=api_key).get_macd(company, interval='daily', series_type='close')
    macd_df = pd.DataFrame(macd).T
    time.sleep(12) # standard API call frequency is 5 calls per minute
    adx, _ = TechIndicators(key=api_key).get_adx(company, interval='daily')
    adx_df = pd.DataFrame(adx).T
    time.sleep(12) # standard API call frequency is 5 calls per minute
    cci, _ = TechIndicators(key=api_key).get_cci(company, interval='daily')
    cci_df = pd.DataFrame(cci).T
    time.sleep(12) # standard API call frequency is 5 calls per minute
    aroon, _ = TechIndicators(key=api_key).get_aroon(company, interval='daily', series_type='close')
    aroon_df = pd.DataFrame(aroon).T
    time.sleep(12) # standard API call frequency is 5 calls per minute
    bbands, _ = TechIndicators(key=api_key).get_bbands(company, interval='daily', series_type='close')
    bbands_df = pd.DataFrame(bbands).T
    time.sleep(12) # standard API call frequency is 5 calls per minute
    ad, _ = TechIndicators(key=api_key).get_ad(company, interval='daily')
    ad_df = pd.DataFrame(ad).T
    time.sleep(12) # standard API call frequency is 5 calls per minute
    obv, _ = TechIndicators(key=api_key).get_obv(company, interval='daily')
    obv_df = pd.DataFrame(obv).T
    time.sleep(12) # standard API call frequency is 5 calls per minute
    data, _ = TimeSeries(key=api_key).get_daily_adjusted(company, outputsize='full')
    ts_df = pd.DataFrame(data).T
    time.sleep(12) # standard API call frequency is 5 calls per minute
    df_list = [sma_df, ema_df, rsi_df, stoch_df, macd_df, adx_df, cci_df, aroon_df, bbands_df, ad_df, obv_df]
    data_df = ts_df.join(df_list).dropna() # drop rows with null value

    data_df.to_csv(company_data_path)
    print('Complete downloading {comp} data'.format(comp=company))
    
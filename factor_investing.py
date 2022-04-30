import os
import json
import boto3
import requests
import pandas as pd
from time import sleep
import csv
import math
import logging
import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_resource = boto3.resource('s3')

PUBLISHABLE = os.environ["PUBLISHABLE"]
S3_BUCKET_NAME = os.environ["S3_BUCKET_NAME"]
ENV = os.environ["ENVIRONMENT"] # "PRODUCTION" or "STAGING" or "DEV"


def compute_rank_zscore(df, field, is_smaller_better):
    
    # COMPUTE RANKING
    df[field + '_rank'] = df[field].rank(ascending = is_smaller_better)
    df.loc[pd.isnull(df[field + '_rank']),field + '_rank'] = df[field + '_rank'].max()+1

    # COMPUTE ZSCORE
    df[field + '_zscore'] = (df[field] - df[field].mean())/df[field].std(ddof=0)
    if (is_smaller_better):
        # Reverse z-score order if smaller is better for this factor
        df[field + '_zscore'] = df[field + '_zscore'] * (-1)
    # KEEP Z-SCORES BETWEEN -3/+3
    df[field + '_zscore_capped'] = df[field + '_zscore'].apply(cap_zscores)


def cap_zscores(x):
    if x>3:
        return 3
    elif x < -3:
        return -3
    return x


def lambda_handler(event, context):
    sleep_time = 0.1
    base_url = "https://sandbox.iexapis.com"
    if ENV.upper() == "PRODUCTION":
        # IEX api limit requests to 100 per second per IP measured in milliseconds, so no more than 1 request per 10 milliseconds. Sandbox Testing has a request limit of 10 requests per second measured in milliseconds. # https://iexcloud.io/docs/api/
        sleep_time = 0.001 # IEX Cloud call take around 100ms and allow burst. 1ms wait time should be enough
        base_url = "https://cloud.iexapis.com"

    #############################################################################################
    ## LIST OF SYMBOLS FROM IEX CLOUD -- cost of 100 units per call #https://iexcloud.io/docs/api/#symbols
    #############################################################################################
    url = base_url + "/stable/ref-data/symbols"
    r = requests.get(url + "?token=" + PUBLISHABLE)
    response_json = r.json()   # ~11k tickers returned
    df_universe = pd.DataFrame(response_json)

    #############################################################################################
    ## LIST OF TICKERS IN S&P TOTAL US INDEX
    #############################################################################################
    url = "https://www.ishares.com/us/products/239724/ishares-core-sp-total-us-stock-market-etf/1467271812596.ajax?fileType=csv&fileName=iShares-Core-SP-Total-US-Stock-Market-ETF_fund&dataType=fund"
    r = requests.get(url)
    if r.status_code != 200:
        logger.error(f'HTTP request returned code {r.status_code}')

    lines = [line.strip() for line in r.text.split('\n') if line.strip()]
    if len(lines) < 9:
        logger.error(f'Too few lines in file from {url}')
    csv_res = [line for line in csv.reader(lines)]
    etf_des = csv_res[:8]
    etf_holdings = csv_res[8:]
    df_etf_holdings = pd.DataFrame.from_records(etf_holdings[1:-1], columns=etf_holdings[0])
    if ENV.upper() != "PRODUCTION":
        df_etf_holdings = df_etf_holdings[:100]
    else:
        df_etf_holdings = df_etf_holdings[:500]
    ######################################################################################
    ## FILTER IEX UNIVERSE TO KEEP ONLY STOCKS FROM THE S&P TOTAL US INDEX
    ######################################################################################
    logger.info("FILTER IEX UNIVERSE TO KEEP ONLY STOCKS FROM THE S&P TOTAL US INDEX")
    df_universe_filtered = df_universe[df_universe["symbol"].isin(df_etf_holdings["Ticker"])]

    ######################################################################################
    ## COLLECT STOCKS STATISTICS - IEX CLOUD
    ######################################################################################
    logger.info("COLLECT STOCKS STATISTICS - IEX CLOUD")
    list_stocks = []
    i=1
    for index, row in df_universe_filtered.iterrows():
        logger.info("DOWNLOAD::IEXCLOUD::STATS_" + str(i) + "/" + str(len(df_universe_filtered)))
        url = base_url + "/stable/stock/{symbol}/advanced-stats"
        url = url.replace("{symbol}",row["symbol"])
        r = requests.get(url + "?token=" + PUBLISHABLE)
        if r.status_code==200:
            r_json = r.json()
            r_json["symbol"] = row["symbol"]
            list_stocks.append(r_json)
            i=i+1
            sleep(sleep_time)
    # Create DataFrame
    df_universe_filtered_stats = pd.DataFrame(list_stocks)

    ######################################################################################
    ### COLLECT STOCKS PRICES FOR VOLATILITY AND AVG VALUE TRADED CALCULATION - IEX CLOUD
    ######################################################################################
    logger.info("COLLECT STOCKS PRICES FOR VOLATILITY AND AVG VALUE TRADED CALCULATION - IEX CLOUD")
    df_all_stocks_prices = pd.DataFrame()
    i=1
    for index, row in df_universe_filtered.iterrows():
        logger.info("DOWNLOAD::IEXCLOUD::PRICES_" + str(i) + "/" + str(len(df_universe_filtered)))
        url = base_url + "/stable/stock/{symbol}/chart/3m?chartCloseOnly=true"
        url = url.replace("{symbol}",row["symbol"])
        r = requests.get(url + "&token=" +PUBLISHABLE)
        if r.status_code==200:
            r_json = r.json()
            df_stock_prices = pd.DataFrame(r_json)
            df_stock_prices["symbol"] = row["symbol"]
            df_all_stocks_prices = df_all_stocks_prices.append(df_stock_prices)
        i=i+1
        sleep(sleep_time)
    df_all_stocks_prices["value_traded"] = df_all_stocks_prices["volume"] * df_all_stocks_prices["close"]
    df_adv = df_all_stocks_prices.groupby("symbol")["value_traded"].mean().reset_index()
    df_vol = df_all_stocks_prices.groupby("symbol")["changePercent"].std().reset_index()
    df_vol["vol"] = df_vol["changePercent"] * math.sqrt(252)

    # Merge with main dataFrame
    df_universe_filtered_stats = df_universe_filtered_stats.merge(df_adv, on ="symbol", how="left") 
    df_universe_filtered_stats = df_universe_filtered_stats.merge(df_vol[["symbol","vol"]], on ="symbol", how="left") 

    ######################################################################################
    ### COMPUTE FACTOR VALUES
    ######################################################################################
    logger.info("COMPUTE FACTOR VALUES")
    df_factors = df_universe_filtered_stats[["symbol"]].copy()

    ##################################################################################
    # FACTOR VALUE 
    ##################################################################################
    # earnings_yield, inverse of P/E
    df_factors["VALUE_earnings_yield"] = 1 / df_universe_filtered_stats["peRatio"]
    # EBITDA/EV,	inverse of EV/EBITDA
    df_factors["VALUE_EBITDA/EV"] = df_universe_filtered_stats["EBITDA"] / df_universe_filtered_stats["enterpriseValue"] 
    # sales_yield,	inverse of Sales/Price
    df_factors["VALUE_sales_yield"] = 1 / df_universe_filtered_stats["priceToSales"]

    ##################################################################################
    # FACTOR MOMENTUM - OK
    ##################################################################################
    # Momentum - price return 12m lag 1m
    df_factors["MOM_12m-1mlag"] = (df_universe_filtered_stats["year1ChangePercent"] +1) / (df_universe_filtered_stats["month1ChangePercent"]+1) -1
    #Odf_factors["MOM_month1ChangePercent"] = df_universe_filtered_stats["month1ChangePercent"]

    ##################################################################################
    # FACTOR RISK
    ##################################################################################
    # Beta
    df_factors["RISK-beta"] = df_universe_filtered_stats["beta"]
    # Volatility 90days
    df_factors["RISK-vol90d"] = df_universe_filtered_stats["vol"]

    ##################################################################################
    # FACTOR SIZE 
    ##################################################################################
    # Sqrt of company market cap
    df_factors["SIZE-mktcap"] = df_universe_filtered_stats["marketcap"].apply(math.sqrt)
    # Average value traded (using daily close*volume as proxy of ADV)
    df_factors["SIZE-avgValueTraded"] = df_universe_filtered_stats["value_traded"]

    ##################################################################################
    # FACTOR QUALITY
    ##################################################################################
    # Debt to equity
    df_factors["QUALITY-debtToEquity"] = df_universe_filtered_stats["debtToEquity"]
    # Gross margin as calculated by gross profit divided by revenue.
    df_factors["QUALITY-GrossMargin"] = df_universe_filtered_stats["grossProfit"] / df_universe_filtered_stats["totalRevenue"]
    # return on equity would also be a nice to have

    ##################################################################################
    # COMPUTE RANKING AND Z-SCORES
    ##################################################################################
    logger.info("COMPUTE RANKING AND Z-SCORES")

    ## VALUE
    compute_rank_zscore(df_factors,"VALUE_earnings_yield", is_smaller_better= False)
    compute_rank_zscore(df_factors,"VALUE_EBITDA/EV", is_smaller_better= False)
    compute_rank_zscore(df_factors,"VALUE_sales_yield", is_smaller_better= False)
    # COMPUTE FINAL RANKING FOR THIS FACTOR
    df_factors["VALUE"] = 1/3 * df_factors["VALUE_earnings_yield_rank"] + 1/3 * df_factors["VALUE_EBITDA/EV_rank"] + 1/3 * df_factors["VALUE_sales_yield_rank"]

    ## MOMENTUM
    compute_rank_zscore(df_factors,"MOM_12m-1mlag", is_smaller_better= False)
    df_factors["MOMENTUM"] = df_factors["MOM_12m-1mlag_rank"]

    ## RISK
    compute_rank_zscore(df_factors,"RISK-beta", is_smaller_better= True)
    compute_rank_zscore(df_factors,"RISK-vol90d", is_smaller_better= True)
    df_factors["RISK"] = 1/2 * df_factors["RISK-beta_rank"] + 1/2 * df_factors["RISK-vol90d_rank"] 

    ## SIZE
    compute_rank_zscore(df_factors,"SIZE-mktcap", is_smaller_better= True)
    compute_rank_zscore(df_factors,"SIZE-avgValueTraded", is_smaller_better= True)
    df_factors["SIZE"] = 1/2 * df_factors["SIZE-mktcap_rank"] + 1/2 * df_factors["SIZE-avgValueTraded_rank"] 

    ## QUALITY
    compute_rank_zscore(df_factors,"QUALITY-debtToEquity", is_smaller_better= True)
    compute_rank_zscore(df_factors,"QUALITY-GrossMargin", is_smaller_better= False)
    df_factors["QUALITY"] = 1/2 * df_factors["QUALITY-debtToEquity_rank"] + 1/2 * df_factors["QUALITY-GrossMargin_rank"] 

    ##################################################################################
    # COMPUTE FINAL STOCK RANKING
    ##################################################################################
    df_factors["FINAL_SCORE"] = (df_factors["VALUE"] + df_factors["MOMENTUM"] + df_factors["RISK"] + df_factors["SIZE"] + df_factors["QUALITY"])/5

    # ##################################################################################
    # # COMPUTE HISTORICAL PERF OF BASKET (WITH FORWARD LOOKING BIAS)
    # ##################################################################################
    # logger.info("COMPUTE HISTORICAL PERF OF BASKET (WITH FORWARD LOOKING BIAS)")
    # # LOOK AT PERFORMANCE OF TOP 100 STOCKS AS RANKED BY OUR FACTOR INVESTING PROCESS
    # df_top_basket = df_factors.sort_values("FINAL_SCORE")[:100]
    # df_bot_basket = df_factors.sort_values("FINAL_SCORE")[-100:]
    # # MERGE TOP BASKETS WITH PAST PERFORMANCE.
    # df_top_perf = df_top_basket[["symbol"]].merge(df_all_stocks_prices[["symbol","date","changeOverTime"]], on = "symbol")
    # df_top_cumperf = df_top_perf.groupby("date")["changeOverTime"].mean().reset_index()
    # # MERGE BOTTOM BASKETS WITH PAST PERFORMANCE.
    # df_bot_basket = df_bot_basket[["symbol"]].merge(df_all_stocks_prices[["symbol","date","changeOverTime"]], on = "symbol")
    # df_bot_cumperf = df_bot_basket.groupby("date")["changeOverTime"].mean().reset_index()
    # # PLOT PERFORMANCE OF TOP/BOTTOM BASKETS OF 100 STOCKS
    # df_perf = df_top_cumperf.merge(df_bot_cumperf, on="date", suffixes=('_top', '_bottom'))
    # df_perf.plot(x ='date', y=['changeOverTime_top','changeOverTime_bottom'], kind = 'line')

    ##################################################################################
    # SAVE TO S3 BUCKET
    ##################################################################################
    logger.info("SAVE TO S3 BUCKET")
    # LOOK AT PERFORMANCE OF TOP 100 STOCKS AS RANKED BY OUR FACTOR INVESTING PROCESS
    df_top_basket = df_factors.sort_values("FINAL_SCORE", ascending=False)[:100]
    df_bot_basket = df_factors.sort_values("FINAL_SCORE")[:100]
    
    df_top_basket.to_csv("/tmp/" + ENV + "df_top_basket.csv")
    s3_resource.meta.client.upload_file("/tmp/" + ENV + "df_top_basket.csv", S3_BUCKET_NAME, ENV + "-" + datetime.datetime.now().strftime("%Y%m%d") + "-" + "df_top_basket.csv")
    df_bot_basket.to_csv("/tmp/" + ENV + "df_bot_basket.csv")
    s3_resource.meta.client.upload_file("/tmp/" + ENV + "df_bot_basket.csv", S3_BUCKET_NAME, ENV + "-" + datetime.datetime.now().strftime("%Y%m%d") + "-" + "df_bot_basket.csv")
    df_factors.to_csv("/tmp/" + ENV + "df_factors.csv")
    s3_resource.meta.client.upload_file("/tmp/" + ENV + "df_factors.csv", S3_BUCKET_NAME, ENV + "-" + datetime.datetime.now().strftime("%Y%m%d") + "-" + "df_factors.csv")


    body = {
        "message": "Successfully computed factors - Saved in bucket :" + str(S3_BUCKET_NAME),
    }
    response = {"statusCode": 200, "body": json.dumps(body)}

    return response

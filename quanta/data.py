#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 19:10:19 2017

@author: rhdzmota
"""

# %% Imports 

import numpy as np
import pandas as pd
import datetime as dt
import requests
import matplotlib.pyplot as plt

# %% variables 
#global thisIsInsane

# %% wake up heroku instance
def wakeUpPlease():
    """
    Please wakeup!
    """
    
    try:
        r = requests.get("https://mxquants-datarepo.herokuapp.com")
        return True 
    except:
        return False 

# %% download prices func 
def downloadPrices(stock_name=None,columns=["Timestamp","AdjPrices"]): 
    
    # check for type 
    if type(stock_name) != str :
        print("Error: stockname (str) must be provided." )
        return None
    
    
    # correct columnnames 
    def correctColumnNames(cols):
        correct = {"Timestamp":"date","AdjPrices":"unadjclose"}
        def try2correct(x):
            try: 
                return correct[x]
            except:
                return x
        return [try2correct(c) for c in cols]
    
    # url for request 
    url = """https://mxquants-datarepo.herokuapp.com/DefaultDownload"""
    parameters = {"columns":str(correctColumnNames(columns)),"stock_name":stock_name.upper(),"pwd":"mxquants-rules"}
    
    # make request 
    r = requests.get(url,params=parameters).json()
    
    try:
        flag = not r["error"]
    except:
        flag = True
    
    if not flag:
        print("Error: data coudn't be downloaded. Check your internet connection or variables stock_name, columns.")
        return None 
    data = r["data"]
    
    # get date as string from timestamp 
    def timestamp2Date(x):
        return dt.datetime.fromtimestamp(int(x)).strftime("%d-%m-%Y")
    
    data["date"] = list(map(timestamp2Date,data["date"]))
    
    # create dataframe 
    df = pd.DataFrame(data)
    df.columns = columns 
    
    return df 
    
# %% Calculate returns

def getReturns(price_vector,_type="log"):
    price_vector = np.asarray(price_vector)
    if _type != "log":
        return price_vector[1:]/price_vector[:-1] - 1
    return np.log(price_vector[1:]/price_vector[:-1] )
    
# %% Banxico Series

def getAvailableBanxicoSeries():
    #/BanxicoSeries?pwd=mxquants-rules&purpose=available_series
    
    # set url and parameters
    url = "https://mxquants-datarepo.herokuapp.com/BanxicoSeries"
    parameters = {"pwd":"mxquants-rules",
                  "purpose":"available_series"}
    
    # make request 
    r = requests.get(url,params=parameters).json()
    return r.get("series_names") 

def getBanxicoSeries(var,_type="df"):
    # var tiie28
    # /BanxicoSeries?pwd=mxquants-rules&purpose=download_data&variable_name=tiie28
    
    # set url and parameters
    url = "https://mxquants-datarepo.herokuapp.com/BanxicoSeries"
    parameters = {"pwd":"mxquants-rules",
                  "purpose":"download_data",
                  "variable_name":str(var)}
    
    # make request 
    try:
        r = requests.get(url,params=parameters).json()
    except:
        print("\nOkay, don't freak out. Something went wrong... May be your internet connection.")
        return None
    
    # exec something insane. 
    exec(r.get("fun"),globals())
    
    res = thisIsInsane(var,r.get("data"))
    if "df" not in _type:
        return eval(res)
    
    try:
        res = eval(res)
        return pd.DataFrame(res["data"])
    except:
        return res
    
    
# %% Main function 

def main():
    try:
        wakeUpPlease()
        is_awake = True
    except:
        is_awake = False
        
    return is_awake
    
# %% 

if __name__ == '__main__':
    main()

# %% 


# %% 
    
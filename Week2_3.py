
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from pymongo import MongoClient
import json

import DEF_MFS_MVP_Storage as mDB


class Ticker:

    def get_tickername(self):
        ticker_number = int(input("Enter number of tickers you want"))
        tickers = []
        for i in range(ticker_number):
            ticker = input("Enter ticker symbol:")
            tickers.append(ticker)
        tickers = " ".join(tickers)
        return tickers


class yfinance:

    # constructor to initialize instance variables
    def __init__(self, ticker, start_date, end_date):
        # ticker: stock ticker symbol
        # star_date : start date from when the stock prices is to be extracted
        # end_date : final date upto which the stock prices is to be extracted
        self.ticker = ticker
        self.start = start_date
        self.end = end_date

    # instance method to extract stock prices of a any one ticker
    def get_stockmarketdata_indv(self):
        # for individual ticker we use the yf.ticker module
        # yf.Ticker to create a ticker object for a particular ticker
        stock = yf.Ticker(self.ticker)
        # print(stock.info)
        # .history to get historical market data
        stock_df = stock.history(period='1d', start=self.start, end=self.end, actions=False)
        # period ='1d' means we want to extract stock market data once every day between the specified dates
        return stock_df

    # instance method to extract stock prices for all specified tickers
    def get_stockmarketdata_all(self):
        # To download the historical data for multiple tickers at once you can use the download module.
        # converting to string
        merged_stock_df = yf.download(self.ticker, period='1d', start=self.start, end=self.end, group_by='tickers')
        return merged_stock_df


class output:
    # class for displaying output and saving files
    def __init__(self, value, output_name):
        self.value = value
        self.name = output_name

    def output_print(self):
        # this function will print values, list, arrays and others
        print(self.value)

    def output_display(self):
        # this function will display pandas dataframe as a table
        print(self.value)

    def output_savecsv(self):
        # this functuion will save output as csv
        self.value.to_csv(self.name + ".csv")

    def output_plot(self):
        # this function will make a line plot of all the stock market data
        open_close = self.value
        open_close.reset_index(inplace=True)
        open_close.plot(x="Date", y=["Open", "Close"], figsize=(12, 12))
        plt.title("Line plot for open and closed stock market prices for : " + self.name, fontsize=20)
        plt.xlabel("Date", fontsize=15)
        plt.ylabel("Stock market Data", fontsize=15)
        plt.show()

#initialize start and end date

start_date = "2021-01-01"
end_date = "2022-12-31"

#define a class object
tesla_Data = yfinance("TSLA", start_date, end_date)

#extract stock data by calling the class's instance method
Tesla=tesla_Data.get_stockmarketdata_indv()

type(Tesla)

#show stock data in tabular form by calling class's instance method
Tesla_output = output(Tesla,"Tesla")
Tesla_output.output_display()

#show lineplot for open and close values
Tesla_output.output_plot()

#save to csv
Tesla_output.output_savecsv()


#define class object for Ford
ford_Data = yfinance("F", start_date, end_date)


#extract stock data
Ford=ford_Data.get_stockmarketdata_indv()



#show stock data in tabular form by calling class's instance method
Ford_output=output(Ford,"Ford")
Ford_output.output_display()



#show lineplot for open and close values
Ford_output.output_plot()

#save stock prices to csv
Ford_output.output_savecsv()



#input tickers using the Ticker class
Tick=Ticker()
tickers=Tick.get_tickername()
print(tickers)

#define a class object
tesla_ford_Data = yfinance(tickers, start_date, end_date)

#extract stock data
Tesla_Ford=tesla_ford_Data.get_stockmarketdata_all()



#show stock data in tabularTesla_Ford form by calling class's instance method
Tsla_ford_output=output(Tesla_Ford,"Tesla_Ford")
Tsla_ford_output.output_display()

Tsla_ford_output.output_savecsv()













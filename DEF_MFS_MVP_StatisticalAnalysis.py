#### This module takes data and calculates its statistical characteristics and returns them. Also, it looks for any missing values, outliers and errors in the data.

## Libraries

#to work with dataframe
import pandas as pd
#to perform mathematical and statistical calculations
import numpy as np
#for visualization to create plots and graphs


## Define class to compute missing values, outliers and any errors

class Find_Anomolies:
    # class constructor
    def __init__(self, dataframe):
        self.df = dataframe

    # we define a class called check_null to find any missing values in ourr stock market dataframe
    # we use isnull()that returns true or false values if there is Nan in the table
    def check_null(self):
        is_null = self.df.isnull().sum()
        print(is_null)
        is_null_dict = is_null.to_dict()
        for i in is_null_dict:
            if is_null_dict[i] == 0:
                print(f"There are no missing values in {i} column \n")

    def check_outliers(self, column_name):
        # we define a class instance method to detect any outliers in our stock market data
        # well find 1st and 3rd quantile, calculate IQR and find upper and lower whisker
        # any datapoint greater than upper whisker and any data lesser than the lower quantile will be our outlier
        # print(type(self.df[column_name]))
        # calculate Q1
        Q1 = np.percentile(self.df[column_name], 25, method='midpoint')
        # calculate Q3
        Q3 = np.percentile(self.df[column_name], 75, method='midpoint')
        # calculate interquartile range
        IQR = Q3 - Q1
        # calculate upper  whisker
        upper = (Q3 + 1.5 * IQR)
        # calculate lower whisker
        lower = (Q1 - 1.5 * IQR)
        # any data points above and lower the upper and lower whickers respectively are considered outliers
        # upper datapoints and lower datapoints
        upper_data = self.df[column_name] >= upper
        lower_data = self.df[column_name] <= lower
        outlier_data = pd.concat([self.df[upper_data], self.df[lower_data]], axis=0)
        # outliers are
        print(column_name)
        print(f"Upper Limit : {upper}       Lower Limit : {lower}")
        print(outlier_data[["Date", column_name]])
        print(f"There are {outlier_data.shape[0]} outliers for {column_name} target feature")
        print("\n\n")


## Define class to compute statistical characteristics of our data

# this class will calculate all the statistical characteristic of our stock market price data
class Statistical_character:
    # class constructor
    def __init__(self, dataframe, column_name):
        self.df = dataframe
        self.col = column_name
        self.stats = []

    # method to show all the descriptive statistic characters of our stock market data
    def calculate_statistic(self):
        # print(type(self.col))

        print(self.col)
        # print(type(self.df[self.col]))
        print("Descriptive Statistics \n")
        print(self.df[self.col].describe())

    # method to calculate minimum values from our data
    def calculate_min(self):
        # print(self.df[self.col])
        min_value = self.df[self.col].min()
        # print(min_value)
        print(f"Minimum of {self.col} stock price is : {min_value}")
        self.stats.append(("min", min_value))

    # method to calculate maximum values from our data
    def calculate_max(self):
        max_value = self.df[self.col].max()
        print(f"Maximum of {self.col} stock price is : {max_value}")
        self.stats.append(("max", max_value))

    # method to calculate range values from our data
    def calculate_range(self):
        range_val = self.df[self.col].max() - self.df[self.col].min()
        print(f"Range of {self.col} stock price is : {range_val}")
        self.stats.append(("range", range_val))

    # method to calculate median from our data(middle value)
    def calculate_median(self):
        median_val = self.df[self.col].median()
        print(f"Median of {self.col} stock price is : {median_val}")
        self.stats.append(("median", median_val))

    # method to calculate mean/average from our data
    def calculate_mean(self):
        mean_val = self.df[self.col].mean()
        print(f"Mean of {self.col} stock price is : {mean_val}")
        self.stats.append(("mean", mean_val))

    # method to calculate variance from our data
    def calculate_variance(self):
        var_val = self.df[self.col].var()
        print(f"Variance of {self.col} stock price is : {var_val}")
        self.stats.append(("variance", var_val))

    # method to calculate standard deviation from our data
    def calculate_stddev(self):
        std_val = self.df[self.col].std()
        print(f"Standard Deviation of {self.col} stock price is : {std_val} \n")
        self.stats.append(("standard_deviation", std_val))

    # method to display all the statistical characters in tabular format
    def stats_table(self):
        print("Statistical Characteristic in tabular form: \n")
        stats_intable = pd.DataFrame(self.stats, columns=["Statistical_Charecteristic", "Values"])
        print(stats_intable)
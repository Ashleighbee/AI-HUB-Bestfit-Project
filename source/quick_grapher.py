import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    df = pd.read_csv('../data/housing_temp.csv')
    price = df.daily_price / df.accommodates
    feature_name = 'sub_dist_1'
    feature = (df[feature_name])

    plt.xlabel(feature_name)
    plt.ylabel('Price ($)')
    plt.scatter(feature, price, alpha=0.1)
    plt.show()

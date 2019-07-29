import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    flist = pd.read_csv('../data/housing_clean_b.csv')
    price = flist.daily_price
    feature_name = 'sub_dist_1'
    feature = (flist[feature_name])

    plt.xlabel(feature_name)
    plt.ylabel('Price ($)')
    plt.scatter(feature, price, alpha=0.1)
    plt.show()

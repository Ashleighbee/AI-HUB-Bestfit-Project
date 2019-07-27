import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

flist = pd.read_csv('data/housing_all.csv')
price = flist.daily_price
feature = flist.Madison_Square_Garden

plt.xlabel('Nearest')
plt.ylabel('Price ($)')
plt.scatter(feature, price, alpha=0.1)
plt.show()

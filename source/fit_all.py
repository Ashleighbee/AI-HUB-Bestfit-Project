import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt


df = pd.read_csv('../data/housing_clean.csv')
df_b = pd.read_csv('../data/housing_price_balanced.csv')
features = [df.subway, df.bus_stop, df.park, df.scenery, df.accommodates, df.bathroom, df.bedroom, df.beds, df.guests,
            df.num_of_review, df.review_score, df.Entire_home, df.host_response_rate, df.superhost, df.crime_rate]
X = pd.concat(features, axis=1).dropna().astype(dtype='float64', copy=False)
y = df.daily_price.dropna()

X_b = pd.concat(features, axis=1).dropna().astype(dtype='float64', copy=False)
y_b = df.daily_price.dropna()

# X_sc = scale(X)
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2)
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X.values, y.values, test_size=0.2)

reg_line = LinearRegression()
reg_ri = RidgeCV(cv=5)
reg_tree = DecisionTreeRegressor(max_depth=10)
reg_bagging = BaggingRegressor()
reg_Forest = RandomForestRegressor(n_estimators=100, max_depth=10)
reg_boosting = GradientBoostingRegressor(n_estimators=100)
reg_ada_boost = AdaBoostRegressor(n_estimators=100)


def linear_all():
    importance = []
    for i in range(len(features)):
        x_try = X_train_b[:, i]
        reg_ri.fit(x_try.reshape(-1, 1), y_train_b)

        importance.append([X.columns.values[i],
                           reg_ri.score(x_try.reshape(-1, 1), y_train_b),
                           reg_ri.score(X_test[:, i].reshape(-1, 1), y_test)])
    importance.sort(key=lambda x: x[1])
    importance.reverse()
    for each in importance:
        print(str(each[0]) + ':\t', each[1], each[2])


def forest_test():
    reg_Forest.fit(X_train_b, y_train_b)
    print('Accuracy:\t', reg_Forest.score(X_test, y_test))
    print('\nImportance for each:')
    importance = []
    for i in range(0, len(X.columns.values)):
        importance.append([X.columns.values[i], reg_Forest.feature_importances_[i]])
    importance.sort(key=lambda x: x[1])
    importance.reverse()
    for each in importance:
        print(each[0] + ':\t', each[1])


if __name__ == '__main__':
    linear_all()
    # forest_test()

    # reg_all = [reg_line, reg_tree, reg_bagging, reg_Forest, reg_boosting, reg_ada_boost]
    # for reg in reg_all:
    #     reg.fit(X_train, y_train)
    #     print(reg.score(X_test, y_test))

    # print(X.columns.values)
    # plt.hist(X.values[:, 3])
    # plt.show()

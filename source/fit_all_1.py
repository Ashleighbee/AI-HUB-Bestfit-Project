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
import joblib
import matplotlib.pyplot as plt


def linear_all_factors(features):
    reg_line = LinearRegression()
    print("\nLinear Regression:\n")
    importance = []
    for i in range(len(features)):
        x_try = X_train[:, i]
        reg_line.fit(x_try.reshape(-1, 1), y_train)

        importance.append([X.columns.values[i],
                           reg_line.score(X_test[:, i].reshape(-1, 1), y_test)])
    importance.sort(key=lambda x: x[1])
    importance.reverse()
    for each in importance:
        print(str(each[0]) + ':\t', each[1])


def visualization(r):
    pred = r.predict(X_test)
    errors = abs(y_test - pred)
    plt.hist(errors)
    plt.show()


def random_forest(X, X_train, X_test, y_train, y_test, store=True):
    reg_Forest = RandomForestRegressor(n_estimators=100, min_samples_split=2, max_depth=10)
    print("\nRandom Forest:\n")
    reg_Forest.fit(X_train, y_train)
    print('Accuracy:\t', reg_Forest.score(X_test, y_test))
    print('\nImportance for each:')
    importance = []
    for i in range(0, len(X.columns.values)):
        importance.append([X.columns.values[i], reg_Forest.feature_importances_[i]])
    importance.sort(key=lambda x: x[1])
    importance.reverse()
    for each in importance:
        print(each[0] + ':\t', each[1])

    if store:
        joblib.dump(reg_Forest, 'RF')


def gradient_boosting(X, X_train, X_test, y_train, y_test, store=True):
    reg_gradient = GradientBoostingRegressor(n_estimators=100, min_samples_split=2, max_depth=10)
    print("\nGradient Decent:\n")
    reg_gradient.fit(X_train, y_train)
    print('Accuracy:\t', reg_gradient.score(X_test, y_test))
    print('\nImportance for each:')
    importance = []
    for i in range(0, len(X.columns.values)):
        importance.append([X.columns.values[i], reg_gradient.feature_importances_[i]])
    importance.sort(key=lambda x: x[1])
    importance.reverse()
    for each in importance:
        print(each[0] + ':\t', each[1])

    if store:
        joblib.dump(reg_gradient, 'GD')


def generate_sets(filename, split=True):
    df = pd.read_csv(filename)
    # df_b = pd.read_csv('../data/housing_price_balanced.csv')
    features = [df.house_la, df.house_ln, df.subway, df.bus_stop, df.park, df.scenery, df.accommodates, df.bathroom,
                df.bedroom, df.beds, df.guests, df.Entire_home, df.response_time_num, df.superhost,
                df.host_response_rate,
                df.crime_rate,
                df.Madison_Square_Garden, df.Flatiron_Building, df.madame_tussauds_new_york, df.Empire_state_Building,
                df.intrepid_sea_air, df.Washington_Square_Park, df.New_york_Public_Library, df.Times_Square,
                df.New_York_University, df.Grand_Centreal_Terminal, df.Top_of_the_Rock, df.St_Patrick_Cathedral,
                df.Museum_of_Modern_Art, df.Manhattan_Skyline, df.United_Nations_Headquarters, df.One_world_trade_cente,
                # df.Central_Park, df.Van_Cortlandt, df.Flushing_Meadows, df.Prospect_Park,
                # df.Bronx_Park, df.Pelham_Bay_Park, df.Floyd_Bennet_Field, df.Jamaica_Bay, df.Jacob_Riis_Park,
                # df.Fort_Tilden, df.Greenbelt, df.The_Metropolitan_Museum_of_Art, df.statue_of_liberty,
                # df.American_Museum_of_Natual_History, df.Fifth_Avenue, df.Brooklyn_Bridge, df.Wall_Street, df.Broadway,
                # df.China_Town, df.West_Point_Academy, df.Columbia_University, df.National_September_11_Memorial_Museum,
                # df.SOHO, df.High_Line_Park,
                df.sub_dist_1, df.sub_dist_2, df.sub_dist_3, df.bus_dist_1, df.bus_dist_2, df.bus_dist_3]

    X = pd.concat(features, axis=1).astype(dtype='float64', copy=False)
    y = df.daily_price

    if split:
        X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.3)
        return X_train, X_test, y_train, y_test
    else:
        return X, y


if __name__ == '__main__':
    X, y = generate_sets('../data/housing_clean.csv', split=False)
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.3)
    X_train = scale(X_train)
    X_test = scale(X_test)
    gradient_boosting(X, X_train, X_test, y_train, y_test)

    # reg_line = LinearRegression()
    # reg_ri = RidgeCV(cv=5)
    # reg_tree = DecisionTreeRegressor(max_depth=10)
    # reg_bagging = BaggingRegressor()
    # reg_Forest = RandomForestRegressor(n_estimators=150, min_samples_split=2, max_depth=10)
    # reg_boosting = GradientBoostingRegressor(n_estimators=100)
    # reg_ada_boost = AdaBoostRegressor(n_estimators=100)

    # linear_all_factors()
    # reg = joblib.load('RF.m')
    # visualization(reg_Forest)

    # reg_all = [reg_line, reg_tree, reg_bagging, reg_Forest, reg_boosting, reg_ada_boost]
    # for reg in reg_all:
    #     reg.fit(X_train, y_train)
    #     print(reg.score(X_test, y_test))

    # X_dropped, y_dropped, reg_model = error_clean(X, y, modelfile='RF')
    # X_train, X_test, y_train, y_test = train_test_split(X_dropped, y_dropped, test_size=0.3)
    #
    # random_forest(X, X_train, X_test, y_train, y_test, store=False)

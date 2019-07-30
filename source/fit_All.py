import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import scale
import joblib
import matplotlib.pyplot as plt


def visualization(reg_model, _x_test, _y_test, _features):
    pred = reg_model.predict(_x_test)
    errors = abs(_y_test - pred)
    plt.hist(errors)
    plt.show()

def linear_try_each_factors(_features, _x_train, _x_test, _y_train, _y_test):
    reg_line = LinearRegression()
    print("\nLinear Regression:\n")
    importance = []
    for i in range(len(_features)):
        x_try = _x_train[:, i]
        reg_line.fit(x_try.reshape(-1, 1), _y_train)

        importance.append([_features[i],
                           reg_line.score(_x_test[:, i].reshape(-1, 1), _y_test)])
    importance.sort(key=lambda x: x[1])
    importance.reverse()
    for each in importance:
        print(str(each[0]) + ':\t', each[1])

def try_all_models(_x_train, _x_test, _y_train, _y_test):
    reg_line = LinearRegression()
    reg_tree = DecisionTreeRegressor()
    reg_bagging = BaggingRegressor()
    reg_Forest = RandomForestRegressor()
    reg_boosting = GradientBoostingRegressor()

    reg_name = ['Linear', 'Tree', 'Bagging', 'Forest', 'Boosting']
    reg_all = [reg_line, reg_tree, reg_bagging, reg_Forest, reg_boosting]
    for reg, name in zip(reg_all, reg_name):
        print(name, ': ')
        reg.fit(_x_train, _y_train)
        print('Train score: ', reg.score(_x_train, _y_train))
        print('Test score: ', reg.score(_x_test, _y_test))

def random_forest(_features, _x_train, _x_test, _y_train, _y_test, store=True, load=False):
    print("\nRandom Forest:\n")

    if load:
        reg_Forest = joblib.load('RF')
    else:
        reg_Forest = RandomForestRegressor(n_estimators=150, max_depth=8, n_jobs=2)
        reg_Forest.fit(_x_train, _y_train)
        if store:
            joblib.dump(reg_Forest, 'RF')

    print('Training Accuracy:\t', reg_Forest.score(_x_train, _y_train))
    print('Testing Accuracy:\t', reg_Forest.score(_x_test, _y_test))
    print('\nImportance for each:')
    importance = []
    for i in range(0, len(_features)):
        importance.append([_features[i], reg_Forest.feature_importances_[i]])
    importance.sort(key=lambda x: x[1], reverse=True)
    for each in importance:
        print(each[0] + ':\t', each[1])
    return reg_Forest

def gradient_boosting(_features, _x_train, _x_test, _y_train, _y_test, store=True, load=False):
    print("\nGradient Boosting:\n")

    if load:
        reg_gradient = joblib.load('GB')
    else:
        reg_gradient = GradientBoostingRegressor(n_estimators=200, min_samples_split=7, max_depth=5)
        reg_gradient.fit(_x_train, _y_train)
        if store:
            joblib.dump(reg_gradient, 'GB')
    print('Training Accuracy:\t', reg_gradient.score(_x_train, _y_train))
    print('Testing Accuracy:\t', reg_gradient.score(_x_test, _y_test))
    print('\nImportance for each:')
    importance = []
    for i in range(0, len(_features)):
        importance.append([_features[i], reg_gradient.feature_importances_[i]])
    importance.sort(key=lambda x: x[1], reverse=True)
    for each in importance:
        print(each[0] + ':\t', each[1])
    return reg_gradient

def cv_for_hp(reg):
    param_distributions = {
        'n_estimators': [50, 80, 100, 150, 180, 200],
        'max_depth': [5, 8, 10, 12, 14, 16, 18],
        'min_samples_split': [2, 5, 7],
        # 'oob_score': [True],
    }
    searcher = RandomizedSearchCV(reg, param_distributions=param_distributions, n_iter=50, n_jobs=-1, cv=5, verbose=3)
    searcher.fit(X_train, y_train)
    print('\n best_R2: \n:', searcher.best_score_)
    print('best parameters:\t', searcher.best_params_)

def generate_sets(filename):
    df = pd.read_csv(filename)

    df['accommodates_s'] = df['accommodates'] ** 2
    df['scenery_s'] = df['scenery'] ** 2
    df['bedroom_s'] = df['bedroom'] ** 2
    df['beds_s'] = df['beds'] ** 2
    # df['subway_s'] = df['subway'] ** 2
    df['accom_bedroom'] = df.accommodates * df.bedroom
    df['bedroom_for_each'] = df.accommodates / df.bedroom
    df['beds_for_each'] = df.accommodates / df.beds

    df['park_scenery'] = df.park * df.scenery

    _features = [df.Entire_home, df.accommodates, df.scenery, df.house_ln,
                 df.Madison_Square_Garden,
                 df.Flatiron_Building, df.madame_tussauds_new_york, df.Empire_state_Building,
                 # df.intrepid_sea_air, df.Washington_Square_Park, df.New_york_Public_Library, df.Times_Square,
                 # df.New_York_University, df.Grand_Centreal_Terminal, df.Top_of_the_Rock, df.St_Patrick_Cathedral,
                 # df.Museum_of_Modern_Art, df.Manhattan_Skyline, df.United_Nations_Headquarters,

                 df.bathroom, df.response_time_num, df.host_response_rate, df.crime_rate,
                 df.guests, df.park, df.bedroom, df.beds, df.house_la, df.subway,
                 df.sub_dist_1, df.sub_dist_2, df.sub_dist_3, df.bus_stop,
                 df.accom_bedroom,

                 # df.One_world_trade_cente, df.Central_Park, df.Van_Cortlandt, df.Flushing_Meadows, df.Prospect_Park,
                 # df.Bronx_Park, df.Pelham_Bay_Park, df.Floyd_Bennet_Field, df.Jamaica_Bay, df.Jacob_Riis_Park,
                 # df.Fort_Tilden, df.Greenbelt, df.The_Metropolitan_Museum_of_Art, df.statue_of_liberty,
                 # df.American_Museum_of_Natual_History, df.Fifth_Avenue, df.Brooklyn_Bridge, df.Wall_Street,
                 # df.Broadway, df.China_Town, df.West_Point_Academy, df.Columbia_University,
                 # df.National_September_11_Memorial_Museum, df.SOHO, df.High_Line_Park,


                 # df.subway_s, df.accommodates_s, df.scenery_s,
                 # df.beds_s, df.beds_for_each, df.bedroom_for_each, df.bedroom_s, df.park_scenery,
                 ]

    _x = pd.concat(_features, axis=1).astype(dtype='float64', copy=False)
    features_name = _x.columns.values
    _y = df.daily_price

    _x_train, _x_test, _y_train, _y_test = train_test_split(_x.values, _y.values, test_size=0.3)
    return scale(_x_train), scale(_x_test), _y_train, _y_test, features_name


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, features = generate_sets('../data/housing_all_clean_.csv')

    linear_try_each_factors(features, X_train, X_test, y_train, y_test)
    reg1 = gradient_boosting(features, X_train, X_test, y_train, y_test)
    reg2 = random_forest(features, X_train, X_test, y_train, y_test)

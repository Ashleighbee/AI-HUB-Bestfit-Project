
import sklearn
from keras import models
from keras import layers
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def setup_nn(n1, n2, n3, dim):
    model = models.Sequential()  # create sequential multi-layer perceptron
    model.add(layers.Dense(n1, input_dim=dim, kernel_initializer='normal', activation='relu'))
    model.add(layers.Dense(n2, kernel_initializer='normal', activation='relu'))
    model.add(layers.Dense(n3, kernel_initializer='normal', activation='relu'))
    model.add(layers.Dense(1, kernel_initializer='normal', activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='Adam')
    return model


df = pd.read_csv('../data/housing_clean.csv')
features = [df.house_la, df.house_ln, df.subway, df.bus_stop, df.accommodates, df.bathroom,
            df.bedroom, df.beds, df.guests, df.Entire_home, df.response_time_num, df.host_response_rate,
            df.superhost, df.crime_rate,df.Central_Park, df.Van_Cortlandt, df.Flushing_Meadows, df.Prospect_Park,
            df.Washington_Square_Park,df.Bronx_Park, df.Pelham_Bay_Park, df.Floyd_Bennet_Field, df.Jamaica_Bay,
            df.Jacob_Riis_Park, df.Fort_Tilden, df.Greenbelt, df.The_Metropolitan_Museum_of_Art, df.statue_of_liberty,
            df.Empire_state_Building, df.American_Museum_of_Natual_History, df.Museum_of_Modern_Art, df.Times_Square,
            df.One_world_trade_cente, df.Grand_Centreal_Terminal, df.intrepid_sea_air, df.Fifth_Avenue,
            df.Brooklyn_Bridge, df.Wall_Street, df.Flatiron_Building, df.New_York_University, df.Broadway,
            df.China_Town, df.New_york_Public_Library, df.West_Point_Academy, df.Columbia_University,
            df.Madison_Square_Garden, df.madame_tussauds_new_york, df.National_September_11_Memorial_Museum,
            df.Manhattan_Skyline, df.United_Nations_Headquarters, df.SOHO, df.St_Patrick_Cathedral,
            df.High_Line_Park, df.Top_of_the_Rock, df.daily_price]
dataset = pd.concat(features, axis=1)
dataset = dataset.dropna().astype(dtype='float64', copy=False)

X = pd.concat([df.house_la, df.house_ln, df.subway, df.bus_stop, df.accommodates, df.bathroom,
            df.bedroom, df.beds, df.guests, df.Entire_home, df.response_time_num, df.host_response_rate,
            df.superhost, df.crime_rate,df.Central_Park, df.Van_Cortlandt, df.Flushing_Meadows, df.Prospect_Park,
            df.Washington_Square_Park,df.Bronx_Park, df.Pelham_Bay_Park, df.Floyd_Bennet_Field, df.Jamaica_Bay,
            df.Jacob_Riis_Park, df.Fort_Tilden, df.Greenbelt, df.The_Metropolitan_Museum_of_Art, df.statue_of_liberty,
            df.Empire_state_Building, df.American_Museum_of_Natual_History, df.Museum_of_Modern_Art, df.Times_Square,
            df.One_world_trade_cente, df.Grand_Centreal_Terminal, df.intrepid_sea_air, df.Fifth_Avenue,
            df.Brooklyn_Bridge, df.Wall_Street, df.Flatiron_Building, df.New_York_University, df.Broadway,
            df.China_Town, df.New_york_Public_Library, df.West_Point_Academy, df.Columbia_University,
            df.Madison_Square_Garden, df.madame_tussauds_new_york, df.National_September_11_Memorial_Museum,
            df.Manhattan_Skyline, df.United_Nations_Headquarters, df.SOHO, df.St_Patrick_Cathedral,
            df.High_Line_Park, df.Top_of_the_Rock], axis=1)
y = dataset.daily_price

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2)
X_sc = StandardScaler()
X_sc.fit(X_train)
X_train = X_sc.transform(X_train)
X_test = X_sc.transform(X_test)


def training(n1, n2, n3):
    model = setup_nn(n1, n2, n3, X_train.shape[1])
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test), verbose=0)
    y_predict = model.predict(X_test)
    r2 = r2_score(y_test, y_predict)

    print(n1, n2, n3, r2)
    return r2


def trying():
    m = 0
    nn = []
    for n1 in range(40, 130, 5):
        for n2 in range(30, 120, 5):
            for n3 in range(20, 80, 5):
                mm = training(n1, n2, n3)
                if mm > m:
                    m = mm
                    nn = [n1, n2, n3]
    print(m, nn)


if __name__ == '__main__':
    trying()

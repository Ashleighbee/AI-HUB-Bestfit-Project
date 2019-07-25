
import sklearn
from keras import models
from keras import layers
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def setup_nn(dim):
    model = models.Sequential()  # create sequential multi-layer perceptron
    model.add(layers.Dense(85, input_dim=dim, kernel_initializer='normal', activation='relu'))
    model.add(layers.Dense(65, kernel_initializer='normal', activation='relu'))
    model.add(layers.Dense(40, kernel_initializer='normal', activation='relu'))
    model.add(layers.Dense(1, kernel_initializer='normal', activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='Adam')
    return model

df = pd.read_csv('../data/housing_price_balanced.csv')
features = [df.house_ln, df.house_la, df.subway, df.bus_stop, df.accommodates, df.bathroom, df.bedroom, df.beds,
            df.guests, df.num_of_review, df.review_score, df.Entire_home, df.crime_rate, df.park, df.scenery,
            df.host_response_rate, df.superhost, df.daily_price]
dataset = pd.concat(features, axis=1)
dataset = dataset.dropna().astype(dtype='float64', copy=False)

X = pd.concat([df.house_ln, df.house_la, df.subway, df.bus_stop, df.accommodates, df.bathroom, df.bedroom, df.beds,
            df.guests, df.num_of_review, df.review_score, df.Entire_home, df.crime_rate, df.park, df.scenery,
            df.host_response_rate, df.superhost], axis=1)
y = dataset.daily_price

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2)
X_sc = StandardScaler()
X_sc.fit(X_train)
X_train = X_sc.transform(X_train)
X_test = X_sc.transform(X_test)


model = setup_nn(X_train.shape[1])
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
y_predict = model.predict(X_test)
r2 = r2_score(y_test, y_predict)

print(r2)
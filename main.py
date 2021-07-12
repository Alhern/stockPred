import numpy as np

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.models import Sequential
from keras.models import model_from_json

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from key import *

from alpha_vantage.timeseries import TimeSeries


############
## PARAMS ##
############

STOCK_TO_PREDICT = 'ibm'

PREDICT_LEN = 1  # prédis si la tendance va rester à la hausse ou à la baisse pendant N jours
DAYS_NUM = 20
NEURONS = 100

# TRAINING CONFIG:
EPOCHS = 10
BATCH_SIZE_TRAIN = 512
VALIDATION_SPLIT = 0.05


def getData(symbol):
    ts = TimeSeries(key=ALPHA_KEY)
    data, meta_data = ts.get_daily_adjusted(symbol=symbol, outputsize="full")

    prices = [float(data[x]["5. adjusted close"]) for x in data.keys()]
    prices.reverse()  # ordre croissant des prix

    dates = [x for x in data.keys()]
    dates.reverse()  # ordre croissant des dates

    return prices, dates


def plotData(prices, dates, symbol):
    dates_len = len(dates)

    fig = figure(figsize=(30, 10), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(dates, prices, color='#00c5b2')

    xticks = []
    for i in range(dates_len):
        if (i % 90 == 0 and (dates_len-i) > 90) or i == dates_len-1:
            xticks.append(dates[i])
        else:
            xticks.append(None)

    x = np.arange(0, len(xticks))

    plt.xticks(x, xticks, rotation='vertical')
    plt.title(f"Daily close prices for {symbol} from {dates[0]} to {dates[dates_len-1]}")
    plt.grid(b=None, which="major", axis="y", linestyle="--")
    plt.show()


def getDataAndPlot(symbol):
    prices, dates = getData(symbol)
    plotData(prices, dates, symbol)


def normalizeData(prices):
    global PREDICT_LEN

    min_p = np.min(prices)
    max_p = np.max(prices)

    scale = max_p - min_p

    normalized_prices = [(x - min_p) / scale for x in prices]

    final_data = []

    for i in range((len(normalized_prices) - DAYS_NUM - PREDICT_LEN) // PREDICT_LEN):
        final_data.append(normalized_prices[i * PREDICT_LEN: i * PREDICT_LEN + DAYS_NUM + PREDICT_LEN])

    final_data = np.array(final_data)

    return final_data, min_p, scale



def initTrainingDatasets(data):
    row = int(round(0.9 * data.shape[0]))  # calcule la taille de 90% des données
    train = data[:row, :]  # 90% of the data used to train
    test = data[row:, :]  # 10% of data used to test

    np.random.shuffle(train)   # on mélange bien

    X_train = train[:, :-PREDICT_LEN]
    X_test = test[:, :-PREDICT_LEN]

    y_train = train[:, -PREDICT_LEN:]
    y_test = test[:, -PREDICT_LEN:]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return [X_train, y_train, X_test, y_test]



def build_model():
    model = Sequential()
    model.add(Bidirectional
              (LSTM
               (100,
                return_sequences=True,
                input_shape=(None, 1)),
               input_shape=(DAYS_NUM, 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=PREDICT_LEN))

    model.add(Activation('linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


def train_model(model, X_train, y_train):
    model.fit(X_train,
              y_train,
              batch_size=BATCH_SIZE_TRAIN,
              epochs=EPOCHS,
              validation_split=VALIDATION_SPLIT,
              shuffle=True)


def denormalize(pred):
    return SCALE * pred + MIN_P


# Save model configuration and weights
def save_modeljson(model):
    model_json = model.to_json()
    with open("pretrained/model_config.json", "w") as f:
        f.write(model_json)
    model.save_weights("pretrained/model_weights.h5")
    print("Saved model config and weights to disk")


# Loading the model configuration and its weights
def load_modeljson(filename, weights):
    with open(filename) as f:
        model = model_from_json(f.read())
        model.load_weights(weights)
    return model


def predict(model, X_test, y_test):
    predictions = []
    correct = 0
    total = PREDICT_LEN * len(X_test)

    for i in range(len(X_test)):
        input = X_test[i]
        y_pred = model.predict(input.reshape(1, DAYS_NUM, 1))
        predictions.append(denormalize(y_pred[0][-1]))  # on dénormalise
        #print("\t previous day\t| true price\t| prediction")
        for j in range(len(y_test[i])):
            #print("\t %f\t| %f\t| %f" % (input[-1][0], denormalize(y_test[i][j]), y_pred[0][j]))
            if y_test[i][j] >= input[-1][0] and y_pred[0][j] >= input[-1][0]:
                correct += 1
            elif y_test[i][j] < input[-1][0] and y_pred[0][j] < input[-1][0]:
                correct += 1

    print("Accuracy:%.2f%%" % (100 * float(correct) / total))

    # Graphique
    y_test = SCALE * y_test + MIN_P
    y_test = y_test[:, -1]
    xs = [i for i, _ in enumerate(y_test)]
    plt.plot(xs, y_test, '#00c5b2', label='real prices')
    plt.plot(xs, predictions, '#dd2d42', label='predictions')
    plt.legend(loc=0)

    plt.title("%s - accuracy: %.2f%%" % (STOCK_TO_PREDICT, 100 * float(correct) / total))
    plt.grid(True, linestyle="--")
    plt.show()


#getDataAndPlot(STOCK_TO_PREDICT)

prices, dates = getData(STOCK_TO_PREDICT)

data, MIN_P, SCALE = normalizeData(prices)

X_train, y_train, X_test, y_test = initTrainingDatasets(data)

#model = build_model()

#train_model(model, X_train, y_train, X_test, y_test)

#save_modeljson(model)

model = load_modeljson("pretrained/model_config.json", "pretrained/model_weights.h5")

predict(model, X_test, y_test)




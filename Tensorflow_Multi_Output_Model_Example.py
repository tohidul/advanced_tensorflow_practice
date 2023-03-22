import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split


def format_output(data):
    y1 = data.pop("Y1")
    y1 = np.array(y1)
    y2 = data.pop("Y2")
    y2 = np.array(y2)

    return y1,y2


def norm(x, dataframe_stats):
    return (x-dataframe_stats["mean"]) / dataframe_stats["std"]

def plot_diff(y_true, Y_pred, title = ""):
    plt.scatter(y_true, Y_pred)
    plt.title(title)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.axis("Equal")
    plt.axis("Square")
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())

    plt.plot([-100,100],[-100,100])
    plt.show()


def plot_metrics(metric_name, title, history, ylim=5,):
    plt.title(title)
    plt.ylim(0, ylim)
    plt.plot(history.history[metric_name], color="blue", label=metric_name)
    plt.plot(history.history["val_"+metric_name], color="green", label="val_"+metric_name)
    plt.show()


URI = r"https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"

df = pd.read_excel(URI)

df = df.sample(frac=1).reset_index(drop=True)
train, test = train_test_split(df, test_size=0.2)

train_stats = train.describe()

train_stats.pop("Y1")
train_stats.pop("Y2")

train_stats = train_stats.transpose()

train_y = format_output(train)
test_y = format_output(test)

norm_train_X = norm(train, train_stats)
norm_test_X = norm(test, train_stats)

input_layer = Input(shape=(len(train.columns),))
first_dense = Dense(units='128', activation='relu')(input_layer)
second_dense = Dense(units='128', activation='relu')(first_dense)

y1_output = Dense(units='1', name='y1_output')(second_dense)

third_dense = Dense(units='64', activation='relu')(second_dense)

y2_output = Dense(units='1', name='y2_output')(third_dense)

model = Model(inputs = input_layer, outputs = [y1_output, y2_output])

print(model.summary())

optimizer = tf.keras.optimizers.SGD(lr=0.001)
model.compile(optimizer=optimizer,
              loss={'y1_output': 'mse',
                    'y2_output': 'mse'},
              metrics={
                  'y1_output': tf.keras.metrics.RootMeanSquaredError(),
                  'y2_output': tf.keras.metrics.RootMeanSquaredError()
              })

history = model.fit(norm_train_X, train_y, epochs=500, batch_size=10, validation_data=(norm_test_X, test_y))

loss, Y1_loss, Y2_loss, Y1_rmse, Y2_rmse = model.evaluate(x=norm_test_X, y=test_y)

print("Loss = {}, Y1_loss = {}, Y1_mse = {}, Y2_loss = {}, Y2_mse = {}".format(loss, Y1_loss, Y1_rmse, Y2_loss, Y2_rmse))


Y_pred = model.predict(norm_test_X)
plot_diff(test_y[0], Y_pred[0], title='Y1')
plot_diff(test_y[1], Y_pred[1], title='Y2')
plot_metrics(metric_name='y1_output_root_mean_squared_error', title='Y1 RMSE', ylim=6)
plot_metrics(metric_name='y2_output_root_mean_squared_error', title='Y2 RMSE', ylim=7)
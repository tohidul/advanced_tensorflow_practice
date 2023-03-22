import tensorflow as tf

from tensorflow.keras.utils import plot_model
import pydot
from tensorflow.keras.models import Model
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

#Example of a model using Sequential API

def build_model_with_sequential():
    seq_model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    return seq_model


#example of a model using functional API

def build_model_with_functional():

    input_layer = tf.keras.Input(shape=(28,28))

    flatten_layer = tf.keras.layers.Flatten()(input_layer)
    first_dense = tf.keras.layers.Dense(128, activation=tf.nn.relu)(flatten_layer)
    output_layer = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(first_dense)

    func_model = Model(inputs = input_layer, outputs=output_layer)

    return func_model


functional_model = build_model_with_functional()
sequential_model = build_model_with_sequential()

#ploting diagram of the functional model
plot_model(functional_model, show_shapes=True, show_layer_names=True, to_file="functional_model_example.png")

#plotting diagram of the sequential model
plot_model(sequential_model, show_shapes=True, show_layer_names=True, to_file="sequential_model_example.png")

##Training the model

#loading fashion mnist dataset
mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

#normalizing images for more accurate result
training_images = training_images/255.0

#compi8ling the model
functional_model.compile(optimizer=tf.optimizers.Adam(),
              loss="sparse_categorical_crossentropy",
              metrics = ['accuracy'])

#Training the functional model
functional_model.fit(training_images, training_labels)

#Evaluating the functional model after training
functional_model.evaluate(test_images, test_labels)



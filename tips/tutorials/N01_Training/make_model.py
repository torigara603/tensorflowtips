
"""
make model tutorial
"""

import tensorflow as tf
import tensorflow.keras as keras

def make_model() -> keras.Model:
    x0 = keras.layers.Input(shape=(28, 28, 3))
    x = keras.layers.Conv2D(32, 3, activation='relu')(x0)
    x = keras.layers.Conv2D(64, 3, activation='relu')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs=(x0), outputs=(x))
    return model

if __name__ == "__main__":
    model = make_model()
    model.summary()

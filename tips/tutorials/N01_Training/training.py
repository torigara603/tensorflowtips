import sys
import pdb

import tensorflow
import tensorflow.keras as keras
import numpy as np

import make_model

def single_traing(model:keras.Model, x:np.ndarray, y:np.ndarray):
    batch_x = x[:1]
    batch_y = y[:1]
    losses = model.train_on_batch(batch_x, batch_y)
    print(f"losses:{losses}")

def multi_training(model:keras.Model, x:np.ndarray, y:np.ndarray):
    batch_x = x
    batch_y = y
    history:keras.callbacks.History = model.fit(batch_x, batch_y, epochs=2)
    print(f"losses:{history.history['loss']}")

if __name__ == "__main__":

    imgs = np.random.random(size=(10, 28, 28, 3))
    print(imgs.shape)
    labels = np.random.randint(low=0, high=10, size=10)
    print(labels)

    model = make_model.make_model()
    model.compile(optimizer=keras.optimizers.Adam(), 
                  loss=keras.losses.SparseCategoricalCrossentropy())

    print("# start single training")
    single_traing(model, imgs, labels)

    print("# start multi training")
    multi_training(model, imgs, labels)

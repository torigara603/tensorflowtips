import os
from pathlib import Path

import tensorflow as tf
import tensorflow.keras as keras


FILE_DIR = Path(os.path.abspath(__file__)).parent

def make_model():
    wsize, hsize, csize = 28, 28, 3
    x0 = keras.layers.Input(shape=(wsize, hsize, csize))
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
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.SparseCategoricalCrossentropy())

    saved_model_file = f"{FILE_DIR}/data/models/saved_model"
    keras_model_file = f"{FILE_DIR}/data/models/keras_model.h5"
    # モデル全体のセーブ
    print("# start save tf")
    model.save(keras_model_file)
    print("# start save h5")                  
    model.save(saved_model_file)

    print("load keras model")
    # keras model インスタンスとしてロード
    # keras model load from h5 model
    keras_model1 :keras.Model = keras.models.load_model(keras_model_file)
    keras_model1.summary()
    del keras_model1
    # keras model load from tf model
    keras_model2 :keras.Model = keras.models.load_model(saved_model_file)
    keras_model2.summary()
    del keras_model2

    # tf形式としてロード
    print("load_tf_model")
    # tf model load from keras model
    # Error
    try:
       tf_model1 = tf.saved_model.load(keras_model_file)
    except Exception as e:
        print(e)
    else:
        del tf_model1
    # tf model load from tf model
    # Success
    tf_model2 = tf.saved_model.load(saved_model_file)
    del tf_model2

    # 重みだけ保存
    weights_file = f"{FILE_DIR}/data/weights/weights.h5"
    checkpoint_file = f"{FILE_DIR}/data/weights/ckpt"
    print("save weights")
    # hdf5 形式
    model.save_weights(weights_file)
    # tf形式
    model.save_weights(checkpoint_file)

    # 重みの読み込み
    print("load weights")    
    model.load_weights(weights_file)
    model.load_weights(checkpoint_file)

    model.load_weights(f"{FILE_DIR}/saved_model/variables/variables")
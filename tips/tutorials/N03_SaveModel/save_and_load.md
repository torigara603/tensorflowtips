# Tensorflow 2系のモデルの保存と読み込み

## 1. 目次

- [1. 目次](#1-目次)
- [2. 概要](#2-概要)
- [3. 開発環境](#3-開発環境)
- [4. モデル準備](#4-モデル準備)
- [5. モデル全体の保存と読み込み](#5-モデル全体の保存と読み込み)
- [6. 重みの保存と読み込み](#6-重みの保存と読み込み)

## 2. 概要

Tensorflow2 でのモデルの保存方法についてメモしておく。  
参考 <https://www.tensorflow.org/guide/keras/save_and_serialize?hl=ja#sequential_%E3%83%A2%E3%83%87%E3%83%AB%E3%81%BE%E3%81%9F%E3%81%AF_functional_api_%E3%83%A2%E3%83%87%E3%83%AB%E3%81%AE%E6%A7%8B%E6%88%90>

Tensorflow2 ではモデル全体を保存する方法とモデルの重みを保存する方法がある。  
また、それらの保存における保存形式として２つの保存形式がある。  

２つの保存形式はそれぞれ`Keras H5`形式と`Tensorflow SavedModel`形式と呼ばれる。


まずはモデル全体の保存する方法をメモしてから、次にモデルの重みを保存する方法を見ていく。

## 3. 開発環境

`Ubuntu 18.04 LTS`


```python
!cat $VIRTUAL_ENV/../pyproject.toml
```

    [tool.poetry]
    name = "tips"
    version = "0.1.0"
    description = ""
    authors = ["Your Name <you@example.com>"]
    
    [tool.poetry.dependencies]
    python = "^3.8"
    numpy = "1.19.3"
    tensorflow-cpu = "2.6.2"
    jupyter = "^1.0.0"
    nbconvert = "^6.3.0"
    Pillow = "^8.4.0"
    
    [tool.poetry.dev-dependencies]
    pytest = "^5.2"
    
    [build-system]
    requires = ["poetry-core>=1.0.0"]
    build-backend = "poetry.core.masonry.api"



```python
import os
from pathlib import Path

import tensorflow as tf
import tensorflow.keras as keras 

FILE_DIR = Path(os.path.abspath(os.path.curdir))
```

## 4. モデル準備


```python
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
```


```python
model = make_model()

model.summary()
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy())

```

    Model: "model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 28, 28, 3)]       0         
    _________________________________________________________________
    conv2d (Conv2D)              (None, 26, 26, 32)        896       
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 24, 24, 64)        18496     
    _________________________________________________________________
    flatten (Flatten)            (None, 36864)             0         
    _________________________________________________________________
    dense (Dense)                (None, 128)               4718720   
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 4,739,402
    Trainable params: 4,739,402
    Non-trainable params: 0
    _________________________________________________________________


    2021-12-18 16:12:32.573350: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.


## 5. モデル全体の保存と読み込み

モデル全体の保存は`keras.Model.save`で行うことができる。  
`Keras h5`形式での保存の場合、拡張子を`.h5`にしたファイル名を指定する。  
それ以外の場合は`SavedModel`形式で保存される。

モデル全体の保存は以下のようにすると保存することができる。


```python
keras_model_file = Path(f"{FILE_DIR}/data/models/keras_model.h5")
saved_model_file = Path(f"{FILE_DIR}/data/models/saved_model")
# モデル全体のセーブ
print("# start save h5")
model.save(keras_model_file)
print("# start save SavedModel")                  
model.save(saved_model_file)
```

    # start save h5
    # start save SavedModel
    INFO:tensorflow:Assets written to: /home/kenta/workspace/mytensorflow/tips/tips/tutorials/tutorial3/data/models/saved_model/assets


この際に`Keras h5`形式の場合はひとつのファイルにモデルの構造や重みが保存されるが、`SavedModel`形式ではディレクトリに保存される。


```python
!ls data/models/*
```

    data/models/keras_model.h5
    
    data/models/saved_model:
    assets	keras_metadata.pb  saved_model.pb  variables


saved_model.pb は saved_model形式でのモデルの構造等が保存してある。  
keras_metadata.pb はtf2.5から導入されたモノで、おそらく`SavedModel`形式から`Keras.Model`を復元するためのモノである。  
この点が`SavedModel`形式で保存するほうがよい理由となる。  
variablesはモデルの重みが保存してある。  

読み込みは以下の方法で行う。


```python
print("load keras model")
# keras model インスタンスとしてロード
# keras model load from h5 model
keras_model1 :keras.Model = keras.models.load_model(keras_model_file)
keras_model1.summary()
del keras_model1

```

    load keras model
    Model: "model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 28, 28, 3)]       0         
    _________________________________________________________________
    conv2d (Conv2D)              (None, 26, 26, 32)        896       
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 24, 24, 64)        18496     
    _________________________________________________________________
    flatten (Flatten)            (None, 36864)             0         
    _________________________________________________________________
    dense (Dense)                (None, 128)               4718720   
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 4,739,402
    Trainable params: 4,739,402
    Non-trainable params: 0
    _________________________________________________________________
    Model: "model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 28, 28, 3)]       0         
    _________________________________________________________________
    conv2d (Conv2D)              (None, 26, 26, 32)        896       
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 24, 24, 64)        18496     
    _________________________________________________________________
    flatten (Flatten)            (None, 36864)             0         
    _________________________________________________________________
    dense (Dense)                (None, 128)               4718720   
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 4,739,402
    Trainable params: 4,739,402
    Non-trainable params: 0
    _________________________________________________________________



```python
# keras model load from tf model
keras_model2 :keras.Model = keras.models.load_model(saved_model_file)
keras_model2.summary()
del keras_model2
```

    Model: "model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 28, 28, 3)]       0         
    _________________________________________________________________
    conv2d (Conv2D)              (None, 26, 26, 32)        896       
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 24, 24, 64)        18496     
    _________________________________________________________________
    flatten (Flatten)            (None, 36864)             0         
    _________________________________________________________________
    dense (Dense)                (None, 128)               4718720   
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 4,739,402
    Trainable params: 4,739,402
    Non-trainable params: 0
    _________________________________________________________________



```python
# tf形式としてロード
print("load_tf_model")
# tf model load from keras model
# Error
try:
    tf_model1 = tf.saved_model.load(keras_model_file)
except Exception as e:
    print("error")
else:
    print(tf_model1)
    del tf_model1

```

    load_tf_model
    error



```python
# tf model load from tf model
# Success
tf_model2 = tf.saved_model.load(str(saved_model_file))
print(tf_model2)
del tf_model2
```

    <tensorflow.python.saved_model.load.Loader._recreate_base_user_object.<locals>._UserObject object at 0x7f0a0437da00>


## 6. 重みの保存と読み込み

モデルの全体だけでなく、重みだけを保存する方法も存在する。  
重みだけの保存は転移学習等で使用するときに使うために用いられる。

重みの保存にも`keras h5`形式と`SavedModel`形式がある。  
どちらも`Keras.Model`のインスタンスで読み込みことができるのでどちらが良いのかは分からない。

重みの保存は以下のコードで行うことができる。


```python
# 重みだけ保存
weights_file = f"{FILE_DIR}/data/weights/weights.h5"
checkpoint_file = f"{FILE_DIR}/data/weights/ckpt"
print("save weights")
# hdf5 形式
model.save_weights(weights_file)
# tf形式
model.save_weights(checkpoint_file)
```

    save weights


これらのデータは以下のように保存されている
`weights.h5`は`weights_file`を引数に渡した時に作成されるファイルである。
それ以外のファイルは`checkpoint_file`を引数に渡した時に作成されるファイルである。  

これは`saved_model`形式でモデル全体を保存した時に作成される重みと同じである。  

`checkpoint`ファイルに関しては分からないので要調査が必要。


```python
!ls data/weights/
!echo "---"
!ls data/models/saved_model/variables
```

    checkpoint  ckpt.data-00000-of-00001  ckpt.index  weights.h5
    ---
    variables.data-00000-of-00001  variables.index


重みの読み込みは以下の方法で行う。
`SavedModel`形式のファイルはドットの前までの名前を使えば読み込むことができる。

モデル全体を保存したときに保存されたファイルも同じように読み込むことができる。


```python
# 重みの読み込み
print("load weights")
model.load_weights(weights_file)
model.load_weights(checkpoint_file)

model.load_weights(f"{FILE_DIR}/data/models/saved_model/variables/variables")
```

    load weights





    <tensorflow.python.training.tracking.util.CheckpointLoadStatus at 0x7f0a044b6a60>



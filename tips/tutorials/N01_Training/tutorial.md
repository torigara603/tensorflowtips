# Tensorflow Tutorials

TensorflowのMNISTを使ったチュートリアルについて記載する

## 1. 目次

- [1. 目次](#1-目次)
- [2. 開発環境](#2-開発環境)
- [3. 数字判定データセットで学習させる](#3-数字判定データセットで学習させる)
  - [3.1. 学習データ準備](#31-学習データ準備)
  - [3.2. モデル作成](#32-モデル作成)
  - [3.3. 学習](#33-学習)

## 2. 開発環境

`Ubuntu 18.04 LTS`

```python
# 開発環境
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
import tensorflow as tf
import tensorflow.keras as keras

import numpy as np
from PIL import Image
import IPython.display as display
```

## 3. 数字判定データセットで学習させる
数字判定のAIモデルを作ってみる。

### 3.1. 学習データ準備

数字と数字ラベルがついたデータをダウンロードする。  
データは60000枚の28x28のサイズの画像(train_x)とそれに対する正解ラベル(train_y)である。  
データの種類は`0~9`までの１０個の数字に分類されている。  


```python
# データ準備
train_data, test_data = keras.datasets.mnist.load_data()
train_x, train_y = train_data
print("train_x")
print(f"[ type  ] : {type(train_x)}")
print(f"[ shape ] : {np.shape(train_x)}")
print(f"[ dtype ] : {train_x.dtype}")
print("train_y")
print(f"[ type  ] : {type(train_y)}")
print(f"[ shape ] : {np.shape(train_y)}")
print(f"[ dtype ] : {train_y.dtype}")
```

    train_x
    [ type  ] : <class 'numpy.ndarray'>
    [ shape ] : (60000, 28, 28)
    [ dtype ] : uint8
    train_y
    [ type  ] : <class 'numpy.ndarray'>
    [ shape ] : (60000,)
    [ dtype ] : uint8


変形を行い、データをモデルに合うようにする。
1. 画措値は`0~255`の値である。モデルの学習を早くするために`0~1`の範囲に圧縮する  
2. データの次元を拡張する。グレイスケール画像なのでチャンネルサイズを`1`とする  


```python
# データ変形
train_x = train_x / 255 # normalize
train_x = np.expand_dims(train_x, axis=-1) # 28x28 -> 28x28x1
print(f"[shape] : {np.shape(train_x)}")
_, hsize, wsize, csize = np.shape(train_x)

```

    [shape] : (60000, 28, 28, 1)


正解データのクラス数を求めておく。  
求めたクラス数はモデルの出力の数を決めるために用いられる。


```python
# 正解データのクラス数（種類）
unique_labels = np.unique(train_y)
num_class = len(unique_labels)
print(f"[Unique label] : {unique_labels}")
print(f"[num class] : {num_class}")
```

    [Unique label] : [0 1 2 3 4 5 6 7 8 9]
    [num class] : 10


ちなみにココで使用したMNISTデータセットはUbuntuの場合`~/.keras/datasets`にダウンロードされる。


```python
!ls ~/.keras/datasets/
```

    mnist.npz


この`npz`ファイルはnumpyのデータが保存されたモノである。そのため`numpy.load`でロードできる。


```python
import os
_temp = np.load(f"{os.environ['HOME']}/.keras/datasets/mnist.npz")
print(_temp.files)
print(_temp['x_train'].shape)
print(_temp['y_train'].shape)
```

    ['x_test', 'x_train', 'y_train', 'y_test']
    (60000, 28, 28)
    (60000,)


### 3.2. モデル作成
入力データと出力データに沿ったモデルを作成する。  

モデルは `Functional API` 形式で作成する。 

モデルを作成したら compile を行う。
compile関数は`optimizer`と損失関数`loss`、そして`metrics`を決定する。  
optimizerにはモデルを更新する勾配を適用する最適化関数である。Adamが使われることが多いのでAdamを用いる。  
lossはモデルの出力と正解データの比較をとる損失関数である。モデルやデータの種類によって変わるが、今回は多クラス分類なので`SparseCategoricalCrossentropy`を用いる。  
metricsはモデルを評価する関数である。今回はモデルの全結果が何％正解しているかを表す`accuuracy（精度)`を用いる。  


```python

x0 = keras.layers.Input((hsize, wsize, csize))
x = keras.layers.Conv2D(32, 3, activation='relu')(x0)
x = keras.layers.Conv2D(64, 3, activation='relu')(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(128, activation='relu')(x)
x = keras.layers.Dense(num_class, activation='softmax')(x)
model = keras.Model(inputs=[x0], outputs=[x])
model.summary()
```

    Model: "model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 28, 28, 1)]       0         
    _________________________________________________________________
    conv2d (Conv2D)              (None, 26, 26, 32)        320       
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 24, 24, 64)        18496     
    _________________________________________________________________
    flatten (Flatten)            (None, 36864)             0         
    _________________________________________________________________
    dense (Dense)                (None, 128)               4718720   
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 4,738,826
    Trainable params: 4,738,826
    Non-trainable params: 0
    _________________________________________________________________


    2021-12-14 22:19:58.873474: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.


### 3.3. 学習

学習方法はいくつかあるが今回は`fit`関数を用いて学習を行う

fit関数は入力データ`train_x`と正解データ`train_y`を渡すことで学習を行うことができる。

この際にエポック数とバッチサイズを決めることができる。

学習データ全体を用いて１回モデルを学習することを１エポックと呼ぶ。１エポックではモデルの性能が満足のいくレベルに到達しないので、ふつうは複数回エポックを繰り返す。この回数をエポック数と呼ぶ。
今回は適当に 3エポック 学習を行うこととする。

バッチサイズは一回に学習するデータの数である。  
バッチサイズを設定する理由はいくつかあるがここでは記載しない
慣例として$2^n$が使われる。
今回は適当に 32 バッチサイズで行うこととする。


```python
model.fit(train_x, train_y, batch_size=32, epochs=3)
```

    Epoch 1/3
    1875/1875 [==============================] - 275s 147ms/step - loss: 0.1501 - accuracy: 0.9561
    Epoch 2/3
    1875/1875 [==============================] - 276s 147ms/step - loss: 0.0616 - accuracy: 0.9807
    Epoch 3/3
    1875/1875 [==============================] - 276s 147ms/step - loss: 0.0451 - accuracy: 0.9856



    <keras.callbacks.History at 0x7f86fc1feb20>

精度を示す`accuracy`が上昇しているのが分かる。

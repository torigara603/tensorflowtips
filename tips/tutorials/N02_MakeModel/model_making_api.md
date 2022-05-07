# Tutorial2 モデルの作成

## 1. 目次

- [1. 目次](#1-目次)
- [2. 結論](#2-結論)
- [3. 概要](#3-概要)
- [4. 開発環境](#4-開発環境)
- [5. データ](#5-データ)
- [6. Subclass API](#6-subclass-api)
- [7. Sequential API](#7-sequential-api)
- [8. Functional API](#8-functional-api)
- [9. 最後に](#9-最後に)

## 2. 結論

Functional API 作成方法でモデルを作ろう。

## 3. 概要

Tensorflow2系でのモデルの作成方法は３つの方法がある。  
それらの方法について解説していく。  

３つ方法は以下の名前で呼ばれる。  

- Subclass API
- Sequetial API
- Functional API

## 4. 開発環境

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
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

```

## 5. データ

今回はMNISTデータセットを用いる


```python
train, test = keras.datasets.mnist.load_data()
train_x, train_y = train

train_x = train_x / 255 # normalize
train_x = np.expand_dims(train_x, axis=-1) # 28x28 -> 28x28x1
print(f"[shape] : {np.shape(train_x)}")
_, hsize, wsize, csize = np.shape(train_x)

unique_labels = np.unique(train_y)
num_class = len(unique_labels)
print(f"[Unique label] : {unique_labels}")
print(f"[num class] : {num_class}")

```

    [shape] : (60000, 28, 28, 1)
    [Unique label] : [0 1 2 3 4 5 6 7 8 9]
    [num class] : 10


## 6. Subclass API

Subclass API はモデルのひな型を継承して、その内部にレイヤーなどを定義する方法である。  
この方法は`pytorch`を使っている人には馴染み深い方法である。  
`pytorch`の時は`torch.nn.Module`を継承してモデルを定義していたが、`Tensorflow2系`では`tensorflow.keras.Model`を継承してモデルを定義する。 
今回はMNISTの分類問題におけるモデルを作成してみる。 


```python
class MyModel(keras.Model):
    def __init__(self, num_class):
        super().__init__()
        self.conv1 = keras.layers.Conv2D(32, 3, activation='relu') # 畳み込み層
        self.conv2 = keras.layers.Conv2D(64, 3, activation='relu') # 畳み込み層
        self.flatten = keras.layers.Flatten() # バッチサイズ以外を平坦化
        self.dense1 = keras.layers.Dense(128, activation='relu') # 全結合層 torch.nn.flattenと同一
        self.dense2 = keras.layers.Dense(num_class, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

model = MyModel(num_class)
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=keras.metrics.Accuracy())

```

    2021-12-16 22:29:28.038684: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.


Subclass API を用いて作成されたモデルは `define by run` 形式で作成されるのでデータが渡されるまで`model.summary()`で構造を確認することができない。


```python
model.summary()
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    /tmp/ipykernel_800/3470139634.py in <module>
    ----> 1 model.summary()
    

    ~/workspace/mytensorflow/tips/.venv/lib/python3.8/site-packages/keras/engine/training.py in summary(self, line_length, positions, print_fn)
       2519     """
       2520     if not self.built:
    -> 2521       raise ValueError('This model has not yet been built. '
       2522                        'Build the model first by calling `build()` or calling '
       2523                        '`fit()` with some data, or specify '


    ValueError: This model has not yet been built. Build the model first by calling `build()` or calling `fit()` with some data, or specify an `input_shape` argument in the first layer(s) for automatic build.


一回データを通すと`model.summary()`を用いてモデルの構造を表示することができるが、`Output Shape`等の詳細な情報を表示されない。


```python
result = model.predict(train_x[:1])
model.summary()
```

    Model: "my_model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              multiple                  320       
    _________________________________________________________________
    conv2d_1 (Conv2D)            multiple                  18496     
    _________________________________________________________________
    flatten (Flatten)            multiple                  0         
    _________________________________________________________________
    dense (Dense)                multiple                  4718720   
    _________________________________________________________________
    dense_1 (Dense)              multiple                  1290      
    =================================================================
    Total params: 4,738,826
    Trainable params: 4,738,826
    Non-trainable params: 0
    _________________________________________________________________


    2021-12-16 22:29:30.558040: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)


これらの問題からあまり推奨できない。

## 7. Sequential API

Sequentaial モデルは一番簡単な作成方法である。  
この方法はレイヤーをリストに入れていき、最終的に`keras.Sequetial()`に引数として渡すだけで定義できる。  


```python
mylayers = [keras.layers.Input(shape=(28, 28, 3)),
            keras.layers.Conv2D(32, 3, activation='relu'), 
            keras.layers.Conv2D(64, 3, activation='relu'), 
            keras.layers.Flatten(), 
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(num_class, activation='softmax')]
            
model = keras.Sequential(mylayers)
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=keras.metrics.Accuracy())
model.summary()

```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_2 (Conv2D)            (None, 26, 26, 32)        896       
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 24, 24, 64)        18496     
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 36864)             0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 128)               4718720   
    _________________________________________________________________
    dense_3 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 4,739,402
    Trainable params: 4,739,402
    Non-trainable params: 0
    _________________________________________________________________


もしくは最初に`Sequential`クラスをインスタンス化しておいて、`add`コマンドでレイヤーを追加していく方式も取れる。


```python
model = keras.Sequential()
model.add(keras.layers.Input(shape=(28, 28, 3)))
model.add(keras.layers.Conv2D(32, 3, activation='relu')) # 畳み込み層
model.add(keras.layers.Conv2D(64, 3, activation='relu')) # 畳み込み層
model.add(keras.layers.Flatten()) # バッチサイズ以外を平坦化
model.add(keras.layers.Dense(128, activation='relu')) # 全結合層 torch.nn.flattenと同一
model.add(keras.layers.Dense(num_class, activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=keras.metrics.Accuracy())
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_4 (Conv2D)            (None, 26, 26, 32)        896       
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 24, 24, 64)        18496     
    _________________________________________________________________
    flatten_2 (Flatten)          (None, 36864)             0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 128)               4718720   
    _________________________________________________________________
    dense_5 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 4,739,402
    Trainable params: 4,739,402
    Non-trainable params: 0
    _________________________________________________________________


しかしコノ方法ではレイヤーが直列したモデルしか書けない。  
なので Functional API の手法を使うのが一番良いだろう。  

## 8. Functional API

Functional API では入力がレイヤーを通過していくように記述していく。  
最後に入力と最終出力を使ってモデルを作成する。  


```python
x0 = keras.layers.Input(shape=(28, 28, 3))
x = keras.layers.Conv2D(32, 3, activation='relu')(x0)
x = keras.layers.Conv2D(64, 3, activation='relu')(x) 
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(128, activation='relu')(x)
x = keras.layers.Dense(num_class, activation='softmax')(x)
model = keras.Model(inputs=[x0], outputs=[x])

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=keras.metrics.Accuracy())
model.summary()
```

    Model: "model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_3 (InputLayer)         [(None, 28, 28, 3)]       0         
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 26, 26, 32)        896       
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, 24, 24, 64)        18496     
    _________________________________________________________________
    flatten_3 (Flatten)          (None, 36864)             0         
    _________________________________________________________________
    dense_6 (Dense)              (None, 128)               4718720   
    _________________________________________________________________
    dense_7 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 4,739,402
    Trainable params: 4,739,402
    Non-trainable params: 0
    _________________________________________________________________


並列なレイヤーを持つモデルを作成してみると以下のように2出力のモデルなどが作れる。


```python
x0 = keras.layers.Input(shape=(28, 28, 3))
x = keras.layers.Conv2D(32, 3, activation='relu')(x0)
x = keras.layers.Conv2D(64, 3, activation='relu')(x) 
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(128, activation='relu')(x)
x1 = keras.layers.Dense(num_class, activation='softmax')(x)
x2 = keras.layers.Dense(100, activation='sigmoid')(x)
model = keras.Model(inputs=[x0], outputs=[x1, x2])

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=keras.metrics.Accuracy())
model.summary()
```

    Model: "model_1"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_4 (InputLayer)            [(None, 28, 28, 3)]  0                                            
    __________________________________________________________________________________________________
    conv2d_8 (Conv2D)               (None, 26, 26, 32)   896         input_4[0][0]                    
    __________________________________________________________________________________________________
    conv2d_9 (Conv2D)               (None, 24, 24, 64)   18496       conv2d_8[0][0]                   
    __________________________________________________________________________________________________
    flatten_4 (Flatten)             (None, 36864)        0           conv2d_9[0][0]                   
    __________________________________________________________________________________________________
    dense_8 (Dense)                 (None, 128)          4718720     flatten_4[0][0]                  
    __________________________________________________________________________________________________
    dense_9 (Dense)                 (None, 10)           1290        dense_8[0][0]                    
    __________________________________________________________________________________________________
    dense_10 (Dense)                (None, 100)          12900       dense_8[0][0]                    
    ==================================================================================================
    Total params: 4,752,302
    Trainable params: 4,752,302
    Non-trainable params: 0
    __________________________________________________________________________________________________


この方式はKerasTensorという仮想Tensorをレイヤーに通してモデルを作成していく過程が意識しやすい。  
例えば`x0`変数は以下の形式である。  
この方式に慣れておいたほうが後々自作レイヤーを作る際に糧となるだろう。  


```python
print(x0)
```

    KerasTensor(type_spec=TensorSpec(shape=(None, 28, 28, 3), dtype=tf.float32, name='input_4'), name='input_4', description="created by layer 'input_4'")


## 9. 最後に

以上３つのモデル作成方法 `SubClassAPI`, `SequentialAPI`, `Functional API`を紹介した。

カスタマイズ性の高さ等を考えると `Functional API`を使うのがよいだろう。

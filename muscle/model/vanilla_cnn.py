# MIT License
# 
# Copyright (c) 2023 Gregory Ditzler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf 

class VanillaCNN:
    """
    Implementation of a Vanilla CNN. This class uses a single resoluation image as the 
    input to the network. 
    """
    
    def __init__(self, 
                 learning_rate:float=0.0005, 
                 image_size:int=160, 
                 epochs:int=50):
        """_summary_

        Args:
            learning_rate (float, optional): _description_. Defaults to 0.0005.
            image_size (int, optional): _description_. Defaults to 160.
            epochs (int, optional): _description_. Defaults to 50.
        """
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.histories = []
        self.epochs = epochs
        KERNEL_SIZE = (3, 3)
        
        input = tf.keras.layers.Input(shape=(image_size, image_size, 3), name="image")
        x = tf.keras.layers.Conv2D(32, KERNEL_SIZE, activation='elu', kernel_initializer='he_uniform', padding='same')(input)
        x = tf.keras.layers.Conv2D(32, KERNEL_SIZE, activation='elu', kernel_initializer='he_uniform', padding='same')(x) 
        x = tf.keras.layers.MaxPooling2D((2, 2))(x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Conv2D(64, KERNEL_SIZE, activation='elu', kernel_initializer='he_uniform', padding='same')(x) 
        x = tf.keras.layers.Conv2D(64, KERNEL_SIZE, activation='elu', kernel_initializer='he_uniform', padding='same', strides=(2, 2))(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Conv2D(128, KERNEL_SIZE, activation='elu', kernel_initializer='he_uniform', padding='same')(x) 
        x = tf.keras.layers.Conv2D(128, KERNEL_SIZE, activation='elu', kernel_initializer='he_uniform', padding='same', strides=(2, 2))(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        predictions = tf.keras.layers.Dense(10, activation = 'softmax')(x)

        model_res = tf.keras.Model(inputs=input, outputs=predictions)

        model_res.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate), 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
                  metrics=['accuracy'])
        self.network = model_res
    
    def train(self, dataset):
        
        history = self.network.fit(
            dataset.train_ds,
            validation_data=dataset.valid_ds,
            epochs=self.epochs,
            verbose=1
        )
        self.histories.append(history) 



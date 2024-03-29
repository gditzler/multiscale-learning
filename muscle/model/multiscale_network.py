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

import numpy as np 
import tensorflow as tf 
from art.estimators.classification import TensorFlowV2Classifier
from art.defences.trainer import AdversarialTrainer, AdversarialTrainerFBF
from art.attacks.evasion import FastGradientMethod

from .utils import read_score_mrn
from .utils import get_backbone, FisherInformationLoss

class MultiResolutionNetwork: 
    def __init__(self, 
                 image_sizes:list=[32,64,160], 
                 learning_rate:float=0.0005, 
                 epochs:int=10, 
                 backbone:str='DenseNet121', 
                 loss:str='cross_entropy'): 
        """_summary_

        Args:
            image_sizes (list, optional): _description_. Defaults to [32,64,160].
            learning_rate (float, optional): _description_. Defaults to 0.0005.
            epochs (int, optional): _description_. Defaults to 10.
        """
        
        # require that we only have three image sizes... no more... no less. 
        if len(image_sizes) != 3: 
            raise(ValueError(
                ''.join(['image_sizes must be of length 3. Currently, len(', 
                str(image_sizes), ') = ', str(len(image_sizes))])
            ))
        
        self.image_sizes = image_sizes
        self.learning_rate = learning_rate
        self.histories = []
        self.epochs = epochs
        self.loss = loss 
        
        # need to set up an iterator than can be used to rename a layer of the network
        # because layer names need to be changed since we are going to use the same type 
        # of backbone. 
        k = 0 
        
        model_backbone = get_backbone(backbone=backbone)
                
        # MODEL 01 - image_size[0]
        model_01 = model_backbone(
            weights='imagenet', 
            include_top=False, 
            input_shape=(self.image_sizes[0], self.image_sizes[0], 3)
        )
        for layer in model_01.layers:
            layer.trainable = True
            layer._name = layer._name+'_'+str(k)
            k += 1

        # MODEL 02 - image_size[1]
        model_02 = model_backbone(
            weights='imagenet', 
            include_top=False, 
            input_shape=(self.image_sizes[1], self.image_sizes[1], 3)
        )
        for layer in model_02.layers:
            layer.trainable = True
            layer._name = layer._name+'_'+str(k)
            k += 1

        # MODEL 03 - image_size[2]
        model_03 = model_backbone(
            weights='imagenet', 
            include_top=False, 
            input_shape=(self.image_sizes[2], self.image_sizes[2], 3)
        )
        for layer in model_03.layers:
            layer.trainable = True
            layer._name = layer._name+'_'+str(k)
            k += 1
        
        x = tf.keras.layers.Flatten()(model_01.output)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        y = tf.keras.layers.Flatten()(model_02.output)
        y = tf.keras.layers.Dense(1024, activation='relu')(y)
        z = tf.keras.layers.Flatten()(model_03.output)
        z = tf.keras.layers.Dense(1024, activation='relu')(z)

        x = tf.keras.layers.concatenate([x, y, z], name="concat_layer")
        
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        predictions = tf.keras.layers.Dense(10, activation = 'softmax')(x)
        
        self.network = tf.keras.models.Model(
            inputs=[model_01.input, model_02.input, model_03.input], 
            outputs=predictions
        )
        
        if self.loss == 'cross_entropy': 
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        elif self.loss == 'fisher_information': 
            loss = FisherInformationLoss()
        else: 
            raise(ValueError(''.join(['Unknown loss: ', self.loss])))
 
        self.network.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate), 
            loss=loss, 
            metrics=['accuracy'], 
        )
        
    def train(self, dataset): 
        history = self.network.fit(
            dataset.train_ds,
            validation_data=dataset.valid_ds,
            epochs=self.epochs,
            verbose=1
        )
        self.histories.append(history) 
        
    def predict(self, dataset): 
        return self.network.predict(dataset)
    
    def evaluate(self, dataset, labels): 
        data = (dataset.X1, dataset.X2, dataset.X3)
        yhat = np.argmax(self.network.predict(data), axis=1) 
        return (labels==yhat).sum()/len(yhat)

    def evaluate_read(self, dataset_benign, dataset_adversary): 
        return read_score_mrn(self.network, dataset_benign, dataset_adversary)
    
      


class MultiResolutionNetworkAT: 
    def __init__(self, 
                 image_sizes:list=[32,64,160], 
                 learning_rate:float=0.0005, 
                 epochs:int=10, 
                 epsilon:float=0.075,
                 fbf:bool=False, 
                 backbone:str='DenseNet121', 
                 batch_size:int=128, 
                 loss:str='cross_entropy'): 
        """_summary_

        Args:
            image_sizes (list, optional): _description_. Defaults to [32,64,160].
            learning_rate (float, optional): _description_. Defaults to 0.0005.
            epochs (int, optional): _description_. Defaults to 10.
        """
        
        # require that we only have three image sizes... no more... no less. 
        if len(image_sizes) != 3: 
            raise(ValueError(
                ''.join(['image_sizes must be of length 3. Currently, len(', 
                str(image_sizes), ') = ', str(len(image_sizes))])
            ))
        
        self.image_sizes = image_sizes
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.histories = []
        self.epochs = epochs
        self.loss = loss
        self.fbf = fbf  
        self.batch_size = batch_size
        
        # need to set up an iterator than can be used to rename a layer of the network
        # because layer names need to be changed since we are going to use the same type 
        # of backbone. 
        k = 0 
        
        model_backbone = get_backbone(backbone=backbone)
                
        # MODEL 01 - image_size[0]
        model_01 = model_backbone(
            weights='imagenet', 
            include_top=False, 
            input_shape=(self.image_sizes[0], self.image_sizes[0], 3)
        )
        for layer in model_01.layers:
            layer.trainable = True
            layer._name = layer._name+'_'+str(k)
            k += 1

        # MODEL 02 - image_size[1]
        model_02 = model_backbone(
            weights='imagenet', 
            include_top=False, 
            input_shape=(self.image_sizes[1], self.image_sizes[1], 3)
        )
        for layer in model_02.layers:
            layer.trainable = True
            layer._name = layer._name+'_'+str(k)
            k += 1

        # MODEL 03 - image_size[2]
        model_03 = model_backbone(
            weights='imagenet', 
            include_top=False, 
            input_shape=(self.image_sizes[2], self.image_sizes[2], 3)
        )
        for layer in model_03.layers:
            layer.trainable = True
            layer._name = layer._name+'_'+str(k)
            k += 1
        
        x = tf.keras.layers.Flatten()(model_01.output)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        y = tf.keras.layers.Flatten()(model_02.output)
        y = tf.keras.layers.Dense(1024, activation='relu')(y)
        z = tf.keras.layers.Flatten()(model_03.output)
        z = tf.keras.layers.Dense(1024, activation='relu')(z)

        x = tf.keras.layers.concatenate([x, y, z], name="concat_layer")
        
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        predictions = tf.keras.layers.Dense(10, activation = 'softmax')(x)
        
        self.network = tf.keras.models.Model(
            inputs=[model_01.input, model_02.input, model_03.input], 
            outputs=predictions
        )
        
        if self.loss == 'cross_entropy': 
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        elif self.loss == 'fisher_information': 
            loss = FisherInformationLoss()
        else: 
            raise(ValueError(''.join(['Unknown loss: ', self.loss])))
 
        self.network.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate), 
            loss=loss, 
            metrics=['accuracy'], 
        )
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    
    def train(self, dataset): 
        def train_step(model, images, labels):
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = self.loss_object(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        

        history = self.network.fit(
            dataset.train_ds,
            validation_data=dataset.valid_ds,
            epochs=self.epochs,
            verbose=1
        )
        
        
        model_art = TensorFlowV2Classifier(
            model=self.network,
            loss_object=self.loss_object, 
            train_step=train_step, 
            nb_classes=10,
            input_shape=[(60,60,3), (80,80,3), (160,160,3)], #(self.image_size,self.image_size,3),
            clip_values=(0,1),
        )

        attack = FastGradientMethod(model_art, eps=self.epsilon)
        if self.fbf: 
            adv_trainer = AdversarialTrainerFBF(model_art, attack)
        else: 
            adv_trainer = AdversarialTrainer(model_art, attack)
            
        adv_trainer.fit_generator(
            dataset.train_ds, # dataset.X_train, dataset.y_train, 
            nb_epochs=self.epochs, 
            batch_size=self.batch_size
        )
        self.adv_trainer = adv_trainer
        
        self.histories.append(history) 
        
    def predict(self, X):
        return self.adv_trainer.get_classifier().predict(X) 
    
    def evaluate(self, X, y): 
        yhat = np.argmax(self.predict(X), axis=1)
        return (y==yhat).sum()/len(y)
    
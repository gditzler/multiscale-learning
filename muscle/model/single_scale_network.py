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

from .utils import read_score
from .utils import get_backbone, FisherInformationLoss

class SingleResolutionNet:  
    
    def __init__(self, 
                 learning_rate:float=0.0005, 
                 image_size:int=160, 
                 epochs:int=50, 
                 backbone:str='DenseNet121', 
                 loss:str='cross_entropy', 
                 fisher_reg:float=1e-6):
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
        self.loss = loss 
                
        model_backbone = get_backbone(backbone=backbone)

        model_res = model_backbone(
            weights='imagenet', 
            include_top=False, 
            input_shape=(self.image_size, self.image_size, 3)
        )
        for layer in model_res.layers:
            layer.trainable = True

        x = tf.keras.layers.Flatten()(model_res.output)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        predictions = tf.keras.layers.Dense(10, activation = 'softmax')(x)

        model_res = tf.keras.Model(inputs=model_res.input, outputs=predictions)
        
        if self.loss == 'cross_entropy': 
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        elif self.loss == 'fisher_information': 
            loss = FisherInformationLoss(lambda_reg=fisher_reg)
        else: 
            raise(ValueError(''.join(['Unknown loss: ', self.loss])))

        model_res.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate), 
                  loss=loss, 
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
    
    def predict(self, X): 
        return self.network.predict(X)
    
    def evaluate(self, X, y): 
        yhat = np.argmax(self.network.predict(X), axis=1) 
        return (y==yhat).sum()/len(y)
    
    def evaluate_read(self, X, Xa):
        return read_score(self.network, X, Xa)
        
 

class SingleResolutionAML: 
    def __init__(self, 
                 image_size:int=160, 
                 backbone:str='DenseNet121', 
                 learning_rate:float=0.0005, 
                 epochs:int=25, 
                 fbf:bool=False, 
                 epsilon:float=0.075,
                 loss:str='cross_entropy',
                 fisher_reg:float=1e-6,  
                 batch_size:int=128):
        """Single Resolution Neural Network with Adversarial Training. 

        Args:
            image_size (int, optional): Image size (image_size x image_size x 3). Defaults to 160.
            backbone (str, optional): Neural network backbone. Defaults to 'DenseNet121'.
            learning_rate (float, optional): Floating point learning rate. Defaults to 0.0005.
            epochs (int, optional): Number of epochs. Defaults to 25.
            epsilon (float, optional): Adversarial budget. Defaults to 0.075.
            batch_size (int, optional): Batch size for training and fine-tuning. Defaults to 128.
        """
        
        self.backbone = backbone
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.fbf = fbf 
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.loss = loss 
        
        model_backbone = get_backbone(backbone)
        model = model_backbone(
            weights='imagenet', 
            include_top=False, 
            input_shape=(image_size, image_size, 3)
        )

        for layer in model.layers:
            layer.trainable = True

        x = tf.keras.layers.Flatten()(model.output)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        predictions = tf.keras.layers.Dense(10, activation = 'softmax')(x)

        model = tf.keras.Model(inputs=model.input, outputs=predictions)
        
        if self.loss == 'cross_entropy': 
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        elif self.loss == 'fisher_information': 
            loss = FisherInformationLoss(lambda_reg=fisher_reg)
        else: 
            raise(ValueError(''.join(['Unknown loss: ', self.loss])))

        model.compile(
            loss=loss, 
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), 
            metrics=['accuracy']
        )
        self.network = model 
        self.loss_object = loss
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    
    def train(self, dataset): 
        def train_step(model, images, labels):
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = self.loss_object(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        self.network.fit(
            dataset.train_ds, 
            epochs=self.epochs
        )
        
        model_art = TensorFlowV2Classifier(
            model=self.network,
            loss_object=self.loss_object, 
            train_step=train_step, 
            nb_classes=10,
            input_shape=(self.image_size,self.image_size,3),
            clip_values=(0,1),
        )

        attack = FastGradientMethod(model_art, eps=self.epsilon)
        if self.fbf: 
            adv_trainer = AdversarialTrainerFBF(model_art, attack)
        else: 
            adv_trainer = AdversarialTrainer(model_art, attack)
            
        adv_trainer.fit(
            dataset.X_train, dataset.y_train, 
            nb_epochs=self.epochs, 
            batch_size=self.batch_size
        )
        self.adv_trainer = adv_trainer

    def predict(self, X:np.ndarray): 
        return self.adv_trainer.get_classifier().predict(X)
    
    def evaluate(self, X, y): 
        yhat = np.argmax(self.predict(X), axis=1)
        return (y==yhat).sum()/len(y)
    
    def evaluate_read(self, X, Xa):
        return read_score(self.network, X, Xa)
 
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

def get_backbone(backbone:str='DenseNet121'):
    """
    """ 
    if backbone == 'DenseNet121': 
        model_backbone = tf.keras.applications.densenet.DenseNet121 
    elif backbone == 'ResNet50': 
        model_backbone = tf.keras.applications.resnet.ResNet50
    elif backbone == 'VGG19': 
        model_backbone = tf.keras.applications.vgg19.VGG19
    else: 
        raise(ValueError(''.join(['Uknown backbone: ', backbone])))
    return model_backbone

class FisherInformationLoss(tf.keras.losses.Loss): 

    def __init__(self, lambda_reg:float=0.5, soften:bool=False): 
        super().__init__()
        self.lambda_reg = lambda_reg
        self.soften = soften
        self.epsilon = 1e-6 
    
    def call(self, y_true, y_pred):
        # reduction=tf.keras.losses.Reduction.SUM
        scce = tf.keras.losses.SparseCategoricalCrossentropy()
        cross_entropy = scce(y_true, y_pred)
        # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_pred), reduction_indices=[1]))
        if self.soften: 
            fisher = tf.reduce_mean(tf.log(1.0/(y_pred+self.epsilon)))
        else: 
            fisher = tf.reduce_mean(1.0/(y_pred+self.epsilon)) 
        return cross_entropy + self.lambda_reg*fisher

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



class ResNet50:  
    
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
        
        model_res = tf.keras.applications.resnet50.ResNet50(
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
    
    def predict(self, X): 
        return self.network.predict(X)
    
    def evaluate(self, X, y): 
        yhat = np.argmax(self.network.predict(X), axis=1) 
        return (y==yhat).sum()/len(y)
 


class DenseNet121:
    
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
        
        model_res = tf.keras.applications.densenet.DenseNet121(
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
    
    def predict(self, X): 
        return self.network.predict(X)
    
    def evaluate(self, X, y): 
        yhat = np.argmax(self.network.predict(X), axis=1) 
        return (y==yhat).sum()/len(y)


class SingleResolutionNet:  
    
    def __init__(self, 
                 learning_rate:float=0.0005, 
                 image_size:int=160, 
                 epochs:int=50, 
                 backbone:str='DenseNet121', 
                 loss:str='cross_entropy'):
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
            loss = FisherInformationLoss()
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
 

class SingleResolutionAML: 
    def __init__(self, 
                 image_size:int=160, 
                 backbone:str='DenseNet121', 
                 learning_rate:float=0.0005, 
                 epochs:int=25, 
                 fbf:bool=False, 
                 epsilon:float=0.075, 
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

        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), 
            metrics=['accuracy']
        )
        self.network = model 
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
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
      

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

import cv2
import pickle
import numpy as np 
import tensorflow as tf
import tensorflow_datasets as tfds


def prepare_adversarial_data(file_path, image_size): 
    data_dict = pickle.load(open(file_path, 'rb'))
    
    # resize the images to accomodate for the model's expected shape. 
    if data_dict['X_adv'].shape[1] != image_size:
        Xadv = np.zeros((data_dict['X_adv'].shape[0], image_size, image_size, 3))
        for i in range(len(Xadv)): 
            Xadv[i] = cv2.resize(data_dict['X_adv'][i], (image_size, image_size))
        yadv = data_dict['y']
    else: 
        Xadv, yadv = data_dict['X_adv'], data_dict['y']  
    
    return Xadv, yadv
 

class DataLoader(): 
    """_summary_
    """
    def __init__(self, image_size:int=160, batch_size:int=128, rotation:int=40, augment:bool=False, store_numpy:bool=False): 
        self.n_classes = 10 
        self.image_size = image_size
        self.batch_size = batch_size
        self.input_shape = (self.image_size, self.image_size, 3)
        self.rotation = rotation
        self.augment = augment
        self.train_ds, self.valid_ds = None, None  
        self.y_valid, self.y_train = None, None 
        self.X_valid, self.X_train = None, None 
        self.store_numpy = store_numpy
        self._load_data()
    
    
    def _load_data(self): 
        data, _ = tfds.load("imagenette/160px-v2", with_info=True, as_supervised=True)
        train_data, valid_data = data['train'], data['validation']
        train_dataset = train_data.map(
            lambda image, label: (tf.image.resize(image, (self.image_size, self.image_size)), label)
        )
        validation_dataset = valid_data.map(
            lambda image, label: (tf.image.resize(image, (self.image_size, self.image_size)), label)
        )
        
        if self.augment: 
            data_augmentation = tf.keras.models.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1),
            ])
            train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y))


        X_train, y_train = list(map(lambda x: x[0], train_dataset)), \
            list(map(lambda x: x[1], train_dataset))
        X_valid, y_valid = list(map(lambda x: x[0], validation_dataset)), \
            list(map(lambda x: x[1], validation_dataset))

        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255, rotation_range=self.rotation, height_shift_range=0.2
        )
        valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

        self.train_ds = tf.keras.preprocessing.image.NumpyArrayIterator(
            x=np.array(X_train), 
            y=np.array(y_train), 
            image_data_generator=train_datagen,
            batch_size=self.batch_size
        )

        self.valid_ds = tf.keras.preprocessing.image.NumpyArrayIterator(
            x=np.array(X_valid), 
            y=np.array(y_valid), 
            image_data_generator=valid_datagen,
            batch_size=self.batch_size
        )
        if self.store_numpy: 
            self.X_train = np.array(X_train)/255.
            self.y_train = np.array(y_train)
            self.X_valid = np.array(X_valid)/255. 
            self.y_valid = np.array(y_valid)


class DataGenFusion(tf.keras.utils.Sequence):
    
    def __init__(self, X1, X2, X3, Y,
                 batch_size,
                 shuffle=True):
        
        self.X1 = X1
        self.X2 = X2
        self.X3 = X3
        self.Y = Y 
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = len(self.X1)
        self.size = len(self.X1)

    def on_epoch_end(self):
        pass
    
    def __getitem__(self, index):
        i = np.random.randint(0, self.n, self.batch_size)
        return (self.X1[i], self.X2[i], self.X3[i]), self.Y[i]
    
    def get_batch(self): 
        return self.__getitem__(0)
    
    def __len__(self):
        return self.n // self.batch_size


class FusionDataLoader(): 
    """_summary_
    """
    def __init__(self, image_size:int=[60,80,160], batch_size:int=128, rotation:int=40, augment:bool=False):
        """_summary_

        Args:
            image_size (int, optional): _description_. Defaults to [60,80,160].
            batch_size (int, optional): _description_. Defaults to 128.
            rotation (int, optional): _description_. Defaults to 40.
            augment (bool, optional): _description_. Defaults to False.
        """
        self.train_ds = None 
        self.valid_ds = None 
        self.valid_ds_adv = None 
        self.initialized = False 
        self.image_size = image_size
        self.batch_size = batch_size
        self.rotation = rotation
        self.augment = augment 
    
    def load_benign(self):              
        dl_01 = DataLoader(
            image_size=self.image_size[0], store_numpy=True, rotation=self.rotation, augment=self.augment
        )
        dl_02 = DataLoader(
            image_size=self.image_size[1], store_numpy=True, rotation=self.rotation, augment=self.augment
        )
        dl_03 = DataLoader(
            image_size=self.image_size[2], store_numpy=True, rotation=self.rotation, augment=self.augment
        )
        self.train_ds = DataGenFusion(
            dl_01.X_train, dl_02.X_train, dl_03.X_train, dl_03.y_train, batch_size=self.batch_size
        )
        self.valid_ds = DataGenFusion(
            dl_01.X_valid, dl_02.X_valid, dl_03.X_valid, dl_03.y_valid, batch_size=self.batch_size
        )
        self.valid_labels = dl_03.y_valid
    
    def load_adversarial(self, file_path:str, image_size=[60,80,160]):

        Xadv_160, yadv_160 = prepare_adversarial_data(file_path=file_path, image_size=image_size[2])
        Xadv_80, yadv_80 = prepare_adversarial_data(file_path=file_path, image_size=image_size[1])
        Xadv_60, yadv_60 = prepare_adversarial_data(file_path=file_path, image_size=image_size[0])
        
        # check that the labels line up 
        # TO DO 
        self.valid_ds = DataGenFusion(
            Xadv_60, Xadv_80, Xadv_160, yadv_160, batch_size=self.batch_size
        )
        self.valid_labels = yadv_160
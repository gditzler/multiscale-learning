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

def get_backbone(backbone:str='DenseNet121'):
    """Fetch the pretrained neural network. 

    Args:
        backbone (str, optional): Neural network backbone from tensorgflow.keras.applications. 
           The available backbones are VGG19 and DenseNet121. Defaults to 'DenseNet121'. 

    Returns:
        tensorflow model: pretrained backbone without the head of the network. 
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
        """Fisher information regularization loss 
        
        """ 
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


def read_score(model, X:np.ndarray, Xa:np.ndarray):
    """_summary_

    Args:
        model (tensorflow.model): Tensorflow neural network (trained)
        X (np.ndarray): Benign data. 
        Xa (np.ndarray): Adversarial data.

    Returns:
        _type_: _description_
    """
    p_clean = model.predict(X) 
    p_adversary = model.predict(Xa) 
    score = 0. 
    if ((p_clean - p_adversary)**2).sum(axis=1).sum() > 0.001: 
        score = (((p_clean*np.log2(p_clean/p_adversary)).sum(axis=1)).mean() \
            + ((p_adversary*np.log2(p_adversary/p_clean)).sum(axis=1)).mean())/2.0 
    return score

def read_score_mrn(model, dataset_benign, dataset_adversary): 
    """_summary_

    Args:
        model (tensorflow.model): Tensorflow neural network. 
        dataset_benign (muscle.data.FusionDataLoader): Dataloader with benign data 
        dataset_adversary (muscle.data.FusionDataLoader): Dataloader with adversarial data 

    Returns:
        _type_: _description_
    """
    data_benign = (dataset_benign.X1, dataset_benign.X2, dataset_benign.X3)
    p_clean = model.predict(data_benign)
    data_adversary= (dataset_adversary.X1, dataset_adversary.X2, dataset_adversary.X3)
    p_adversary  = model.predict(data_adversary)
    score = 0. 
    if ((p_clean - p_adversary)**2).sum(axis=1).sum() > 0.001: 
        score = (((p_clean*np.log2(p_clean/p_adversary)).sum(axis=1)).mean() \
            + ((p_adversary*np.log2(p_adversary/p_clean)).sum(axis=1)).mean())/2.0 
    return score
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

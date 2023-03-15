# MIT License
# 
# Copyright (c) 2023 Gregory Ditzler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies ofthe Software, and to permit persons to whom the Software is
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
from art.attacks.evasion import FastGradientMethod, DeepFool, ProjectedGradientDescent
from art.attacks.evasion import CarliniLInfMethod, CarliniL2Method, CarliniL0Method
from art.attacks.evasion import AutoAttack, BasicIterativeMethod

class Attacker: 
    def __init__(self, 
                 attack_type:str='FastGradientSignMethod', 
                 epsilon:float=0.1, clip_values:tuple=(0, 1), 
                 image_shape:tuple=(160,160,3), 
                 nb_classes:int=10, 
                 max_iter:int=10): 
        """_summary_

        Args:
            attack_type (str, optional): _description_. Defaults to 'FastGradientSignMethod'.
            epsilon (float, optional): _description_. Defaults to 0.1.
            clip_values (tuple, optional): _description_. Defaults to (0, 1).
            image_shape (tuple, optional): _description_. Defaults to (160,160,3).
            nb_classes (int, optional): _description_. Defaults to 10.
            max_iter (int, optional): _description_. Defaults to 10.
        """
        self.epsilon = epsilon
        self.attack_type = attack_type
        self.clip_values = clip_values
        self.image_shape = image_shape
        self.nb_classes = nb_classes
        self.max_iter = max_iter
        
    def attack(self, network, X, y=None): 
        loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        classifier = TensorFlowV2Classifier(
            model=network,
            loss_object=loss_object, 
            nb_classes=self.nb_classes,
            input_shape=self.image_shape,
            clip_values=self.clip_values,
        )
                        
        if self.attack_type == 'FastGradientSignMethod': 
            adv_crafter = FastGradientMethod(classifier, eps=self.epsilon)
        elif self.attack_type == 'FastGradientMethod': 
            adv_crafter = FastGradientMethod(classifier, eps=self.epsilon, norm=2)
        elif self.attack_type == 'DeepFool': 
            adv_crafter = DeepFool(classifier)
        elif self.attack_type == 'ProjectedGradientDescent': 
            adv_crafter = ProjectedGradientDescent(classifier, eps=self.epsilon, max_iter=self.max_iter)
        # from here down, the attacks take way too long to generate, 
        elif self.attack_type == 'CarliniWagnerL0': 
            adv_crafter = CarliniL0Method(classifier, max_iter=self.max_iter, targeted=False)
        elif self.attack_type == 'CarliniWagnerL2': 
            adv_crafter = CarliniL2Method(classifier, max_iter=self.max_iter, targeted=False)
        elif self.attack_type == 'CarliniWagnerLinf': 
            adv_crafter = CarliniLInfMethod(classifier, max_iter=self.max_iter, targeted=False)
        elif self.attack_type == 'AutoAttack': 
            adv_crafter = AutoAttack(classifier, eps=self.epsilon)
        elif self.attack_type == 'BasicIterativeMethod': 
            adv_crafter = BasicIterativeMethod(classifier, eps=self.epsilon)
        else: 
            ValueError('Unknown attack type')
        
        Xadv = adv_crafter.generate(x=X)
        return Xadv
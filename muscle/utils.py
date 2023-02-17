# MIT License
# 
# Copyright (c) 2022 Gregory Ditzler
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

from .models import DenseNet121, MultiResolutionNetwork
from .data import DataLoader, FusionDataLoader, prepare_adversarial_data

epsilons = [(i+1)/100 for i in range(20)]
 
def load_train_evaluate(params, image_size): 
    performance = {}
    indices = [
        'FastGradientMethod', 
        'ProjectedGradientDescent', 
        'AutoAttack', 
        'BasicIterativeMethod'
    ]
    
    if type(image_size) is int: 
        dataloader = DataLoader(
            image_size=image_size, 
            batch_size=params['batch_size'], 
            rotation=params['rotation'], 
            augment=params['augment'], 
            store_numpy=True
        )
       
        network = DenseNet121(
            learning_rate=params['learning_rate'],
            image_size=image_size, 
            epochs=params['epochs']
        )
    elif len(image_size) == 3: 
        dataloader = FusionDataLoader(
            image_size=image_size, 
            batch_size=params['batch_size'], 
            rotation=params['rotation'], 
            augment=False
        )
        dataloader.load_benign()
        network = MultiResolutionNetwork(
            image_sizes=image_size, 
            learning_rate=params['learning_rate'], 
            epochs=params['epochs']
        )
    else: 
        raise(ValueError('image_size needs to be len() 3 or 1.'))
    
    network.train(dataloader)
    
    # evaluate the benign performance of the model  
    if type(image_size) is int: 
        performance['Benign'] = network.evaluate(dataloader.X_valid, dataloader.y_valid)
    else:  
        performance['Benign'] = network.evaluate(dataloader.valid_ds, dataloader.valid_labels)
    
    # evaluate the methods that have an epsilon parameter in the attack. 
    for index in indices:     
        perf = np.zeros((len(epsilons,)))
        for n, eps in enumerate(epsilons):
            file_path = ''.join(['outputs/Adversarial_', index, '_eps', str(eps), '.pkl'])
            if type(image_size) is int:  
                Xadv, yadv = prepare_adversarial_data(file_path=file_path, image_size=image_size)
                perf[n] = network.evaluate(Xadv, yadv)
            else: 
                dataloader = FusionDataLoader(
                    image_size=image_size, 
                    batch_size=params['batch_size'], 
                    rotation=params['rotation'], 
                    augment=False
                )
                dataloader.load_adversarial(file_path=file_path)
                perf[n] = network.evaluate(dataloader.valid_ds, dataloader.valid_labels)
        performance[index] = perf
    
    # evaluate deepfool 
    file_path = 'outputs/Adversarial_DeepFool.pkl'
    if type(image_size) is int:  
        Xadv, yadv = prepare_adversarial_data(file_path=file_path, image_size=image_size)
        performance['DeepFool'] = network.evaluate(Xadv, yadv)
    else: 
        dataloader = FusionDataLoader(
            image_size=image_size, 
            batch_size=params['batch_size'], 
            rotation=params['rotation'], 
            augment=False
        )
        dataloader.load_adversarial(file_path=file_path)
        performance['DeepFool'] = network.evaluate(dataloader.valid_ds, dataloader.valid_labels)
    return performance



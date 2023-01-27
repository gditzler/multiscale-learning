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

import pickle 
import numpy as np 
import tensorflow as tf
import tensorflow_datasets as tfds
from .models import DenseNet121

epsilons = [(i+1)/100 for i in range(20)]
 
def prepare_adversarial_data(file_path, image_size): 
    data_dict = pickle.load(open(file_path, 'rb'))
    
    # resize the images to accomodate for the model's expected shape. 
    if data_dict['X_adv'].shape[1] != image_size:
        Xadv = np.zeros((data_dict['Xadv'].shape[0], image_size, image_size, 3))
        for i in range(len(Xadv)): 
            Xadv[i] = cv2.resize(data_dict['Xadv'], (image_size, image_size))
        yadv = data_dict['y']
    else: 
        Xadv, yadv = data_dict['X_adv'], data_dict['y']  
    
    return Xadv, yadv
 

def load_train_evaluate(image_size): 
    dataloader = DataLoader(
        image_size=image_size, 
        batch_size=params['batch_size'], 
        rotation=params['rotation'], 
        augment=params['augment'], 
        store_numpy=True
    )
       
    network = DenseNet121(
        learning_rate=0.0005, 
        image_size=80, 
        epochs=10
    )
    network.train(dataloader)
    
    performance = {}
    performance['Benign'] = network.network.evaluate(dataloader.X_valid, dataloader.y_valid)
    
    perf = np.zeros((len(epsilons,)))
    for n, eps in enumerate(epsilons): 
        file_path = ''.join('outputs/Adversarial_FastGradientMethod_eps', str(eps), '.pkl')
        Xadv, yadv = prepare_adversarial_data(file_path=file_path, image_size=image_size)
        perf[n] = network.network.evaluate(Xadv, yadv)
    performance['FastGradientMethod'] = perf
   
    perf = np.zeros((len(epsilons,)))
    for n, eps in enumerate(epsilons): 
        file_path = ''.join('outputs/Adversarial_ProjectedGradientDescent_eps', str(eps), '.pkl')
        Xadv, yadv = prepare_adversarial_data(file_path=file_path, image_size=image_size)
        perf[n] = network.network.evaluate(Xadv, yadv)
    performance['FastGradientMethod'] = perf 
    
    file_path = ''.join('outputs/Adversarial_DeepFool.pkl')
    Xadv, yadv = prepare_adversarial_data(file_path=file_path, image_size=image_size)
    performance['DeepFool'] = network.network.evaluate(Xadv, yadv)
    return performance


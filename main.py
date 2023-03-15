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
import yaml
import pickle
from muscle.utils import load_train_evaluate, load_amltrain_evaluate

yaml_params = 'configs/config-vgg19.yaml'
 
def main():
    # load the parameters 
    with open(yaml_params, 'rb') as f: 
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    output_path = ''.join([
        params['output_path'], 
        str(params['seed']), '_', 
        params['backbone'],  '_', 
        params['loss'], 
        '.pkl'
    ])
    
    # set the random seed 
    tf.random.set_seed(params['seed'])
    
    performance_advt = load_amltrain_evaluate(params, 160, 0.05)
    
    # train and evaluate the different models. the number of epochs needs to be
    # changed for each resolution model. 
    performance_full = load_train_evaluate(params, [60, 80, 160])
    performance_160 = load_train_evaluate(params, 160)
    params['epochs'] = 35
    performance_80 = load_train_evaluate(params, 80)
    params['epochs'] = 75
    performance_60 = load_train_evaluate(params, 60)
    
    # write the results into a pickle file 
    results = {
        'performance_full': performance_full, 
        'performance_160': performance_160, 
        'performance_80': performance_80,
        'performance_60': performance_60,  
        'performance_advt': performance_advt, 
        'params': params
    }
    with open(output_path, 'wb') as f:
        pickle.dump(results, f) 
        
if __name__ == '__main__': 
    main()
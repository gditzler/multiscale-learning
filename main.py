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
import pickle
from muscle.utils import load_train_evaluate

path_output = 'outputs/single_resolution_performances_seed_'
params = {
    'batch_size': 128, 
    'rotation': 40, 
    'augment': False, 
    'store_numpy': True, 
    'learning_rate': 0.0005, 
    'epochs': 20,
    'seed': 1234
}
    
def main():
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
        'params': params
    }
    with open(''.join([path_output, str(params['seed']), '.pkl']), 'wb') as f:
        pickle.dump(results, f) 
        
if __name__ == '__main__': 
    tf.random.set_seed(params['seed'])
    main()
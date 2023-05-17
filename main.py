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
import argparse
import pickle
from muscle.utils import load_train_evaluate

parser = argparse.ArgumentParser()
parser.add_argument(
    '--yaml_params', 
    type=str, 
    default='configs/config-densenet121.yaml'
)
parser.add_argument(
    '--seed', 
    type=int, 
    default=1234
)

args = parser.parse_args()
 
def main():
    # load the parameters 
    with open(args.yaml_params, 'rb') as f: 
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    output_path = ''.join([
        params['output_path'], 
        str(args.seed), '_', 
        params['backbone'],  '_', 
        params['loss'], 
        '.pkl'
    ])
    
    # set the random seed 
    tf.random.set_seed(params['seed'])
    
    # run the adversarial training experiment 
    performance_advt = load_train_evaluate(params, 160, adversarial_training=True, epsilon=0.075)
    #performance_advt_full = load_train_evaluate(params, [60, 80, 160], adversarial_training=True, epsilon=0.075)
    
    # train and evaluate the different models. the number of epochs needs to be
    # changed for each resolution model. 
    # ---- multiscale model ---- 
    performance_full = load_train_evaluate(params, [60, 80, 160])
    # ---- single res model ----
    performance_160 = load_train_evaluate(params, 160)
    # ---- single FIM model ---- 
    params['loss'] = 'fisher_information'
    performance_fim = load_train_evaluate(params, 160)
    
    # write the results into a pickle file 
    results = {
        'performance_full': performance_full,
        # 'performance_advt_full': performance_advt_full,  
        'performance_160': performance_160, 
        'performance_advt': performance_advt, 
        'performance_fim': performance_fim, 
        'params': params
    }
    with open(output_path, 'wb') as f:
        pickle.dump(results, f) 
        
if __name__ == '__main__': 
    main()
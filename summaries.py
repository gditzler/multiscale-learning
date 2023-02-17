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

import pickle 
import matplotlib.pylab as plt 


if __name__ == '__main__': 
    epsilons = [(i+1)/100 for i in range(20)]
    file_name = 'outputs/single_resolution_performances_seed_1234.pkl'
    with open(file_name, 'rb') as file: 
        data_dict = pickle.load(file)
    
    
    plt.figure() 
    plt.plot(epsilons, data_dict['performance_full']['FastGradientMethod'], 'r', marker='o', 
             label=''.join(['MNR (', str(int(1000*data_dict['performance_full']['Benign'])/10), ')']))
    plt.plot(epsilons, data_dict['performance_160']['FastGradientMethod'], 'b', marker='o', 
             label=''.join(['Dense-160 (', str(int(1000*data_dict['performance_160']['Benign'])/10), ')']))
    plt.plot(epsilons, data_dict['performance_80']['FastGradientMethod'], 'k', marker='o', 
             label=''.join(['Dense-80 (', str(int(1000*data_dict['performance_80']['Benign'])/10), ')']))
    plt.plot(epsilons, data_dict['performance_60']['FastGradientMethod'], 'm', marker='o',  
             label=''.join(['Dense-60 (', str(int(1000*data_dict['performance_60']['Benign'])/10), ')']))
    plt.legend() 
    plt.xlabel('epsilon')
    plt.ylabel('Adversarial Performance')
    plt.title('Fast Gradient Sign Method')
    plt.show() 

    plt.figure() 
    plt.plot(epsilons, data_dict['performance_full']['ProjectedGradientDescent'], 'r', marker='o', 
             label=''.join(['MNR (', str(int(1000*data_dict['performance_full']['Benign'])/10), ')']))
    plt.plot(epsilons, data_dict['performance_160']['ProjectedGradientDescent'], 'b', marker='o', 
             label=''.join(['Dense-160 (', str(int(1000*data_dict['performance_160']['Benign'])/10), ')']))
    plt.plot(epsilons, data_dict['performance_80']['ProjectedGradientDescent'], 'k', marker='o', 
             label=''.join(['Dense-80 (', str(int(1000*data_dict['performance_80']['Benign'])/10), ')']))
    plt.plot(epsilons, data_dict['performance_60']['ProjectedGradientDescent'], 'm', marker='o',  
             label=''.join(['Dense-60 (', str(int(1000*data_dict['performance_60']['Benign'])/10), ')']))
    plt.legend() 
    plt.xlabel('epsilon')
    plt.ylabel('Adversarial Performance')
    plt.title('Projected Gradient Descent')
    plt.show() 
    
    plt.figure() 
    plt.plot(epsilons, data_dict['performance_full']['ProjectedGradientDescent'], 'r', marker='o', 
             label=''.join(['MNR-PGD (', str(int(1000*data_dict['performance_full']['Benign'])/10), ')']))
    plt.plot(epsilons, data_dict['performance_160']['ProjectedGradientDescent'], 'b', marker='o', 
             label=''.join(['Dense-160-PGD (', str(int(1000*data_dict['performance_160']['Benign'])/10), ')']))
    plt.plot(epsilons, data_dict['performance_full']['FastGradientMethod'], 'k', marker='s', 
             label=''.join(['MNR-FGSM (', str(int(1000*data_dict['performance_full']['Benign'])/10), ')']))
    plt.plot(epsilons, data_dict['performance_160']['FastGradientMethod'], 'm', marker='s', 
             label=''.join(['Dense-160-FGSM (', str(int(1000*data_dict['performance_160']['Benign'])/10), ')']))
    plt.legend() 
    plt.xlabel('epsilon')
    plt.ylabel('Adversarial Performance')
    plt.show()
    
    print('|---------------------------------------------|')
    print(''.join(['MNR-DeepFool:  ', str(int(1000*data_dict['performance_full']['DeepFool'])/10)])) 
    print(''.join(['Dense160-DeepFool:  ', str(int(1000*data_dict['performance_160']['DeepFool'])/10)]))
    print(''.join(['Dense80-DeepFool:  ', str(int(1000*data_dict['performance_80']['DeepFool'])/10)])) 
    print(''.join(['Dense60-DeepFool:  ', str(int(1000*data_dict['performance_60']['DeepFool'])/10)])) 
    print('|---------------------------------------------|')

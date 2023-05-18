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
    file_name = 'outputs/results_seed_4321_DenseNet121_cross_entropy.pkl'
    with open(file_name, 'rb') as file: 
        data_dict = pickle.load(file)
    
    
    plt.figure()
    plt.plot(epsilons, data_dict['performance_full']['FastGradientSignMethod'], 'r', marker='o', 
             label=''.join(['FGSM: MRN (', str(int(1000*data_dict['performance_full']['Benign'])/10), ')']))
    plt.plot(epsilons, data_dict['performance_160']['FastGradientSignMethod'], 'b', marker='p', 
             label=''.join(['FGSM: Dense-160 (', str(int(1000*data_dict['performance_160']['Benign'])/10), ')']))
    # plt.plot(epsilons, data_dict['performance_advt']['FastGradientSignMethod'], 'k', marker='*', 
    #          label=''.join(['FGSM: AT (', str(int(1000*data_dict['performance_advt']['Benign'])/10), ')']))
    plt.plot(epsilons, data_dict['performance_full']['ProjectedGradientDescent'], 'r', marker='o', linestyle='dashed',
             label='PGD: MRN')
    plt.plot(epsilons, data_dict['performance_160']['ProjectedGradientDescent'], 'b', marker='p', linestyle='dashed',
             label='PGD: Dense-160')
    # plt.plot(epsilons, data_dict['performance_advt']['ProjectedGradientDescent'], 'k', marker='*', linestyle='dashed',
    #          label='PGD: AT')
    #plt.plot(epsilons, data_dict['performance_fim']['FastGradientSignMethod'], 'm', marker='s', 
    #         label=''.join(['FIM (', str(int(1000*data_dict['performance_fim']['Benign'])/10), ')']))
    plt.legend() 
    plt.xlabel('epsilon')
    plt.ylabel('Adversarial Accuracy')
    # plt.title('Fast Gradient Sign Method')
    plt.show()
    
      
    print('|---------------------------------------------|')
    print('| Adversarial Performance (MRN vs Dense-160)  |')
    print(''.join(['MNR-DeepFool:         ', str(int(1000*data_dict['performance_full']['DeepFool'])/10)])) 
    print(''.join(['Dense160-DeepFool:    ', str(int(1000*data_dict['performance_160']['DeepFool'])/10)]))
    print(''.join(['Dense160-AT-DeepFool: ', str(int(1000*data_dict['performance_advt']['DeepFool'])/10)]))
    print('|---------------------------------------------|')
    print('| Adversarial READ (MRN vs Dense-160)         |')
    print(''.join(['MNR-DeepFool:         ', str(int(1000*data_dict['performance_full']['DeepFool_READ'])/10)])) 
    print(''.join(['Dense160-DeepFool:    ', str(int(1000*data_dict['performance_160']['DeepFool_READ'])/10)]))
    print(''.join(['Dense160-AT-DeepFool: ', str(int(1000*data_dict['performance_advt']['DeepFool_READ'])/10)]))
    print('|---------------------------------------------|')
    
    
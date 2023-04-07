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
import argparse
import tensorflow as tf 

from muscle.data import DataLoader
from muscle.model import SingleResolutionNet
from muscle.adversary import Attacker

parser = argparse.ArgumentParser(
    description = 'Generate the adversarial samples using the Imagenette dataset',
    epilog = 'Text at the bottom of help'
) 
parser.add_argument(
    '-s', '--seed', 
    type=int, 
    default=1234, 
    help='Random seed.'
)
parser.add_argument(
    '-o', '--output', 
    type=str, 
    help='Output Path', 
)
parser.add_argument(
    '-a', '--attack', 
    type=str, 
    default='FastGradientMethod', 
    help='Attack [FastGradientMethod, ProjectedGradientDescent, DeepFool, CarliniWagnerL0]'
)

args = parser.parse_args() 
epsilons = [(i+1)/100 for i in range(20)]
single_attacks = [
    'DeepFool', 
    'CarliniWagnerL0', 
    'CarliniWagnerL2', 
    'CarliniWagnerLinf',  
]
epsilon_attacks = [
    'FastGradientMethod', 
    'FastGradientSignMethod', 
    'ProjectedGradientDescent' ,
    'AutoAttack', 
    'BasicIterativeMethod'
]

if __name__ == '__main__': 
    tf.random.set_seed(args.seed)
    dataset = DataLoader(
        image_size=120, 
        batch_size=128, 
        rotation=40, 
        augment=False,  
        store_numpy=True
    )
    network = SingleResolutionNet(
            learning_rate=0.0005,
            image_size=120, 
            backbone='DenseNet121', 
            epochs=1
    )
    network.train(dataset)
    
    if args.attack in epsilon_attacks:
        # the attacker function call for the FGSM and PGD attack are the same since we
        # need to loop over the different values of epsilon in the attack. 
        for eps in epsilons:
            attack = Attacker(
                    attack_type=args.attack, 
                    epsilon=eps,
                    clip_values=(0,1)
            )
            X = attack.attack(network.network, dataset.X_valid, dataset.y_valid)
            pickle.dump(
                {
                    'X_adv': X, 
                    'y': dataset.y_valid, 
                    'args': args
                }, 
                open(''.join([args.output, '/Adversarial_', args.attack, '_eps', str(eps), '.pkl']), 'wb')
            )
    elif args.attack in single_attacks:
        attack = Attacker(
            attack_type=args.attack, 
            clip_values=(0,1)
        )
        # y is only used for deepfool 
        X = attack.attack(network.network, dataset.X_valid, dataset.y_valid)
        pickle.dump(
            {
                'X_adv': X, 
                'y': dataset.y_valid, 
                'args': args
            }, 
            open(''.join([args.output, '/Adversarial_', args.attack, '.pkl']), 'wb')
        )
    else: 
        ValueError(''.join(['Unknown attack ', args.attack]))
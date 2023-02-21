# To Do

* Convert `main.py` to a script that accepts commandline arguments 
* Add new attacks from `adversarial-robustness-toolbox`
  * Added: FastGradientMethod, FastGradientSignMethod, ProjectedGradientDescent, DeepFool
  * Implemented (takes a long time): CW (L0, L2, Linf), BIM, AutoAttack
* Experiments 
  * Run experiments with multiple random seeds. average the results
  * Need to work on choosing the parameters for Resnet et al. DenseNet training relatively easily but the performance of ResNet is low. 
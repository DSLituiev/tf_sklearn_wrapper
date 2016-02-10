# tflearn
a thin `scikit-learn` style wrapper for  `tensorflow` framework

## Usage 
The wrapper takes care of training (`fit()`) and prediction(`predict()`) infrastructure.
What is left to do is to specify the network topology in `_create_network()` method and 
loss function in `_create_loss()` method.

There are two subclasses of the `tflearn` class 
which come with ready to use `_create_loss()` method:
 + `rtflearn` for regression 
 + `ctflearn` for classification (in progress)


## Examples

`tf_lasso.py`: Lasso regression, a simplest example

`tf_factorization_machine.ipynb` : Factorization machines

`gan_dense.py` :  Generative adversarial network with dense perceptron layers

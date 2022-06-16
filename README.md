# numpynet
A simple and light-weight FC neural network implementation written in pure Numpy. Supports Adam, SGD, Momentum and NAG optimizers.

## Usage
```python
from Neural_Network import Network
net=Network(in_features,out_features,hidden_layers,activation)
net.train(X,y,learning_rate,batch_size)
```
example:
```python
net=Network(500,2,[100,10],'relu')
net.train(X,y,0.001,16)
```

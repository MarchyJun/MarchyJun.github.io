---
layout: article
title: NeuralNetwork - Fashion-MNIST with Tensorflow and Keras
mathjax: true
---

```python
import tensorflow as tf
from tensorflow import keras
```


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from time import time
from datetime import datetime
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import ops
from scipy.special import softmax
from sklearn.metrics import r2_score
```


```python
%run NeuralNetwork_functions.ipynb
```

# Import Fashion_MNIST dataset

MNIST data is very famous image data of numbers that is written by hand. So people usually use this data to study how to build classification deep learning model. But since it is too easy and overused, people replaced MNIST with Fashion-MNIST.        
             
It is my first time to study deep learning, so I want to learn the general process of deep learning modeling and to be familiar with tensorflow and keras by using this Fashion-MNIST dataset.


```python
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

```

# About Data


```python
train_images.shape

```




    (60000, 28, 28)



train_images have 60000 cloth images, and each image has 28 x 28 configuration. SO, in training, we have to flatten 28 x 28 array of each image to vectors with length 28 x 28 = 784.


```python
train_labels.shape

```




    (60000,)




```python
train_labels

```




    array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)



train_labels indicate to which categroy each pictures is categroized. Labels have 9 classes.

0 : T-shirt/top      
1 : Trouser       
2 : Pullover      
3 : Dress      
4 : Coat      
5 : Sandal       
6 : Shirt       
7 : Sneaker       
8 : Bag       
9 : Ankle boot        


```python
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
class_names

```




    ['T-shirt/top',
     'Trouser',
     'Pullover',
     'Dress',
     'Coat',
     'Sandal',
     'Shirt',
     'Sneaker',
     'Bag',
     'Ankle boot']




```python
test_images.shape

```




    (10000, 28, 28)



test_images have 10000 cloth images.


```python
test_labels.shape

```




    (10000,)




```python
test_labels

```




    array([9, 2, 1, ..., 8, 1, 5], dtype=uint8)



test_iamges also have the label which shows class of each cloth image.

Let's take a look at the first image of train_images


```python
temp = train_images[0,:]

plt.figure()
plt.imshow(temp)
plt.colorbar()
plt.grid(False)
plt.show()

```
![Image](/assets/images/NeuralNetwork_3.1_MNISTwithTensorflowandKeras_files/NeuralNetwork_3.1_MNISTwithTensorflowandKeras_20_0.png)



```python
class_names[train_labels[0]]
            
```




    'Ankle boot'



If we look at the label of first image, it is categorized to 'Ankle boot' class. Let's look at first 25 images and labels.


```python
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(train_images[i,:])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(class_names[train_labels[i]])
plt.show()

```


![Image](/assets/images/NeuralNetwork_3.1_MNISTwithTensorflowandKeras_files/NeuralNetwork_3.1_MNISTwithTensorflowandKeras_23_0.png)


# Data Preprocessing

### Normalizing Images


```python
print(np.min(train_images), np.max(train_images))
print(np.min(test_images), np.max(test_images))
```

    0 255
    0 255


Since our image data has value from 0 to 255, let's normalize data.


```python
train_images = train_images/255
test_images = test_images/255

print(np.min(train_images), np.max(train_images))
print(np.min(test_images), np.max(test_images))
```

    0.0 1.0
    0.0 1.0


### Flatten Images


```python
train_images.shape, test_images.shape
```




    ((60000, 28, 28), (10000, 28, 28))



For training neural network, our train data have to be shape of (number of features , number of data).


```python
train_images = train_images.reshape(train_images.shape[0], -1).T
test_images = test_images.reshape(test_images.shape[0], -1).T

train_images.shape, test_images.shape
```




    ((784, 60000), (784, 10000))



### One Hot Encoding of Labels


```python
np.unique(train_labels), np.unique(test_labels)
```




    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8),
     array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8))



Since our label data is vector with numbers ranging from 0 to 9, let's do one-hot encoding.


```python
ohm = np.zeros((train_labels.shape[0], 10))
ohm[np.arange(train_labels.shape[0]), train_labels] = 1

train_labels = ohm.T
```


```python
ohm = np.zeros((test_labels.shape[0], 10))
ohm[np.arange(test_labels.shape[0]), test_labels] = 1

test_labels = ohm.T
```


```python
train_labels.shape, test_labels.shape
```




    ((10, 60000), (10, 10000))



After one hot encoding, our labels datas have shape of (number of classes, number of data). Let's check our preprocessed data.


```python
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(train_images[:, i].reshape(28,28))
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(class_names[np.where(train_labels[:,i] == 1)[0][0]])
plt.show()

```


![Image](/assets/images/NeuralNetwork_3.1_MNISTwithTensorflowandKeras_files/NeuralNetwork_3.1_MNISTwithTensorflowandKeras_40_0.png)


# Neural Network Modeling

## 1. With TensorFlow

MLP is suitable for classification prediction probleme and also for regression prediction problem. So, it is recomended to test MLP in our any problems. The result model can be used as a baseline model to be compared with other models.

### Setting network parameters


```python
def tf_check_network_parameter(train_images, train_labels, layer_dims):
    n_x = train_images.shape[0]
    n_y = train_labels.shape[0]
    
    assert(layer_dims[0])  == n_x
    assert(layer_dims[-1]) == n_y
    
    return layer_dims
```


```python
## test ##

layer_dims = tf_check_network_parameter(train_images, train_labels, [784,3,2,10])

for i in range(len(layer_dims)):
        if i == 0:
            print('There are ' + str(layer_dims[i]) + ' nodes in input layer')
        elif i == len(layer_dims)-1:
            print('There are ' + str(layer_dims[i]) + ' nodes in output layer')
        else:
            print('There are ' + str(layer_dims[i]) + ' nodes in hidden layer ' + str(i))

```

    There are 784 nodes in input layer
    There are 3 nodes in hidden layer 1
    There are 2 nodes in hidden layer 2
    There are 10 nodes in output layer


If I want to make MLP with 4 layers(last layer is output layer), each of which have 748, 3, 2, 10 nodes, I can just input [748, 3, 2, 10] list. Then function tf_check_network_parameter will print my MLP structure.

### Creating placeholders


```python
def tf_create_placeholders(layer_dims):
    X = tf.placeholder(tf.float32, shape = (layer_dims[0] , None), name = 'X')
    Y = tf.placeholder(tf.float32, shape = (layer_dims[-1], None), name = 'Y')
    
    return X, Y

```


```python
## test ## 

tf.reset_default_graph()
X, Y = tf_create_placeholders(layer_dims)
X, Y

```




    (<tf.Tensor 'X:0' shape=(784, ?) dtype=float32>,
     <tf.Tensor 'Y:0' shape=(10, ?) dtype=float32>)



Input argument layer_dims have information about n_x and n_y. create_placeholders will return placeholders X and Y with size n_x and n_y.

### Initialising parameters

For training the neural network model, we need to initialize the weights and biases of each layer. During training, these weights and biases are updated by backpropagation. Weights and biases in each layer have shape as follow.


```python
# W_1     : [n_h_1 , n_x]          
# b_1     : [n_h_1 , 1]
# W_i     : [n_h_i , n_h_(i-1)]             
# b_i     : [n_h_i , 1]              
# W_last  : [n_y   , n_h_(last-1)]
# b_last  : [n_y   , 1]             
```


```python
def tf_initialize_parameters(layer_dims):
    tf.set_random_seed(42)
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1,L):
        if l != L:
            with tf.name_scope('hidden_layer' + str(l+1)):
                parameters['W' + str(l)] = tf.get_variable(shape = [layer_dims[l], layer_dims[l-1]], initializer = tf.contrib.layers.xavier_initializer(seed = 1), name = 'W' + str(l))
                parameters['b' + str(l)] = tf.get_variable(shape = [layer_dims[l], 1]            , initializer = tf.zeros_initializer()                          , name = 'b' + str(l))
        elif l == L:
            with tf.name_scoe('output_layer'):
                parameters['W' + str(l)] = tf.get_variable(shape = [layer_dims[l], layer_dims[l-1]], initializer = tf.contrib.layers.xavier_initializer(seed = 1), name = 'W' + str(l))
                parameters['b' + str(l)] = tf.get_variable(shape = [layer_dims[l], 1]            , initializer = tf.zeros_initializer()                          , name = 'b' + str(l))
            
    return parameters

```


```python
## test ##

layer_dims = tf_check_network_parameter(train_images, train_labels, [784,3,2,10])

tf.reset_default_graph()
parameters = tf_initialize_parameters(layer_dims)
parameters
```

    WARNING:tensorflow:From /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.





    {'W1': <tf.Variable 'W1:0' shape=(3, 784) dtype=float32_ref>,
     'b1': <tf.Variable 'b1:0' shape=(3, 1) dtype=float32_ref>,
     'W2': <tf.Variable 'W2:0' shape=(2, 3) dtype=float32_ref>,
     'b2': <tf.Variable 'b2:0' shape=(2, 1) dtype=float32_ref>,
     'W3': <tf.Variable 'W3:0' shape=(10, 2) dtype=float32_ref>,
     'b3': <tf.Variable 'b3:0' shape=(10, 1) dtype=float32_ref>}



### Forward propagation

Now if input vector and initalized weightes and biases are linealy calculated by XW + b and put this result into acitvate function in each layer, we can calculate the result. Since we want to get probabilities of 10 classes, we need to use softmax function at last. 


```python
def tf_forward_propagation(X, parameters,hidden_activation, output_activation):
    # Check hidden & output acitvation function
    if not hidden_activation in ['relu', 'sigmoid','tanh','None']:
        raise ValueError('hidden_ativation should be relu, sigmoid, tanh, or None')
    if not output_activation in ['sigmoid', 'softmax','None']:
        raise ValueError('output_ativation should be sigmoid, sotfmax, or None')
    
    # Set hidden activation function
    if hidden_activation == 'relu':
        hidden_activation_f = tf.nn.relu
    elif hidden_activation == 'sigmoid':
        hidden_activation_f = tf.nn.sigmoid
    elif hidden_activation == 'tanh':
        hidden_activation_f = tf.nn.tanh
    elif hidden_activation == 'None':
        hidden_activation_f = lambda z : z
        
    # Set output activation function
    if output_activation == 'sigmoid':
        output_activation_f = tf.nn.sigmoid
    elif output_activation == 'softmax':
        output_activation_f = lambda z : z
    elif output_activation == 'None':
        output_activation_f = lambda z : z
    
    # Forward propagation
    L = len(parameters) // 2
    A = X
    
    for l in range(1, L):
        with tf.name_scope('hidden_layer' + str(l) +'/'):
            Z = tf.add(tf.matmul(parameters['W' + str(l)], A), parameters['b' + str(l)])
            A = hidden_activation_f(Z)
    
    with tf.name_scope('output_layer/'):
        Z = tf.add(tf.matmul(parameters['W' + str(L)], A), parameters['b' + str(L)])
        A = output_activation_f(Z)
        
    return A

```

### Computin cost & Backpropagation

Now we need to compute the cost that is difference between our calculated output Z and actual class Y. Since there are 10 classes, I can use cross_entropy metrics. 
                        
Based on this cost, our weights and biases will be updated during backpropagaton. Among many optimization method of backpropagation, Adam is usually used. So I will use Adam optimiser.          


```python
def tf_compute_cost(Y_h, Y, output_activation):
    if output_activation == 'softmax':
        with tf.name_scope('Cross-Entropy'):
            logits = tf.transpose(Y_h)
            labels = tf.transpose(Y)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits)
            cost = tf.reduce_mean(loss, name = 'cross_entropy')
    
    elif output_activation == 'sigmoid':
        with tf.name_scope('Cross-Entropy'):
            loss = tf.multiply(Y, tf.log(Y_h)) + tf.multiply(1-Y, tf.log(1-Y_h))
            cost = tf.reduce_mean(loss, name = 'cross_entropy')
    
    elif output_activation == 'None':
        with tf.name_scope('MSE'):
            loss = tf.square(Y - Y_h)
            cost = tf.reduce_mean(loss, name = 'MSE')
            
    return cost
```


```python
tf.reset_default_graph()

layer_dims = tf_check_network_parameter(train_images, train_labels, [784,3,2,10])

X, Y = tf_create_placeholders(layer_dims)

parameters = tf_initialize_parameters(layer_dims)

AL = tf_forward_propagation(X, parameters, hidden_activation='relu', output_activation='softmax')

tf_compute_cost(Y_h = AL, Y = Y, output_activation = 'softmax')

```

    WARNING:tensorflow:From <ipython-input-32-404b64d9c3e8>:6: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    
    Future major versions of TensorFlow will allow gradients to flow
    into the labels input on backprop by default.
    
    See `tf.nn.softmax_cross_entropy_with_logits_v2`.
    





    <tf.Tensor 'Cross-Entropy/cross_entropy:0' shape=() dtype=float32>




```python
now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = 'tf_logs'
logdir = "{}/test/run-{}".format(root_logdir, now)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
```
![Image](/assets/images/NeuralNetwork_3.1_MNISTwithTensorflowandKeras_files/tensorboard_of_test.png)

### Modeling

Now I need to decide other hyperparameter: learning_rate, num_epochs, minibatch_size       

- learning_rate: learning rate of the optimisation
- num_epochs: how many time will the model do optimization.
- minibatch_size: how many data will the model use for optimization.


```python
def model(X_train, Y_train, X_test, Y_test, layer_dims, hidden_activation = 'relu', output_activation = 'softmax', optimizer = 'gd', learning_rate = 0.0001, n_epochs = 1500, mini_batch_size = 64, print_cost = False, plot_cost = False):
    ops.reset_default_graph()
    tf.set_random_seed(42)
    seed = 1
    
    # Set network parameters
    (n_x, m) = X_train.shape
    n_y      = Y_train.shape[0]
    cost_list = []
    
    # Get network parameter
    layer_dims = tf_check_network_parameter(X_train, Y_train, layer_dims)
    
    # Create placeholders of shape n_x, n_y
    X, Y = tf_create_placeholders(layer_dims)
    
    # Initialize parameters
    parameters = tf_initialize_parameters(layer_dims)
    
    # Forward propagation
    AL = tf_forward_propagation(X, parameters, hidden_activation, output_activation)
    
    # Get cross_entropy
    cost = tf_compute_cost(Y_h = AL, Y = Y, output_activation = output_activation)
    
    # Define the optimization algorithm
    with tf.name_scope('Optimization'):
        if optimizer == 'gd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        elif optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum = 0.9).minimize(cost)
        elif optimizer == 'rms':
            optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
        elif optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        
    # Initialize all the variables
    init = tf.global_variables_initializer()
    
    # Start session to compute Tensorflow graph
    with tf.Session() as sess: 
        # Run initializtion
        sess.run(init)
            
        # Training loop
        for epoch in range(n_epochs):                
            epoch_cost = 0
            
            n_minibatche = int(m // mini_batch_size) + 1
            seed = seed + 1
            mini_batch_list = random_mini_batches(X_train, Y_train, mini_batch_size, seed) 
            
            for minibatch in mini_batch_list:
                (X_minibatch, Y_minibatch) = minibatch
                
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict = {X: X_minibatch, Y: Y_minibatch})
            
                epoch_cost += minibatch_cost / n_minibatche
            
           
            # Print the cost
            if print_cost and epoch % 10 == 0:                                 
                print('Cost after {} epoch : {}'.format(epoch + 10, epoch_cost))
            
            cost_list.append(epoch_cost)
        
        # Plot accuracy
        if plot_cost:
            plt.figure(figsize = (16, 5))
            plt.plot(np.squeeze(cost_list), color = '#2A688B')
            plt.xticks(range(1,n_epochs + 1))
            plt.ylabel("cost")
            plt.xlabel("iterations")
            plt.title('learning rate = {rate}'.format(rate = learning_rate))
            plt.show()
        
        # Save parameters
        parameters = sess.run(parameters)
        print('Parameters have been trained')
        
        # Calculate the correct predictions
        with tf.name_scope('Evaluation'):
            if output_activation == 'softmax':
                correct_prediction = tf.equal(tf.argmax(AL), tf.argmax(Y))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
            elif output_activation == 'sigmoid':
                correct_prediction = tf.equal(tf.argmax(AL), tf.argmax(Y))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
            elif output_activation == 'None':
                accuracy = r2_score(y_pred = AL, y_true = Y)
        
        print('Train Accuracy:', accuracy.eval({X: X_train, Y: Y_train}))
        print('Test Accuracy:' , accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters
    
```


```python
parameters = model(X_train = train_images, Y_train = train_labels, X_test = test_images, Y_test = test_labels, 
                   layer_dims = [784, 50, 10], hidden_activation = 'relu', output_activation = 'softmax', 
                   optimizer = 'adam', learning_rate = 0.01, 
                   n_epochs = 20, mini_batch_size = 64, 
                   print_cost = True, plot_cost = True)

    
```

    Cost after 10 epoch : 0.5155012403755815
    Cost after 20 epoch : 0.34287964178523317



![Image](/assets/images/NeuralNetwork_3.1_MNISTwithTensorflowandKeras_files/NeuralNetwork_3.1_MNISTwithTensorflowandKeras_69_1.png)


    Parameters have been trained
    Train Accuracy: 0.8941333
    Test Accuracy: 0.8607


# 2. With TensorFlow and Keras


```python
# from tensorflow.python.keras.layers import Reshape, MaxPooling2D
# from tensorflow.python.keras.layers import InputLayer, Input
# from tensorflow.python.keras import backend as K
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
# from tensorflow.python.keras.callbacks import TensorBoard
# from tensorflow.python.keras.optimizers import Adam
# from tensorflow.python.keras.models import load_model
```


```python
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.optimizers import Adam
```


```python
def create_model(learning_rate, num_layers, num_nodes, hidden_activation):
    if num_layers <= 2:
        raise ValueError('num_layers must be greater than 2')
    
    # Set other para
    tf.reset_default_graph()
    K.clear_session()
    
    model = Sequential()
        
    for i in range(1, num_layers):
        name = 'layer_{}'.format(i)
        if i == 1:
            model.add(Dense(num_nodes, activation = hidden_activation, name = name, input_dim = train_images.shape[0]))
        elif i != num_layers-1:
            model.add(Dense(num_nodes, activation = hidden_activation, name = name))
        elif i == num_layers-1:
            model.add(Dense(train_labels.shape[0], activation = 'softmax', name = name))
    
    optimizer = Adam(lr = learning_rate)
    
    model.compile(optimizer = optimizer,
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    
    return model
```

If I input learning rate, number of layers, number of nodes and hidden activation function (output activation is defaulted as softmax), then function create_model makes keras sequential model with adam optimizer.



```python
model = create_model(learning_rate = 0.01, num_layers = 4, num_nodes = 100, hidden_activation='relu')

model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    layer_1 (Dense)              (None, 100)               78500     
    _________________________________________________________________
    layer_2 (Dense)              (None, 100)               10100     
    _________________________________________________________________
    layer_3 (Dense)              (None, 10)                1010      
    =================================================================
    Total params: 89,610
    Trainable params: 89,610
    Non-trainable params: 0
    _________________________________________________________________


This is output structure of create_model function with 4 layers and 100 nodes.


```python
model.fit(train_images.T, train_labels.T, epochs = 5, batch_size = 100)

```

    Epoch 1/5
    60000/60000 [==============================] - 2s 30us/sample - loss: 0.5219 - acc: 0.8146
    Epoch 2/5
    60000/60000 [==============================] - 1s 24us/sample - loss: 0.4037 - acc: 0.8535
    Epoch 3/5
    60000/60000 [==============================] - 2s 25us/sample - loss: 0.3830 - acc: 0.8590
    Epoch 4/5
    60000/60000 [==============================] - 1s 24us/sample - loss: 0.3627 - acc: 0.86810s - loss: 0.3
    Epoch 5/5
    60000/60000 [==============================] - 2s 25us/sample - loss: 0.3562 - acc: 0.8710





    <tensorflow.python.keras.callbacks.History at 0x152d38240>



Train accuracy by model with 5 epochs and 100 batch size is 86%


```python
test_loss, test_acc = model.evaluate(test_images.T, test_labels.T)

test_acc, test_loss
```

    10000/10000 [==============================] - 0s 28us/sample - loss: 0.4209 - acc: 0.8543





    (0.8543, 0.4209355818986893)



Test accuracy by model with 5 epochs and 100 batch size is 85%

We don't know best hyperparameters combination, so we have to do tunning our hyperparameters. We can do hyperparameter tunning with some packages, and I will study sickit optimize.

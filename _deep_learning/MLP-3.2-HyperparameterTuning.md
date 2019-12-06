---
layout: article
title: MLP - (3.2) Hyperparameter Tuning
mathjax: true
aside:
  toc: true
---

```python
%run NeuralNetwork_MNIST.ipynb
```


```python
import warnings
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args
from tensorflow.keras.models import load_model
warnings.filterwarnings(action='ignore')
```

I will check hyperparameters combinations from following hyperparameters list.
            
- learning rate: 10^-4 ~ 10^-1
- number of layers: 2 ~ 10
- number of nodes: 10 ~ 300
- hidden activation function: sigmoid, tanh, relu

To use scikit optimize, we have to make dimension of hyperparameters .


```python
dim_learning_rate = Real(low = 1e-4, high = 1e-1, prior = 'log-uniform', name = 'learning_rate')

dim_num_layers = Integer(low = 3, high = 10, name = 'num_layers')


dim_num_nodes = Integer(low = 10, high = 300, name = 'num_nodes')


dim_hidden_activation = Categorical(categories = ['sigmoid','tanh','relu'], name = 'hidden_activation')

dimensions = [dim_learning_rate,
              dim_num_layers,
              dim_num_nodes,
              dim_hidden_activation]
```

And, let's set best model name: 'best_model.keras' and default best accuracy: 0. During tuning, best combination model will be saved as 'best_model.keras' and best accuracy will be updated. 


```python
path_best_model = 'best_model.keras'

best_accuracy = 0.0
```

log_dir_name function will be used when we save tensorboard of model of each hypermarameter combinations. And let's denote test set as validation_data.


```python
def log_dir_name(learning_rate, num_layers, num_nodes, hidden_activation):
    log_dir_temp = "tf_logs/Hypertuning/lr_{}_layers_{}_nodes_{}_activation_{}"
    log_dir = log_dir_temp.format(learning_rate,
                                  num_layers,
                                  num_nodes,
                                  hidden_activation)
    return log_dir
```


```python
validation_data = (test_images.T, test_labels.T)
```

fitness function will caculate accuracy of model of each hyperparameters combinations. Since gp_minimize, which I will use for hyperparameter tuning, is minimization function, the fitness function return -accuracy.


```python
@use_named_args(dimensions = dimensions)
def fitness(learning_rate, num_layers, num_nodes, hidden_activation):
    print('learning_rate : {}'.format(learning_rate))
    print('num_layers : {}'.format(num_layers))
    print('num_nodes : {}'.format(num_nodes))
    print('hidden_activation : {}'.format(hidden_activation))
    print('--------------------------------------------------------------')
    
    model = create_model(learning_rate = learning_rate, num_layers = num_layers, num_nodes = num_nodes, hidden_activation = hidden_activation)
    
    log_dir = log_dir_name(learning_rate, num_layers, num_nodes, hidden_activation)
    
    callback_log = TensorBoard(
        log_dir = log_dir,
        histogram_freq = 0,
        batch_size = 32,
        write_graph = True,
        write_grads = False,
        write_images = False)
    
    history = model.fit(x = train_images.T,
                        y = train_labels.T,
                        epochs = 5,
                        batch_size = 64,
                        validation_data = validation_data,
                        callbacks = [callback_log])
    
    accuracy = history.history['val_accuracy'][-1]
    
    print('Accuracy: {}'.format(accuracy))
    print('===============================================================')
    
    global best_accuracy
    
    if accuracy > best_accuracy:
        model.save(path_best_model)
        best_accuracy = accuracy
        
    del model
    K.clear_session()
    
    return -accuracy
```

Let's test fitness function with default_parameters.


```python
default_parameters = [0.01, 3, 16, 'relu']

fitness(x = default_parameters)
```

    learning_rate : 0.01
    num_layers : 3
    num_nodes : 16
    hidden_activation : relu
    --------------------------------------------------------------
    WARNING:tensorflow:`batch_size` is no longer needed in the `TensorBoard` Callback and will be ignored in TensorFlow 2.0.
    Train on 60000 samples, validate on 10000 samples
    Epoch 1/5
    60000/60000 [==============================] - 2s 34us/sample - loss: 0.6026 - accuracy: 0.7893 - val_loss: 0.4848 - val_accuracy: 0.8273
    Epoch 2/5
    60000/60000 [==============================] - 2s 28us/sample - loss: 0.4684 - accuracy: 0.8353 - val_loss: 0.5019 - val_accuracy: 0.8281
    Epoch 3/5
    60000/60000 [==============================] - 1s 25us/sample - loss: 0.4513 - accuracy: 0.8416 - val_loss: 0.4834 - val_accuracy: 0.8336
    Epoch 4/5
    60000/60000 [==============================] - 1s 24us/sample - loss: 0.4391 - accuracy: 0.8452 - val_loss: 0.5206 - val_accuracy: 0.8229
    Epoch 5/5
    60000/60000 [==============================] - 2s 26us/sample - loss: 0.4317 - accuracy: 0.8466 - val_loss: 0.4922 - val_accuracy: 0.8278
    Accuracy: 0.8277999758720398
    ===============================================================





    -0.8278



Now, let's find best hyperparameters combinations with gp_minimize.

gp_minimize : Bayesian optimization using Gaussian Processes.

If every function evaluation is expensive, for instance when the parameters are the hyperparameters of a neural network and the function evaluation is the mean cross-validation score across ten folds, optimizing the hyperparameters by standard optimization routines would take for ever!

The idea is to approximate the function using a Gaussian process. In other words the function values are assumed to follow a multivariate gaussian. The covariance of the function values are given by a GP kernel between the parameters. Then a smart choice to choose the next parameter to evaluate can be made by the acquisition function over the Gaussian prior which is much quicker to evaluate.

(https://scikit-optimize.github.io/#skopt.gp_minimize)


```python
search_result_gp = gp_minimize(func = fitness,
                               dimensions = dimensions,
                               acq_func = 'EI',
                               n_calls = 50,
                               x0 = default_parameters)
```

    learning_rate : 0.01
    num_layers : 3
    num_nodes : 16
    hidden_activation : relu
    --------------------------------------------------------------
    WARNING:tensorflow:`batch_size` is no longer needed in the `TensorBoard` Callback and will be ignored in TensorFlow 2.0.
    Train on 60000 samples, validate on 10000 samples
    Epoch 1/5
    60000/60000 [==============================] - 2s 34us/sample - loss: 0.5351 - accuracy: 0.8112 - val_loss: 0.4627 - val_accuracy: 0.8365
    Epoch 2/5
    60000/60000 [==============================] - 2s 28us/sample - loss: 0.4352 - accuracy: 0.8452 - val_loss: 0.4499 - val_accuracy: 0.8396
    Epoch 3/5
    60000/60000 [==============================] - 2s 27us/sample - loss: 0.4177 - accuracy: 0.8512 - val_loss: 0.4732 - val_accuracy: 0.8314
    Epoch 4/5
    60000/60000 [==============================] - 2s 33us/sample - loss: 0.4047 - accuracy: 0.8550 - val_loss: 0.4712 - val_accuracy: 0.8340
    Epoch 5/5
    60000/60000 [==============================] - 2s 31us/sample - loss: 0.3981 - accuracy: 0.8570 - val_loss: 0.4623 - val_accuracy: 0.8422
    Accuracy: 0.842199981212616
    ===============================================================
   
    ......
    
    ===============================================================
    learning_rate : 0.0008458057097445288
    num_layers : 3
    num_nodes : 10
    hidden_activation : relu
    --------------------------------------------------------------
    WARNING:tensorflow:`batch_size` is no longer needed in the `TensorBoard` Callback and will be ignored in TensorFlow 2.0.
    Train on 60000 samples, validate on 10000 samples
    Epoch 1/5
    60000/60000 [==============================] - 2s 34us/sample - loss: 0.8253 - accuracy: 0.7280 - val_loss: 0.5696 - val_accuracy: 0.8094
    Epoch 2/5
    60000/60000 [==============================] - 2s 28us/sample - loss: 0.5070 - accuracy: 0.8276 - val_loss: 0.5119 - val_accuracy: 0.8230
    Epoch 3/5
    60000/60000 [==============================] - 2s 27us/sample - loss: 0.4648 - accuracy: 0.8397 - val_loss: 0.4886 - val_accuracy: 0.8273
    Epoch 4/5
    60000/60000 [==============================] - 2s 25us/sample - loss: 0.4449 - accuracy: 0.8461 - val_loss: 0.4764 - val_accuracy: 0.8297
    Epoch 5/5
    60000/60000 [==============================] - 2s 26us/sample - loss: 0.4326 - accuracy: 0.8506 - val_loss: 0.4662 - val_accuracy: 0.8363
    Accuracy: 0.8363000154495239
    ===============================================================
    learning_rate : 0.0008482758527035384
    num_layers : 4
    num_nodes : 300
    hidden_activation : relu
    --------------------------------------------------------------
    WARNING:tensorflow:`batch_size` is no longer needed in the `TensorBoard` Callback and will be ignored in TensorFlow 2.0.
    Train on 60000 samples, validate on 10000 samples
    Epoch 1/5
    60000/60000 [==============================] - 4s 71us/sample - loss: 0.4772 - accuracy: 0.8282 - val_loss: 0.4176 - val_accuracy: 0.8490
    Epoch 2/5
    60000/60000 [==============================] - 3s 56us/sample - loss: 0.3541 - accuracy: 0.8693 - val_loss: 0.3830 - val_accuracy: 0.8567
    Epoch 3/5
    60000/60000 [==============================] - 3s 53us/sample - loss: 0.3202 - accuracy: 0.8824 - val_loss: 0.3756 - val_accuracy: 0.8650
    Epoch 4/5
    60000/60000 [==============================] - 3s 53us/sample - loss: 0.2956 - accuracy: 0.8892 - val_loss: 0.3493 - val_accuracy: 0.8705
    Epoch 5/5
    60000/60000 [==============================] - 3s 58us/sample - loss: 0.2796 - accuracy: 0.8956 - val_loss: 0.3374 - val_accuracy: 0.8780
    Accuracy: 0.878000020980835
    ===============================================================



```python
plot_convergence(search_result_gp)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x635f0cc18>



![Image](/assets/images/NeuralNetwork_3.2_HyperparameterTuning_files/NeuralNetwork_3.2_HyperparameterTuning_17_1.png))


```python
sorted(zip(search_result.func_vals, search_result.x_iters))
```




    [(-0.8802, [0.0002959462219949908, 7, 300, 'relu']),
     (-0.8788, [0.002190069004164229, 3, 188, 'relu']),
     (-0.8767, [0.0019588685500944304, 3, 198, 'sigmoid']),
     (-0.8753, [0.0008136447333568535, 6, 204, 'relu']),
     (-0.875, [0.0002543070771366, 5, 300, 'relu']),
     (-0.875, [0.0004263545856054099, 10, 300, 'relu']),
     (-0.8748, [0.00211913276820656, 5, 204, 'relu']),
     (-0.8742, [0.0006245130655079226, 6, 110, 'relu']),
     (-0.8741, [0.0021441596751991033, 4, 205, 'relu']),
     (-0.8741, [0.0031078230575074003, 5, 179, 'relu']),
     (-0.8733, [0.0005368559350389544, 3, 247, 'relu']),
     (-0.8733, [0.0014917236123558131, 5, 300, 'relu']),
     (-0.8733, [0.0025403614669968007, 5, 215, 'relu']),
     (-0.873, [0.0005777821047247657, 5, 300, 'tanh']),
     (-0.8728, [0.0010921985570836535, 4, 295, 'sigmoid']),
     (-0.8725, [0.00021325222660068934, 8, 289, 'tanh']),
     (-0.8705, [0.0001731221179533443, 6, 201, 'tanh']),
     (-0.8694, [0.0024891292746113992, 3, 117, 'relu']),
     (-0.8692, [0.00022148616780747255, 9, 125, 'relu']),
     (-0.8688, [0.0014607434782379874, 5, 300, 'relu']),
     (-0.8683, [0.0013718030910478862, 3, 300, 'tanh']),
     (-0.8676, [0.0015377237202258772, 8, 300, 'relu']),
     (-0.8673, [0.004161663364656439, 5, 242, 'sigmoid']),
     (-0.8667, [0.0001, 10, 300, 'relu']),
     (-0.8666, [0.0006384236766955954, 3, 300, 'sigmoid']),
     (-0.8659, [0.0041838244571849375, 4, 296, 'relu']),
     (-0.8656, [0.0001, 6, 300, 'relu']),
     (-0.8656, [0.002310206296046986, 7, 179, 'relu']),
     (-0.8646, [0.002342779409035649, 4, 222, 'relu']),
     (-0.8642, [0.000687514796452139, 3, 63, 'relu']),
     (-0.8622, [0.0021703045203809884, 5, 205, 'relu']),
     (-0.8614, [0.0022692413840411214, 6, 85, 'sigmoid']),
     (-0.859, [0.00228613065086726, 4, 212, 'sigmoid']),
     (-0.8553, [0.0030360313129710326, 3, 50, 'sigmoid']),
     (-0.8516, [0.0001, 3, 300, 'relu']),
     (-0.8417, [0.009723073301251328, 3, 300, 'sigmoid']),
     (-0.8371, [0.0012403912672125997, 3, 10, 'sigmoid']),
     (-0.8335, [0.009137676680624587, 7, 183, 'relu']),
     (-0.8323, [0.0001, 3, 300, 'sigmoid']),
     (-0.8315, [0.002164647984422405, 4, 10, 'relu']),
     (-0.8314, [0.00014594682763003234, 9, 25, 'relu']),
     (-0.831, [0.01, 3, 16, 'relu']),
     (-0.8293, [0.00024465639924243034, 3, 10, 'relu']),
     (-0.8221, [0.0008311749291350613, 10, 10, 'relu']),
     (-0.8186, [0.020110421609520018, 3, 147, 'sigmoid']),
     (-0.8087, [0.0022611616626187223, 9, 264, 'sigmoid']),
     (-0.799, [0.0001071718121090779, 8, 10, 'tanh']),
     (-0.7958, [0.00038907294706676173, 8, 224, 'sigmoid']),
     (-0.6556, [0.05774360324692692, 3, 281, 'tanh']),
     (-0.5557, [0.012861204033050726, 10, 10, 'sigmoid'])]




```python
print('Best Parameter Set')
print('- Learning rate : {}'.format(search_result.x[0]))
print('- Number of layers : {}'.format(search_result.x[1]))
print('- Number of nodes : {}'.format(search_result.x[2]))
print('- Activation function : {}'.format(search_result.x[3]))
```

    Best Parameter Set
    - Learning rate : 0.0002959462219949908
    - Number of layers : 7
    - Number of nodes : 300
    - Activation function : relu



```python
model = load_model(path_best_model)
```


```python
test_loss, test_acc = model.evaluate(x = test_images.T, 
                                     y = test_labels.T)
```

    10000/1[==============================] - 1s 69us/sample - loss: 0.2637 - accuracy: 0.8802



```python
print('Test accuracy and loss with best model')
print('Test loss : {}'.format(test_loss))
print('Test accuracy : {}'.format(test_acc))
```

    Test accuracy and loss with best model
    Test loss : 0.3292090726137161
    Test accuracy : 0.8802000284194946


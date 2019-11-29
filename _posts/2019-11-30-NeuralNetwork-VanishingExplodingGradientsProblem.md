---
layout: article
title: NeuralNetwork - Vanishing and Exploding Gradients Problem
mathjax: true
---

```python
%run NeuralNetwork_functions.ipynb
```

# Vanishing & Exploding Gradients Problem

One of the problems of training very deep neural networks is vanishing and exploding gradients. This means that when we are training a very deep network, our derivatives can sometimes get either very big or very small, and this makes training difficult. Let's consider following deep network without b. For the sake of simplicity, let's say we using linear activation function.

![Image](/assets/images/NeuralNetwork_2.2_VanishingExplodingGradientsProblem_files/deep.png)


$dZ^{[L]} \\
 dW^{[L]} = dZ^{[L]}Z^{[L-1]} \\
 \qquad\:\: = dZ^{[L]}W^{[L-1]}W^{[L-2]} \dots W^{[1]}X \\
 $
             
$dZ^{[L-1]} \:\, = dZ^{[L]}W^{[L]}\\
 dW^{[L-1]} = dZ^{[L-1]}Z^{[L-2]} \\
 \qquad\:\:\:\:\:\, = dZ^{[L-1]}W^{[L-2]} \dots W^{[1]}X \\
 \qquad\:\:\:\:\:\, = dZ^{[L]}W^{[L]}W^{[L-2]} \dots W^{[1]}X \\
 $

$dZ^{[L-2]} \:\, = dZ^{[L-1]}W^{[L-1]}\\
 dW^{[L-2]} = dZ^{[L-2]}Z^{[L-3]} \\
 \qquad\:\:\:\:\:\, = dZ^{[L-2]}W^{[L-3]}W^{[L-4]} \dots W^{[1]}X \\
 \qquad\:\:\:\:\:\, = dZ^{[L-1]}W^{[L-1]}W^{[L-3]}W^{[L-4]} \dots W^{[1]}X \\
 \qquad\:\:\:\:\:\, = dZ^{[L]}W^{[L]}W^{[L-1]}W^{[L-3]}W^{[L-4]} \dots W^{[1]}X \\ 
 $

$dW^{[l]} = dZ^{[L]}W^{[L]}W^{[L-1]} \dots W^{[l+1]}W^{[l-1]}W^{[l-2]} \dots W^{[1]}X \\ 
\qquad\,  = dZ^{[L]}\prod_{i=1,\neq l}^{L}W^{[i]}
$
 

$ So,\: if \:\: W^{[l]} = \begin{bmatrix} 1.5 & 0 \\ 0 & 1.5 \end{bmatrix} > I \:\: for\: l = 1 \dots L,\: then\: dW^{[l]} = dZ^{[L]} \begin{bmatrix} 1.5^{L-2} & 0 \\ 0 & 1.5^{L-2}\end{bmatrix} W^{[1]}X \\
$                    

It shows gradients $dW^{[l]}$ will explode exponentially.

$ Also,\: if \:\: W^{[l]} = \begin{bmatrix} 0.5 & 0 \\ 0 & 0.5 \end{bmatrix} < I \:\: for\: l = 1 \dots L,\: then\: dW^{[l]} = dZ^{[L]} \begin{bmatrix} 0.5^{L-2} & 0 \\ 0 & 0.5^{L-2}\end{bmatrix} W^{[1]}X \\
$ 
              
It shows gradients $dW^{[l]}$ will vanish exponentially.
                  


# Initialization Methods for Vanishing & Exploding Gradients Problem

It turns out that a partial solution to above vanishing & exploding gradient problem is more careful choice of the random initialization for our neural network.

![Image](/assets/images/NeuralNetwork_2.2_VanishingExplodingGradientsProblem_files/deep2.png)

Let's consider above simple network. 

$Z^{[l]}_{1} = W^{[l]}Z^{[l-1]} \\
\quad\:\: = W^{[l]}_{1}Z^{[l-1]}_{1} + W^{[l]}_{2}Z^{[l-1]}_{2} + \dots + W^{[l]}_{n^{[l-1]}}Z^{[l-1]}_{n^{[l-1]}}$
                
So, if $n^{[l-1]}$ is big(= have many neurons), then since $Z^{[l]}$ is sum of $n^{[l-1]}$ terms, it has high probability to explode or diminish. So, we want to set $W^{[l]}$ to be small. We can do this by add some terms in our random initialization.

$W^{[l]}$ = random_initialization * $\sqrt{V(W^{[l]})}$
- Relu activation :  $V(W^{[l]}) = \frac{2}{n^{[l-1]}} $
- tanh activation :  $V(W^{[l]}) = \frac{1}{n^{[l-1]}} $
     
Let's compare different initialization methods.

# Compare Initialization Methods

### Zero initialization


```python
initialize_parameters_zero([2,3,1])
```




    {'W1': array([[0., 0.],
            [0., 0.],
            [0., 0.]]), 'b1': array([[0.],
            [0.],
            [0.]]), 'W2': array([[0., 0., 0.]]), 'b2': array([[0.]])}




```python
parameters_zero = nn_model(X, Y, [2,3,1], initialization_method = 'zero', hidden_activation = 'tanh', output_activation = 'sigmoid', num_iterations = 10000, learning_rate = 1.2, print_cost = True, plot_cost = True)
```

    Cost after 1000 iteration : 0.693147
    Cost after 2000 iteration : 0.693147
    Cost after 3000 iteration : 0.693147
    Cost after 4000 iteration : 0.693147
    Cost after 5000 iteration : 0.693147
    Cost after 6000 iteration : 0.693147
    Cost after 7000 iteration : 0.693147
    Cost after 8000 iteration : 0.693147
    Cost after 9000 iteration : 0.693147
    Cost after 10000 iteration : 0.693147



![Image](/assets/images/NeuralNetwork_2.2_VanishingExplodingGradientsProblem_files/NeuralNetwork_2.2_VanishingExplodingGradientsProblem_13_1.png)



```python
predictions_zero = predict(parameters_zero, X, hidden_activation = 'tanh', output_activation = 'sigmoid')

confusion_matrix_zero = metrics.confusion_matrix(Y[0], predictions_zero[0])

accuracy_zero, F1_score_zero = get_accuracy(confusion_matrix_zero)

print('When initialized weights are zero')
print('Accuracy : {}%'.format(round(accuracy_zero * 100, 4)))
print('F1 Score : {}%'.format(round(F1_score_zero, 3)))
print('-------------------------------------')

plt.figure(figsize = (12,5))

plt.subplot(1,2,1)
sns.scatterplot(X[0,:], X[1,:], hue = Y[0])
plt.title('Actual Data')

plt.subplot(1,2,2)
sns.scatterplot(X[0,:], X[1,:], hue = predictions_zero[0])
plt.title('Zero initialized')

plt.show()
```

    When initialized weights are zero
    Accuracy : 50.0%
    F1 Score : 0.667%
    -------------------------------------



![Image](/assets/images/NeuralNetwork_2.2_VanishingExplodingGradientsProblem_files/NeuralNetwork_2.2_VanishingExplodingGradientsProblem_14_1.png)


The model's performace is very bad. If we check the prediction, then the model is predicting 0 for every example.        
          
In general, initializing all the weights to zero results in the network failing to break symmetry. This means every neuron in each layer will learn the same thing, and we might as well be training a neural network with $\ n^{[l]} = 1$ for every layer, and the network is no more powerful than a linear classifier such as logistic regression.           
 
Thus weights should be initialized randomly to break symmetry. It is okay to initialize the biases to zeros because symmetry will be broken as long as W is initialized randomly.

### Random initialization with large number

To break symmetry, let's initialize the weights randomly. Following random initialization, each neuron can then proceed to learn a different function of its inputs. But even though we initialize weights randomly, if initialized weights are big, then there can be some problems with the performance.


```python
def initialize_parameters_random_howbig(layer_dims, how_big):
    np.random.seed(42)
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1,L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * how_big
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1)) 
    
    return parameters   
```


```python
def nn_model_howbig(X, Y, layer_dims, initialization_method = 'random',how_big = 0.01, hidden_activation = 'sigmoid', output_activation = 'None', num_iterations = 10000, learning_rate = 0.01, print_cost = False, plot_cost = False):
    np.random.seed(42)
    cost_list = []
    
    # Check network parameters
    layer_dims = check_network_parameter(X, Y, layer_dims)
    
    # Initialize parameters
    if not initialization_method in ['zero', 'random']:
        raise ValueError('initialization_method should be zero or random')
    if initialization_method == 'zero':
        parameters = initialize_parameters_zero(layer_dims)
    else:
        parameters = initialize_parameters_random_howbig(layer_dims, how_big)
    
    # Iteration for optimizing our parameters
    for i in range(num_iterations):
        # Forward propagation
        cache = forward_propagation(X, parameters, hidden_activation, output_activation)
        AL = cache['AL']
        
        # Compute cost
        cost = compute_cost(Y_h = AL, Y = Y, output_activation = output_activation)
        
        # Backwoard propagation
        grads = backward_propagation(X, Y, parameters, cache, hidden_activation, output_activation)
        
        # Gradient descent parameter update
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # print the cost every 1000 iterations
        if i % 100 == 0:
            cost_list.append(cost)
        if (i % 1000 == 0) & print_cost:
            print('Cost after %i iteration : %f' %(i + 1000, cost))
    
    if plot_cost:
        plt.plot(cost_list)
        plt.xlabel('Iterations')
        plt.ylabel('cost')
        plt.title('Learning rate = ' + str(learning_rate))

    return parameters
```


```python
# The case that initialized number is very big
initialize_parameters_random_howbig([2,3,1], 10)
```




    {'W1': array([[ 4.96714153, -1.38264301],
            [ 6.47688538, 15.23029856],
            [-2.34153375, -2.34136957]]), 'b1': array([[0.],
            [0.],
            [0.]]), 'W2': array([[15.79212816,  7.67434729, -4.69474386]]), 'b2': array([[0.]])}




```python
print('When initialized weights are big')
parameters_random_big = nn_model_howbig(X, Y, [2,3,1], initialization_method = 'random', how_big = 10,hidden_activation = 'tanh', output_activation = 'sigmoid', num_iterations = 10000, learning_rate = 1.2, print_cost = True, plot_cost = True)

```

    When initialized weights are big
    Cost after 1000 iteration : 9.685794
    Cost after 2000 iteration : 0.440218
    Cost after 3000 iteration : 0.438117
    Cost after 4000 iteration : 0.437805
    Cost after 5000 iteration : 0.437631
    Cost after 6000 iteration : 0.437500
    Cost after 7000 iteration : 0.437371
    Cost after 8000 iteration : 0.437242
    Cost after 9000 iteration : 0.437112
    Cost after 10000 iteration : 0.436981



![Image](/assets/images/NeuralNetwork_2.2_VanishingExplodingGradientsProblem_files/NeuralNetwork_2.2_VanishingExplodingGradientsProblem_21_1.png)


The cost starts very high, this is because initialized weights are very big. With these big weights, our first outputs will be very close to 0 or 1 for some examples and it cause very high losses. Also, if we compare with following small weights, we can see that optimization rarely occures. This is due to high initialized weights, which result in vanishing gradients.


```python
predictions_big = predict(parameters_random_big, X, hidden_activation = 'tanh', output_activation = 'sigmoid')

confusion_matrix_big = metrics.confusion_matrix(Y[0], predictions_big[0])

accuracy_big, F1_score_big = get_accuracy(confusion_matrix_big)

print('When initialized weights are big')
print('Accuracy : {}%'.format(round(accuracy_big * 100, 4)))
print('F1 Score : {}%'.format(round(F1_score_big, 3)))
print('-------------------------------------')

plt.figure(figsize = (12,5))

plt.subplot(1,2,1)
sns.scatterplot(X[0,:], X[1,:], hue = Y[0])
plt.title('Actual Data')

plt.subplot(1,2,2)
sns.scatterplot(X[0,:], X[1,:], hue = predictions_big[0])
plt.title('Big number initialized')

plt.show()
```

    When initialized weights are big
    Accuracy : 74.2%
    F1 Score : 0.596%
    -------------------------------------


![Image](/assets/images/NeuralNetwork_2.2_VanishingExplodingGradientsProblem_files/NeuralNetwork_2.2_VanishingExplodingGradientsProblem_23_1.png)


As we can see in the above example, initializing weights to very large random values does not work well, and initializing with small random values does better.
                 
Then how small should be these random values be? 

### Random initialization with appropriate random values corresponding to activation function

There is typical rule for appropriate random values corresponding to each activation function.

$W^{[l]}$ = random_initialization * $\sqrt{V(W^{[l]})}$
- If Relu activation :  $V(W^{[l]}) = \frac{2}{n^{[l-1]}} $
- If tanh activation :  $V(W^{[l]}) = \frac{1}{n^{[l-1]}} $


```python
def initialize_parameters_random_approp(layer_dims, hidden_activation):
    np.random.seed(42)
    parameters = {}
    L = len(layer_dims)
    
    if hidden_activation == 'sigmoid' or 'tanh':
        for l in range(1,L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * (1/layer_dims[l-1])**(1/2)
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1)) 
    elif hidden_activation == 'relu':
        for l in range(1,L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * (2/layer_dims[l-1])**(1/2)
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1)) 

    return parameters   

```


```python
def nn_model_approp(X, Y, layer_dims, hidden_activation = 'sigmoid', output_activation = 'None', num_iterations = 10000, learning_rate = 0.01, print_cost = False, plot_cost = False):
    np.random.seed(42)
    cost_list = []
    
    # Check network parameters
    layer_dims = check_network_parameter(X, Y, layer_dims)
    
    # Initialize parameters
    parameters = initialize_parameters_random_approp(layer_dims, hidden_activation)
    
    # Iteration for optimizing our parameters
    for i in range(num_iterations):
        # Forward propagation
        cache = forward_propagation(X, parameters, hidden_activation, output_activation)
        AL = cache['AL']
        
        # Compute cost
        cost = compute_cost(Y_h = AL, Y = Y, output_activation = output_activation)
        
        # Backwoard propagation
        grads = backward_propagation(X, Y, parameters, cache, hidden_activation, output_activation)
        
        # Gradient descent parameter update
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # print the cost every 1000 iterations
        if i % 100 == 0:
            cost_list.append(cost)
        if (i % 1000 == 0) & print_cost:
            print('Cost after %i iteration : %f' %(i + 1000, cost))
    
    if plot_cost:
        plt.plot(cost_list)
        plt.xlabel('Iterations')
        plt.ylabel('cost')
        plt.title('Learning rate = ' + str(learning_rate))

    return parameters
```


```python
parameters_random_approp = nn_model_approp(X, Y, [2,3,1],  hidden_activation = 'tanh', output_activation = 'sigmoid', num_iterations = 10000, learning_rate = 1.2, print_cost = True, plot_cost = True)

```

    Cost after 1000 iteration : 0.788206
    Cost after 2000 iteration : 0.226253
    Cost after 3000 iteration : 0.220115
    Cost after 4000 iteration : 0.217613
    Cost after 5000 iteration : 0.216162
    Cost after 6000 iteration : 0.215133
    Cost after 7000 iteration : 0.214300
    Cost after 8000 iteration : 0.213565
    Cost after 9000 iteration : 0.212881
    Cost after 10000 iteration : 0.212226



![Image](/assets/images/NeuralNetwork_2.2_VanishingExplodingGradientsProblem_files/NeuralNetwork_2.2_VanishingExplodingGradientsProblem_30_1.png)



```python
predictions_approp = predict(parameters_random_approp, X, hidden_activation = 'tanh', output_activation = 'sigmoid')

confusion_matrix_approp = metrics.confusion_matrix(Y[0], predictions_approp[0])

accuracy_approp, F1_score_approp = get_accuracy(confusion_matrix_approp)

print('When initialized weights are appropriately small values')
print('Accuracy : {}%'.format(round(accuracy_approp * 100, 2)))
print('F1 Score : {}%'.format(round(F1_score_approp, 3)))
print('-------------------------------------')

plt.figure(figsize = (12, 5))

plt.subplot(1,2,1)
sns.scatterplot(X[0,:], X[1,:], hue = Y[0])
plt.title('Actual Data')

plt.subplot(1,2,2)
sns.scatterplot(X[0,:], X[1,:], hue = predictions_approp[0])
plt.title('Appropriately small number initialized')

plt.show()
```

    When initialized weights are appropriately small values
    Accuracy : 92.4%
    F1 Score : 0.652%
    -------------------------------------



![Image](/assets/images/NeuralNetwork_2.2_VanishingExplodingGradientsProblem_files/NeuralNetwork_2.2_VanishingExplodingGradientsProblem_31_1.png)


### Summary


```python
model = ['2/3/1 NN with zero initialization', 
         '2/3/1 NN with large random initialization',
         '2/3/1 NN with appropriate small initialization']

accuracy = [accuracy_zero, accuracy_big, accuracy_approp]
comment = ['fails to break symmetry',
           'too large weights',
           'recommended method']

result = pd.DataFrame({'Model' : model,
                       'Train accuracy' : accuracy,
                       'Problem/Comment' : comment})

predictions = [Y, predictions_zero, predictions_big, predictions_approp]
method = ['Actual data', 'Zero initialization', 'Large random initialization', 'Appropriate small initialziation']

plt.figure(figsize = (20, 4))

for i,p in enumerate(predictions):
    plt.subplot(1,4,i+1)
    sns.scatterplot(X[0,:], X[1,:], hue = p[0])
    if i > 0:
        plt.xlabel('Accuracy : {}'.format(accuracy[i-1]))
    plt.title(method[i])


plt.show()

result.style.set_properties(subset=["Model",'Train accuracy', 'Problem/Comment'], **{'text-align': 'left'})
```


![Image](/assets/images/NeuralNetwork_2.2_VanishingExplodingGradientsProblem_files/NeuralNetwork_2.2_VanishingExplodingGradientsProblem_33_0.png)

|---
| Model | Train accuracy | Problem / Comment 
|-|:-|:-:|-:
| 2/3/1 NN with zero initialization | 0.5 | fails to break symmetry
| 2/3/1 NN with large random initialization |0.742 | too large weights
| 2/3/1 NN with appropriate small initialization |0.924 | recommended method
|---

<style  type="text/css" >
    #T_6c8cc518_94f9_11e9_bea6_38f9d34d80a9row0_col0 {
            text-align:  left;
        }    #T_6c8cc518_94f9_11e9_bea6_38f9d34d80a9row0_col1 {
            text-align:  left;
        }    #T_6c8cc518_94f9_11e9_bea6_38f9d34d80a9row0_col2 {
            text-align:  left;
        }    #T_6c8cc518_94f9_11e9_bea6_38f9d34d80a9row1_col0 {
            text-align:  left;
        }    #T_6c8cc518_94f9_11e9_bea6_38f9d34d80a9row1_col1 {
            text-align:  left;
        }    #T_6c8cc518_94f9_11e9_bea6_38f9d34d80a9row1_col2 {
            text-align:  left;
        }    #T_6c8cc518_94f9_11e9_bea6_38f9d34d80a9row2_col0 {
            text-align:  left;
        }    #T_6c8cc518_94f9_11e9_bea6_38f9d34d80a9row2_col1 {
            text-align:  left;
        }    #T_6c8cc518_94f9_11e9_bea6_38f9d34d80a9row2_col2 {
            text-align:  left;
        }</style><table id="T_6c8cc518_94f9_11e9_bea6_38f9d34d80a9" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Model</th>        <th class="col_heading level0 col1" >Train accuracy</th>        <th class="col_heading level0 col2" >Problem/Comment</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_6c8cc518_94f9_11e9_bea6_38f9d34d80a9level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_6c8cc518_94f9_11e9_bea6_38f9d34d80a9row0_col0" class="data row0 col0" >2/3/1 NN with zero initialization</td>
                        <td id="T_6c8cc518_94f9_11e9_bea6_38f9d34d80a9row0_col1" class="data row0 col1" >0.5</td>
                        <td id="T_6c8cc518_94f9_11e9_bea6_38f9d34d80a9row0_col2" class="data row0 col2" >fails to break symmetry</td>
            </tr>
            <tr>
                        <th id="T_6c8cc518_94f9_11e9_bea6_38f9d34d80a9level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_6c8cc518_94f9_11e9_bea6_38f9d34d80a9row1_col0" class="data row1 col0" >2/3/1 NN with large random initialization</td>
                        <td id="T_6c8cc518_94f9_11e9_bea6_38f9d34d80a9row1_col1" class="data row1 col1" >0.742</td>
                        <td id="T_6c8cc518_94f9_11e9_bea6_38f9d34d80a9row1_col2" class="data row1 col2" >too large weights</td>
            </tr>
            <tr>
                        <th id="T_6c8cc518_94f9_11e9_bea6_38f9d34d80a9level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_6c8cc518_94f9_11e9_bea6_38f9d34d80a9row2_col0" class="data row2 col0" >2/3/1 NN with appropriate small initialization</td>
                        <td id="T_6c8cc518_94f9_11e9_bea6_38f9d34d80a9row2_col1" class="data row2 col1" >0.924</td>
                        <td id="T_6c8cc518_94f9_11e9_bea6_38f9d34d80a9row2_col2" class="data row2 col2" >recommended method</td>
            </tr>
    </tbody></table>



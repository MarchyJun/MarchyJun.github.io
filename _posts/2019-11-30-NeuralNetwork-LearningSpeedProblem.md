---
layout: article
title: NeuralNetwork - Learning Speed Problem
mathjax: true
---


```python
%run NeuralNetwork_functions.ipynb
```

# Gradient descent and Learning Speed Problem

when we optimize our parameters W and b, we used gradiant descent algorithm. This gradient algorithm perform only just one update only after gradients are computed with total data. 

Let's consider training data has size m. We vectorized relevant parameters to speed our optimization procedure. 

$ Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]} \\
  A^{[l]} = g^{[l]}(Z^{[l]}) \\
  J(W, b) = \frac{1}{m}\sum_{i = 1}^{m}L(\hat{Y}^{(i)}, Y^{(i)})$
         
Even though we do vectorization, if the data size m is big, implementation of gradient descent on our entire training set wiil be very slow. So it turns out that we can get a faster algorithm if we let gradient descent start to make some progress even before we finish processing our entire training set.

# Mini-Batch Gradient descent

Let's split up our entire training set into smaller training set, which is called mini-batches. If we set mini-batch size as 1000 and our entire training data has size 10,000 , then our mini-batches will be like this: 

![Image](/assets/images/NeuralNetwork_2.3_LearningSpeedProblem_files/minibatch.png)

So, algorithm to do gradient descent with each mini batch $X^{\small\{t\small\}}, Y^{\small\{t\small\}}$ is called minibatch gradient descent rather than with entire 10000 training data, which is called batch gradient descent.

For t from 1 to 10       
                    
$\quad\:\: Z^{[1]} = W^{[1]}X^{\small\{t\small\}} + b^{[l]} \\
 \quad\:\: A^{[1]} = g^{[1]}(Z^{[1]}) \\
 \quad\:\: \vdots \\
 \quad\:\: Z^{[L]} = W^{[L]}A^{[L-1]} + b^{[L]} \\
 \quad\:\: A^{[L]} = g^{[L]}(Z^{[L]}) \\ $
  
$ \quad\:\: J(W, b) = \frac{1}{1000}\sum_{i = 1}^{1000}L(\hat{Y}^{(i)}, Y^{(i)})\: for\: X^{\small\{t\small\}}, Y^{\small\{t\small\}} $  

    Compute gradients and update parameters for l from 1 to L       
                    
$ \quad\:\: W^{[l]} = W^{[l]} - \alpha dW^{[l]}\\
  \quad\:\: b^{[l]} \:\:\:= b^{[l]} \:\:\:- \alpha db^{[l]} $  

All procedure to update parameters with 10 mini-batches is 1 epoch of training.

![Image](/assets/images/NeuralNetwork_2.3_LearningSpeedProblem_files/GDcompare.png)

We can set variance mini-batch size:   
- mini-batch size = m : batch gradient descent(BGD)
- mini-batch size = 1 : stochestic gradient descent(SGD)

Batch gradient descent is guaranteed to converge to the global minimum for convex error surfaces and to a local minimum for non-convex surfaces. However, since BGD use all training data to update parameters, it performs redundant computations for large datasets. So, BGD takes too long time per iteration. 
                   
On the other hand, since SGD use only 1 training example before updating the gradients, SGD does away with redundant computations like BGD. So SGD usually much faster than BGD, but SGD also lose some speed from vectorization. Although SGD performs frequent updates with a high variance that cause the objective function to fluctuate heavily, it has been shown that when we slowly decrease the learning rate, SGD shows the same convergence behaviour as BGD, almost certainly converging to a local or the global minimum for non-convex and convex optimization respectively. 
                             
In practice, people usually select mini-batch size somewhere between 1 and m so that we can use vectorization to speed up, and also can make progress without the process on entire training set. But the parameters will still oscillate toward the minimum like SGD rather than converge smoothly because we only use small portion of training data for updating.

So, there is guideline for selecting appropriate mini-batch size.
- if training set is small(m $\leq$ 20000) : just use batch gradient descent(BGD)
- if training set is big(m $>$ 20000) : typical mini-batch sizes are 64, 128, 256, 512 (for making some minibatch fit in CPU/GPU memory)


```python
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    np.random.seed(seed)
    m = X.shape[1]
    mini_batch_list = []
    
    # Step1 : Shuffle X, Y
    permutation = list(np.random.permutation(m))
    X_shuffled = X[:, permutation]
    Y_shuffled = Y[:, permutation].reshape(Y.shape)

    # Step2 : Partition
    num_minibatches = (m//mini_batch_size) + 1
    for t in range(num_minibatches-1):
        mini_batch_X = X_shuffled[:, t*mini_batch_size : (t+1)*mini_batch_size]
        mini_batch_Y = Y_shuffled[:, t*mini_batch_size : (t+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batch_list.append(mini_batch)
        
    if (m % mini_batch_size) != 0:
        mini_batch_X = X_shuffled[:, (num_minibatches-1) * mini_batch_size:]
        mini_batch_Y = Y_shuffled[:, (num_minibatches-1) * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batch_list.append(mini_batch)
      
    return mini_batch_list
```


```python
mini_batch_list = random_mini_batches(X, Y, 256, 42)
```


```python
print('X has shape : {}'.format(X.shape))
for i, t in enumerate(mini_batch_list):
    a, _ = mini_batch_list[i]
    print('shape of the {} mini_batch_X : {}'.format(i+1, a.shape))
    
print('--------------------------------------')    
    
print('Y has shape : {}'.format(Y.shape))    
for i, t in enumerate(mini_batch_list):
    _, b = mini_batch_list[i]
    print('shape of the {} mini_batch_Y : {}'.format(i+1, b.shape))
    
```

    X has shape : (2, 1000)
    shape of the 1 mini_batch_X : (2, 256)
    shape of the 2 mini_batch_X : (2, 256)
    shape of the 3 mini_batch_X : (2, 256)
    shape of the 4 mini_batch_X : (2, 232)
    --------------------------------------
    Y has shape : (1, 1000)
    shape of the 1 mini_batch_Y : (1, 256)
    shape of the 2 mini_batch_Y : (1, 256)
    shape of the 3 mini_batch_Y : (1, 256)
    shape of the 4 mini_batch_Y : (1, 232)


# More Sophisticated Gradient Descent

If we use mini-batch gradient descent, our optimization will oscillate toward the minimum and this up and down oscillation slows down our optimization procedure. Moreover, it prevents us from using much larger learning rate, because it we use a much larger learning rate, we might end up over shooting and end up diverging. So, the need to prevent the oscillation from getting too big forces us to use a learning rate that is not itself too large.

### Exponentially weighted moving averages

So, to build up more sophisticated optimization algorithms, we need exponentially weighted moving averages.
                          
Let's suppose we have data : $\theta_{1}  \,\dots\, \theta_{t}$              
Then, 

$ \quad\qquad\,               V_{0} = 0 \\
  \quad\theta_{1} \rightarrow V_{1} = \beta\, V_{0} + (1-\beta\,)\,\theta_{1} \\
  \quad\theta_{2} \rightarrow V_{2} = \beta\, V_{1} + (1-\beta\,)\,\theta_{2} 
                              \:\:\,= \beta\,(1- \beta\,)\,\theta_{1} + (1-\beta\,)\,\theta_{2} \\
  \quad\theta_{3} \rightarrow V_{3} = \beta\, V_{2} + (1-\beta\,)\,\theta_{3} 
                              \:\:\,= \beta^{2}\,(1- \beta\,)\,\theta_{1} + 
                                      \beta\,(1-\beta\,)\,\theta_{2} +
                                      (1-\beta\,)\,\theta_{3}\\
  \quad\vdots     \\
  \quad\theta_{t} \,\rightarrow V_{t} = \beta\, V_{t-1} + (1-\beta\,)\,\theta_{t}
                                    \:= \beta^{t-1}\,(1- \beta\,)\,\theta_{1} + 
                                        \beta^{t-2}\,(1- \beta\,)\,\theta_{2} + \dots +
                                        (1- \beta\,)\,\theta_{t} \\
$
                                        
So, $V_{t}$ is form of weighted averages of $\theta_{1} \dots \theta_{t}$
But since we set $V_{0} = 0$, our $V$ values start from very low values which is very different from actual value $\theta$. So, there is one technical detail called biase correction that can make our computation of these averages more accurately: $\frac{V_{t}}{1-\beta^{t}}$

If there are 5 datas :
                

![Image](/assets/images/NeuralNetwork_2.3_LearningSpeedProblem_files/moveaverage1.png)

$\beta\,$ is number between 0 and 1, and if $\beta\,$ becomes close to 1, it gives a lot of weight($\:\beta\:$) to the previous values and a much smaller weights($\:1-\beta\:$) to whatever we are seeing right now. So, at high value of $\beta$, exponentially weighted average adapts more slowly when the new data changes.

![Image](/assets/images/NeuralNetwork_2.3_LearningSpeedProblem_files/moveaverage2.png)

### Gradient Descent with Momentum

Optimization algorith with this exponentially weight moving average is called gradient descent with momentum that almost always works faster than the standard gradient descent algorithm. The basic idea is to compute an exponentially weighted average of our gradients, and then use that gradient to update our weights instead. In practice, people usually don't do bias correction, because after just ten iterations, our moving average will be warmed up and is no longer a bias estimate.

$ Set\:\, V_{dW} = 0,\, V_{db} = 0 $
                
$ On \:\, iteration \:\, t $

    compute dW, db on current mini-batch
$
\quad\:\: V_{dW} = \beta_{1}\,V_{dW} + (1-\,\beta_{1}\,)\,dW \\
\quad\:\: V_{db} \:\,= \beta_{1}\,V_{db} \:\,+ (1-\,\beta_{1}\,)\,db \\
$
            
$
\quad\:\: W = W - \alpha\, V_{dW} \\
\quad\:\: b \:\,\,= b \:\:\,- \alpha\, V_{db} $

In practice, people usually set $\beta_{1}$ as a number close to 0.9


![Image](/assets/images/NeuralNetwork_2.3_LearningSpeedProblem_files/momentum.png)

Let's say we computed above left gradients in the las few derivatives. Then on the vertical direction, gradients have opposite directions, so the average will be little bit close to 0. On the other hand, on the horizontal direction, all gradients are pointing to the right, so average will be big. Therefore, gradient descent with momentum ends up just taking steps that are much smaller oscillations, but are moving quickly in the horizontal direction.


```python
def initialize_momentum(parameters):
    L = len(parameters) // 2
    v = {}
    
    for l in range(L):
        v['dW' + str(l+1)] = np.zeros(parameters['W'+str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters['b'+str(l+1)].shape)
        
    return v
```


```python
def update_parameters_with_momentum(parameters, grads, v, beta1, learning_rate):
    L = len(parameters) // 2
    
    for l in range(L):
        v["dW" + str(l+1)] = beta1*v['dW'+str(l+1)] + (1-beta1)*grads['dW'+str(l+1)]
        v["db" + str(l+1)] = beta1*v['db'+str(l+1)] + (1-beta1)*grads['db'+str(l+1)]
        
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v["db" + str(l+1)]
        
    return parameters, v
```

### Gradient Descent with Root Mean Square prop

There is another algorithm called RMSprop, which can also speed up gradient descent. 

![Image](/assets/images/NeuralNetwork_2.3_LearningSpeedProblem_files/rms1.png)

Let's consider above example. If we check the left image, we want to slow down our learning on vertical direction, while we want to speed up our learning on horizontal direction. So, if we update our parameters like this:

$ Set\:\, S_{dW} = 0,\, S_{db} = 0 $
                
$ On \:\, iteration \:\, t $

    compute dW, db on current mini-batch
$
\quad\:\: S_{dW} = \beta_{2}\,S_{dW} + (1-\,\beta_{2}\,)\,(dW)^{2} \\
\quad\:\: S_{db} \:\,= \beta_{2}\,S_{db} \:\,+ (1-\,\beta_{2}\,)\,(db)^{2} \\
$
            
$
\quad\:\: W = W - \alpha\, \frac{dW}{\sqrt{S_{dW}} + \epsilon} \\
\quad\:\: b \:\,\,= b \:\:\,- \alpha\, \frac{db}{\sqrt{S_{db}} + \epsilon} $

$\epsilon\:$ is for making our algorithm do not divide $dW$ and $db$ by 0. People usually set $\epsilon$ as $10^{-8}$

If we check above image, function is sloped much more steeply in the vertical direction than in the horizontal direction:
- $db$ is relatively large. Since our update in vertical direction is divided by $db$ which is relatively large number, it helps slow down the learning in the b direction.
- $dW$ is relatively small. Since our update in horizontal direction is divided by $dW$ which is relatively small number, so it helps speed up the learning in the W direction.

As a result, RMSprop might damp out the oscillations, so we can use a larger learning rate alpha and get faster learning without diverging in the vertical direction.


```python
def initialize_rms(parameters):
    L = len(parameters) // 2
    s = {}
    
    for l in range(L):
        s['dW' + str(l+1)] = np.zeros(parameters['W'+str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(parameters['b'+str(l+1)].shape)
        
    return s
```


```python
def update_parameters_with_rms(parameters, grads, s, beta2, learning_rate):
    L = len(parameters) // 2
    
    for l in range(L):
        s["dW" + str(l+1)] = beta2*s['dW'+str(l+1)] + (1-beta2)*grads['dW'+str(l+1)]**2
        s["db" + str(l+1)] = beta2*s['db'+str(l+1)] + (1-beta2)*grads['db'+str(l+1)]**2
        
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)] / (s["dW" + str(l+1)]**(1/2) + 10**(-8))   
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)] / (s["db" + str(l+1)]**(1/2) + 10**(-8))
        
    return parameters, s
```

### Gradient Descent with ADAptive Moment estimation

Adam optimization algorithm is basically taking momentum and rms prop and putting them together. This adam optimization algorithm is one of those rare algorithms that has been shown to work well across a wide range of deep learning architectures.

$ Set\:\, V_{dW} = 0,\, V_{db} = 0,\, S_{dW} = 0,\, S_{db} = 0 $
                
$ On \:\, iteration \:\, t $

    compute dW, db on current mini-batch
$
\quad\:\: V_{dW} = \beta_{1}\,V_{dW} + (1-\,\beta_{1}\,)\,dW 
\qquad\:\: V^{correct}_{dW} = \frac{V_{dW}}{(1-\beta_{1}^{t})} \\
\quad\:\: V_{db} \:\,= \beta_{1}\,V_{db} \:\,+ (1-\,\beta_{1}\,)\,db 
\qquad\:\:\:\:\, V^{correct}_{db} = \frac{V_{db}}{(1-\beta_{1}^{t})} \\
$

$
\quad\:\: S_{dW} = \beta_{2}\,S_{dW} + (1-\,\beta_{2}\,)\,(dW)^{2} 
\quad\:\: S^{correct}_{dW} = \frac{S_{dW}}{(1-\beta_{2}^{t})}\\
\quad\:\: S_{db} \:\,= \beta_{2}\,S_{db} \:\,+ (1-\,\beta_{2}\,)\,(db)^{2} 
\qquad S^{correct}_{db} = \frac{S_{db}}{(1-\beta_{2}^{t})} \\
$
         
$
\quad\:\: W = W - \alpha\, \frac{V^{correct}_{dW}}{\sqrt{S^{correct}_{dW}} + \epsilon} \\
\quad\:\: b \:\,\,= b \:\:\,- \alpha\, \frac{V^{correct}_{db}}{\sqrt{S^{correct}_{db}} + \epsilon} $

When use adam optimization, people usually just use default hyper parameter and try to tune only $\alpha$:
- $\beta_{1}$ : 0.9
- $\beta_{2}$ : 0.999
- $\epsilon\:\:$ : $10^{-8}$


```python
def initialize_adam(parameters):
    L = len(parameters) // 2
    v = {}
    s = {}
    
    for l in range(L):
        v['dW' + str(l+1)] = np.zeros(parameters['W'+str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters['b'+str(l+1)].shape)
        s['dW' + str(l+1)] = np.zeros(parameters['W'+str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(parameters['b'+str(l+1)].shape)
        
    return v, s
```


```python
def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon):
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}
    
    for l in range(L):
        v["dW" + str(l+1)] = beta1 * v['dW'+str(l+1)] + (1 - beta1) * grads['dW'+str(l+1)]
        v["db" + str(l+1)] = beta1 * v['db'+str(l+1)] + (1 - beta1) * grads['db'+str(l+1)]
        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1 - beta1**t)
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1 - beta1**t)
        
        s["dW" + str(l+1)] = beta2*s['dW'+str(l+1)] + (1-beta2)*grads['dW'+str(l+1)]**2
        s["db" + str(l+1)] = beta2*s['db'+str(l+1)] + (1-beta2)*grads['db'+str(l+1)]**2
        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1 - beta2**t)
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1 - beta2**t)
        
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v_corrected["dW" + str(l+1)]/(s_corrected["dW" + str(l+1)]**(1/2) + epsilon)
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v_corrected["db" + str(l+1)]/(s_corrected["db" + str(l+1)]**(1/2) + epsilon)
        
    return parameters, v, s
```

# Compare Different Gradient Descent Algorithms

Let's consider following training set and classification problem.


```python
n_samples = 1000
noisy_circles = datasets.make_moons(n_samples = n_samples, noise=.2, random_state=42)


X = noisy_circles[0].T
Y = noisy_circles[1].reshape([1,1000])

print('Train data X shape:' + str(X.shape))
print('Train data Y shape:' + str(Y.shape))

sns.scatterplot(X[0,:], X[1,:], hue = Y[0])
plt.show()
```

    Train data X shape:(2, 1000)
    Train data Y shape:(1, 1000)




![Image](/assets/images/NeuralNetwork_2.3_LearningSpeedProblem_files/NeuralNetwork_2.3_LearningSpeedProblem_41_1.png)


Let's build neural network with 2/5/2/1


```python
layer_dims = [X.shape[0], 5, 2, 1]
```


```python
def nn_model_optimization(X, Y, layer_dims, hidden_activation = 'sigmoid', output_activation = 'sigmoid', optimizer = 'adam', learning_rate = 0.0007, mini_batch_size = 64, num_iterations = 10000, 
                          beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, print_cost = True, plot_cost = True):
    cost_list = []
    t = 0.001
    seed = 10
    
    # Check network parameters
    layer_dims = check_network_parameter(X, Y, layer_dims)
    
    # Initialize parameters
    parameters = initialize_parameters_random_approp(layer_dims, hidden_activation)
    
    # Initialize the optimizer
    if optimizer == 'gd':
        pass
    elif optimizer == 'momentum':
        v = initialize_momentum(parameters)
    elif optimizer == 'rms':
        s = initialize_rms(parameters)
    elif optimizer == 'adam':
        v, s = initialize_adam(parameters)
        
    # Optimization loop
    for i in range(num_iterations):
        seed = seed + 1
        mini_batch_list = random_mini_batches(X, Y, mini_batch_size, seed)
        
        for minibatch in mini_batch_list:
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch
            
            # Forward propagation
            cache = forward_propagation(minibatch_X, parameters, hidden_activation, output_activation)
            AL = cache['AL']
            
            # Compute cost
            cost = compute_cost(Y_h = AL, Y = minibatch_Y, output_activation = output_activation)
            
            # Backward propagation
            grads = backward_propagation(minibatch_X, minibatch_Y, parameters, cache, hidden_activation, output_activation)
            
            # Update parameters
            if optimizer == 'gd':
                parameters = update_parameters(parameters, grads, learning_rate)
            elif optimizer == 'momentum':
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta1, learning_rate)
            elif optimizer == 'rms':
                parameters, s = update_parameters_with_rms(parameters, grads, s, beta2, learning_rate)
            elif optimizer == 'adam':
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)
        
        if print_cost and i % 1000 == 0:
            print('Cost after {} epoch : {}'.format(i+1000, cost))
        if i % 100 == 0:
            cost_list.append(cost)
            
    if plot_cost:
        plt.plot(cost_list)
        plt.xlabel('Iterations (per 100)')
        plt.ylabel('cost')
        plt.title('Learning rate = ' + str(learning_rate))
        plt.show()
            
    return parameters
            
```

### Mini-batch Gradient descent


```python
tick = time.time()
parameters_gd = nn_model_optimization(X, Y, layer_dims, optimizer = 'gd')
tock = time.time()
gd_time = tock - tick
print('Spent time for training : {} sec'.format(gd_time))
```

    Cost after 1000 epoch : 0.6767814069494891
    Cost after 2000 epoch : 0.691419340389962
    Cost after 3000 epoch : 0.6872272813583813
    Cost after 4000 epoch : 0.6822692576873256
    Cost after 5000 epoch : 0.6667325920400261
    Cost after 6000 epoch : 0.6410002036279266
    Cost after 7000 epoch : 0.6070760907518016
    Cost after 8000 epoch : 0.5721262551936012
    Cost after 9000 epoch : 0.4827610061461587
    Cost after 10000 epoch : 0.45178227409834265




![Image](/assets/images/NeuralNetwork_2.3_LearningSpeedProblem_files/NeuralNetwork_2.3_LearningSpeedProblem_46_1.png)


    Spent time for training : 24.979933261871338



```python
predictions_gd = predict(parameters_gd, X, hidden_activation = 'sigmoid', output_activation = 'sigmoid')

confusion_matrix_gd = metrics.confusion_matrix(Y[0], predictions_gd[0])

accuracy_gd, F1_score_gd = get_accuracy(confusion_matrix_gd)

print('When initialized weights are big')
print('Accuracy : {}%'.format(round(accuracy_gd * 100, 4)))
print('F1 Score : {}%'.format(round(F1_score_gd, 3)))
print('-------------------------------------')

plt.figure(figsize = (12,5))

plt.subplot(1,2,1)
sns.scatterplot(X[0,:], X[1,:], hue = Y[0])
plt.title('Actual Data')

plt.subplot(1,2,2)
sns.scatterplot(X[0,:], X[1,:], hue = predictions_gd[0])
plt.title('Mini-batch Gradient descent')

plt.show()
```

    When initialized weights are big
    Accuracy : 84.8%
    F1 Score : 0.629%
    -------------------------------------




![Image](/assets/images/NeuralNetwork_2.3_LearningSpeedProblem_files/NeuralNetwork_2.3_LearningSpeedProblem_47_1.png)


### Mini-batch with Momentum


```python
tick = time.time()
parameters_momentum = nn_model_optimization(X, Y, layer_dims, optimizer = 'momentum')
tock = time.time()
momentum_time = tock - tick
print('Spent time for training : {} sec'.format(momentum_time))

```

    Cost after 1000 epoch : 0.6767638934795258
    Cost after 2000 epoch : 0.6914194731864537
    Cost after 3000 epoch : 0.6872317544551713
    Cost after 4000 epoch : 0.6822804610936675
    Cost after 5000 epoch : 0.6667487170861078
    Cost after 6000 epoch : 0.6410439938717618
    Cost after 7000 epoch : 0.607147553882122
    Cost after 8000 epoch : 0.5722113850794566
    Cost after 9000 epoch : 0.48287968849422025
    Cost after 10000 epoch : 0.4518662313213742




![Image](/assets/images/NeuralNetwork_2.3_LearningSpeedProblem_files/NeuralNetwork_2.3_LearningSpeedProblem_49_1.png)


    Spent time for training : 30.84908699989319 sec



```python
predictions_momentum = predict(parameters_momentum, X, hidden_activation = 'sigmoid', output_activation = 'sigmoid')

confusion_matrix_momentum = metrics.confusion_matrix(Y[0], predictions_momentum[0])

accuracy_momentum, F1_score_momentum = get_accuracy(confusion_matrix_momentum)

print('When initialized weights are big')
print('Accuracy : {}%'.format(round(accuracy_momentum * 100, 4)))
print('F1 Score : {}%'.format(round(F1_score_momentum, 3)))
print('-------------------------------------')

plt.figure(figsize = (12,5))

plt.subplot(1,2,1)
sns.scatterplot(X[0,:], X[1,:], hue = Y[0])
plt.title('Actual Data')

plt.subplot(1,2,2)
sns.scatterplot(X[0,:], X[1,:], hue = predictions_momentum[0])
plt.title('Mini-batch with Momentum')

plt.show()
```

    When initialized weights are big
    Accuracy : 84.8%
    F1 Score : 0.629%
    -------------------------------------




![Image](/assets/images/NeuralNetwork_2.3_LearningSpeedProblem_files/NeuralNetwork_2.3_LearningSpeedProblem_50_1.png)


### Mini-batch with RMSprop


```python
tick = time.time()
parameters_rms = nn_model_optimization(X, Y, layer_dims, optimizer = 'rms')
tock = time.time()
rms_time = tock - tick
print('Spent time for training : {} sec'.format(rms_time))
```

    Cost after 1000 epoch : 0.6823452094738172
    Cost after 2000 epoch : 0.28998047860010107
    Cost after 3000 epoch : 0.19032531213287335
    Cost after 4000 epoch : 0.12125955533816393
    Cost after 5000 epoch : 0.17974415553528725
    Cost after 6000 epoch : 0.026466024595121807
    Cost after 7000 epoch : 0.024551621026829465
    Cost after 8000 epoch : 0.08440196128227671
    Cost after 9000 epoch : 0.17558148612597124
    Cost after 10000 epoch : 0.10959283827640018



![Image](/assets/images/NeuralNetwork_2.3_LearningSpeedProblem_files/NeuralNetwork_2.3_LearningSpeedProblem_52_1.png)


    Spent time for training : 32.749738931655884 sec



```python
predictions_rms = predict(parameters_rms, X, hidden_activation = 'sigmoid', output_activation = 'sigmoid')

confusion_matrix_rms = metrics.confusion_matrix(Y[0], predictions_rms[0])

accuracy_rms, F1_score_rms = get_accuracy(confusion_matrix_rms)

print('When initialized weights are big')
print('Accuracy : {}%'.format(round(accuracy_rms * 100, 4)))
print('F1 Score : {}%'.format(round(F1_score_rms, 3)))
print('-------------------------------------')

plt.figure(figsize = (12,5))

plt.subplot(1,2,1)
sns.scatterplot(X[0,:], X[1,:], hue = Y[0])
plt.title('Actual Data')

plt.subplot(1,2,2)
sns.scatterplot(X[0,:], X[1,:], hue = predictions_rms[0])
plt.title('Mini-batch with RMSprop')

plt.show()
```

    When initialized weights are big
    Accuracy : 97.5%
    F1 Score : 0.661%
    -------------------------------------




![Image](/assets/images/NeuralNetwork_2.3_LearningSpeedProblem_files/NeuralNetwork_2.3_LearningSpeedProblem_53_1.png)


### Mini-batch with Adam


```python
tick = time.time()
parameters_adam = nn_model_optimization(X, Y, layer_dims, optimizer = 'adam')
tock = time.time()
adam_time = tock - tick
print('Spent time for training : {} sec'.format(adam_time))
```

    Cost after 1000 epoch : 0.66753519074358
    Cost after 2000 epoch : 0.009234517163738918
    Cost after 3000 epoch : 0.076900415100215
    Cost after 4000 epoch : 0.0784744912761556
    Cost after 5000 epoch : 0.0859538195012183
    Cost after 6000 epoch : 0.012500261583350363
    Cost after 7000 epoch : 0.010566665497110984
    Cost after 8000 epoch : 0.06655156682968585
    Cost after 9000 epoch : 0.12237572293239748
    Cost after 10000 epoch : 0.07096588051939703




![Image](/assets/images/NeuralNetwork_2.3_LearningSpeedProblem_files/NeuralNetwork_2.3_LearningSpeedProblem_55_1.png)


    Spent time for training : 41.99417805671692 sec



```python
predictions_adam = predict(parameters_adam, X, hidden_activation = 'sigmoid', output_activation = 'sigmoid')

confusion_matrix_adam = metrics.confusion_matrix(Y[0], predictions_adam[0])

accuracy_adam, F1_score_adam = get_accuracy(confusion_matrix_adam)

print('When initialized weights are big')
print('Accuracy : {}%'.format(round(accuracy_adam * 100, 4)))
print('F1 Score : {}%'.format(round(F1_score_adam, 3)))
print('-------------------------------------')

plt.figure(figsize = (12,5))

plt.subplot(1,2,1)
sns.scatterplot(X[0,:], X[1,:], hue = Y[0])
plt.title('Actual Data')

plt.subplot(1,2,2)
sns.scatterplot(X[0,:], X[1,:], hue = predictions_adam[0])
plt.title('Mini-batch with Adam')

plt.show()
```

    When initialized weights are big
    Accuracy : 97.5%
    F1 Score : 0.66%
    -------------------------------------




![Image](/assets/images/NeuralNetwork_2.3_LearningSpeedProblem_files/NeuralNetwork_2.3_LearningSpeedProblem_56_1.png)


### Summary


```python
optimization = ['Mini-batch with GD', 
                'Mini-batch with Momentum',
                'Mini-batch with RMSprop',
                'Mini-batch with Adam']

accuracy = [accuracy_gd, accuracy_momentum, accuracy_rms, accuracy_adam]
cost_shape = ['oscillations',
              'oscillations',
              'smoother',
              'smoother']

result = pd.DataFrame({'Optimization Method' : optimization,
                       'Train accuracy' : accuracy,
                       'Cost Shape' : cost_shape})



predictions = [Y, predictions_gd, predictions_momentum, predictions_rms, predictions_adam]
method = ['Actual data', 'Mini-batch with GD', 'Mini-batch with Momentum', 'Mini-batch with RMSprop', 'Mini-batch with Adam']

plt.figure(figsize = (20, 4))

for i,p in enumerate(predictions):
    plt.subplot(1,5,i+1)
    sns.scatterplot(X[0,:], X[1,:], hue = p[0])
    if i > 0:
        plt.xlabel('Accuracy : {}'.format(accuracy[i-1]))
    plt.title(method[i])

plt.show()
result.style.set_properties(subset=["Optimization Method",'Train accuracy', 'Cost Shape'], **{'text-align': 'left'})

```



![Image](/assets/images/NeuralNetwork_2.3_LearningSpeedProblem_files/NeuralNetwork_2.3_LearningSpeedProblem_58_0.png)





<style  type="text/css" >
    #T_b32cf062_94f2_11e9_90a2_38f9d34d80a9row0_col0 {
            text-align:  left;
        }    #T_b32cf062_94f2_11e9_90a2_38f9d34d80a9row0_col1 {
            text-align:  left;
        }    #T_b32cf062_94f2_11e9_90a2_38f9d34d80a9row0_col2 {
            text-align:  left;
        }    #T_b32cf062_94f2_11e9_90a2_38f9d34d80a9row1_col0 {
            text-align:  left;
        }    #T_b32cf062_94f2_11e9_90a2_38f9d34d80a9row1_col1 {
            text-align:  left;
        }    #T_b32cf062_94f2_11e9_90a2_38f9d34d80a9row1_col2 {
            text-align:  left;
        }    #T_b32cf062_94f2_11e9_90a2_38f9d34d80a9row2_col0 {
            text-align:  left;
        }    #T_b32cf062_94f2_11e9_90a2_38f9d34d80a9row2_col1 {
            text-align:  left;
        }    #T_b32cf062_94f2_11e9_90a2_38f9d34d80a9row2_col2 {
            text-align:  left;
        }    #T_b32cf062_94f2_11e9_90a2_38f9d34d80a9row3_col0 {
            text-align:  left;
        }    #T_b32cf062_94f2_11e9_90a2_38f9d34d80a9row3_col1 {
            text-align:  left;
        }    #T_b32cf062_94f2_11e9_90a2_38f9d34d80a9row3_col2 {
            text-align:  left;
        }</style><table id="T_b32cf062_94f2_11e9_90a2_38f9d34d80a9" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Optimization Method</th>        <th class="col_heading level0 col1" >Train accuracy</th>        <th class="col_heading level0 col2" >Cost Shape</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_b32cf062_94f2_11e9_90a2_38f9d34d80a9level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_b32cf062_94f2_11e9_90a2_38f9d34d80a9row0_col0" class="data row0 col0" >Mini-batch with GD</td>
                        <td id="T_b32cf062_94f2_11e9_90a2_38f9d34d80a9row0_col1" class="data row0 col1" >0.848</td>
                        <td id="T_b32cf062_94f2_11e9_90a2_38f9d34d80a9row0_col2" class="data row0 col2" >oscillations</td>
            </tr>
            <tr>
                        <th id="T_b32cf062_94f2_11e9_90a2_38f9d34d80a9level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_b32cf062_94f2_11e9_90a2_38f9d34d80a9row1_col0" class="data row1 col0" >Mini-batch with Momentum</td>
                        <td id="T_b32cf062_94f2_11e9_90a2_38f9d34d80a9row1_col1" class="data row1 col1" >0.848</td>
                        <td id="T_b32cf062_94f2_11e9_90a2_38f9d34d80a9row1_col2" class="data row1 col2" >oscillations</td>
            </tr>
            <tr>
                        <th id="T_b32cf062_94f2_11e9_90a2_38f9d34d80a9level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_b32cf062_94f2_11e9_90a2_38f9d34d80a9row2_col0" class="data row2 col0" >Mini-batch with RMSprop</td>
                        <td id="T_b32cf062_94f2_11e9_90a2_38f9d34d80a9row2_col1" class="data row2 col1" >0.975</td>
                        <td id="T_b32cf062_94f2_11e9_90a2_38f9d34d80a9row2_col2" class="data row2 col2" >smoother</td>
            </tr>
            <tr>
                        <th id="T_b32cf062_94f2_11e9_90a2_38f9d34d80a9level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_b32cf062_94f2_11e9_90a2_38f9d34d80a9row3_col0" class="data row3 col0" >Mini-batch with Adam</td>
                        <td id="T_b32cf062_94f2_11e9_90a2_38f9d34d80a9row3_col1" class="data row3 col1" >0.975</td>
                        <td id="T_b32cf062_94f2_11e9_90a2_38f9d34d80a9row3_col2" class="data row3 col2" >smoother</td>
            </tr>
    </tbody></table>



---
layout: article
title: MLP - Overfitting Problem
mathjax: true
---

# Overfitting Problem

If our model is over fitting our training data, test accuracy will be very lower than test accuracy. This is because our model is too complex, so we have to reduce our model's complexity.

![Image](/assets/images/NeuralNetwork_2.1_OverfittingProblem_files/fitting.png)

One of the first things we have to try is regularization. The other way to address high variance is to get more training data.

# Regularization

### 1. L1 & L2 Regularization

L1 & L2 regularization methods are to add regularization term to cost function to restrict our parameters. We usually restrict only W parameters, because W is a high dimensional parameter vector whereas b is just small dimensional parameter vector. It means regularization on b will not make much of difference because almost all parameters are in W vector rather than b vector. 


- L1 regularization : $$ J(W^{[1]}, b^{[1]} \:\dots\: W^{[L]}, b^{[L]} ) = \frac{1}{m}\sum_{i = 1}^{m}L(\hat{Y^{(i)}}, Y^{(i)}) + \frac{\lambda}{2m} \sum_{l = 1}^{L}
\begin{Vmatrix} W^{[l]} \end{Vmatrix}_{1}, \:\: where \begin{Vmatrix} W^{[l]} \end{Vmatrix}_{1} = \sum_{i = 1}^{n^{[l-1]}}\sum_{j=1}^{n^{[l]}}\begin{vmatrix}W^{[l]}_{ij}\end{vmatrix} \\ $$
                  
- L2 regularization : $$ J(W^{[1]}, b^{[1]} \:\dots\: W^{[L]}, b^{[L]} ) = \frac{1}{m}\sum_{i = 1}^{m}L(\hat{Y^{(i)}}, Y^{(i)}) + \frac{\lambda}{2m} \sum_{l = 1}^{L}
\begin{Vmatrix} W^{[l]} \end{Vmatrix}_{2}^{2}, \:\: where \begin{Vmatrix} W^{[l]} \end{Vmatrix}_{2}^{2} = \sum_{i = 1}^{n^{[l-1]}}\sum_{j=1}^{n^{[l]}}(W^{[l]}_{ij})^2 \\
$$
              
- $$\lambda\;$$is regularization parameter, so this is another hyperparameter that we have to tune.

In practice, L1 regularization helps only a little bit, so when people train networks, L2 regularization is just used much more often. Let's consider just L2 regularization. Since we add some term in our cots function, we have to add derivate of this term when we update our parameter W.
$$ \,\\
W^{[l]} = W^{[l]} - \alpha \frac{\partial J(W^{[1]}, b^{[1]} \:\dots\: W^{[L]}, b^{[L]} )}{\partial W^{[l]}} \\ \quad\:\:\,\, 
        = W^{[l]} - \alpha dW^{[l]} -\alpha \frac{\lambda}{m} W^{[l]} $$


Let's consider following simple case without parameter b to understand why L2 regularization term help us to solve overfitting problem.

![Image](/assets/images/NeuralNetwork_2.1_OverfittingProblem_files/L2costfunction.png)


Our cost function $J_{total}$ is sum of $J_{original}$ and $J_{L2}$. So, $W$ that minimize $J_{total}$ is located in where $J_
{original}$ and $J_{L2}$ meet. This shows that our L2 regularization term set W to be reasonably close to zero for a lot of hidden nodes and it means our neural network is simplified and become much smaller.

In other aspect, let's assume our activation function is s-shaped funtion. Then, if $\alpha$ is getting bigger, W is getting smaller and Z is also gettin smaller, so g(Z) will be roughly linear. In this way, every layer will be roughly linear, so model can't fit very complicated decision which cause our model to overfit to dataset.

### 2. Dropout Regularization

Dropout is a widely used regularization technique that is specific to deep learning. It randomly shuts down some neurons and remove all the outgoing things from that neurons as well in each iteration. So, we end up with a much smaller, diminished network.

![Image](/assets/images/NeuralNetwork_2.1_OverfittingProblem_files/dropout.png)

Under drop out regularization, each neuron in $[l]$ layer can't rely on only one neuron in $[l-1]$ layer because any features could go away. So, every neuron in $[l]$ layer spread weights to each of the neurons in $[l-1]$ layer. And by spreading all weights, this will tend to have an effect of shrinking the squared norm of the weights. And so, similar to L2 regularization.

### 3. Data Augmentation

If there is overfittin problem in our model, then getting more training data can help, but getting more training data can be expensive and sometimes we just can't get more data. But what we can do is augment our training data like this :

![Image](/assets/images/NeuralNetwork_2.1_OverfittingProblem_files/dataaugmentation.png)

These extra fake training examples might do not add as much information as brand new independent examples. But because we can do this almost for free, this can be an inexpensive way to give our algorithm more data and therefore sort of regularize it and reduce over fitting.

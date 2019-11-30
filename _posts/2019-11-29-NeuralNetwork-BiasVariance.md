---
layout: article
title: NeuralNetwork - Bias & Variance
mathjax: true
articles:
  data_source: mlp
---


# Bias Variance Trade-off

In machine learning and deep learning, we can check the performance of our models by bias and variance. Let's consider the cost function MSE that is commonly used in the regression problem. MSE is one cost function that is used to measure our model performance, and it can be decomposed as bias term and variance term.         
$$ \, \\
MSE = \frac{1}{n} \sum_{i=1}^{n} (Y_{i} - \hat{Y_{i}})^{2} \\ \qquad\:\: 
    = E[(Y - \hat{Y})^{2}] = Var(\hat{Y}) + Bias(\hat{Y})^{2} + Var(e) $$

Here, Var(e) is a inherent limit of the data that we can't reduce this error with any model. So this is also called irreducible error or noise. On the other hand, the bias and variance can be reduced depending on what model we use. 
Bias is related to training data. High bias mean our model is not accuracy with train data, and low bias mean our model is quite accuracy with train data. Variance is related to the performance of validation or test data. High variance means our model overfits to the train data, so if we put new datas, that don't be used as training, to our model, then our model performance has very low accuracy. In the following images, blue dots mean our model's predictions and the red center of the circle is a acutal value.

![Image](/assets/images/NeuralNetwork_1.2_BiasVariance_files/biasvariance.png)

So best model is that have lowest bias and lowest variance. But in traditional supervised learning problem, there is bias variance trade off. It means it is impossible for models to accurately capture the regularities in its training data and to generalize well to unseen data. So This is general ways that we can reduce bias or variance of our model by using.

- High bias : It is related to train data performance.     
    - Use bigger network
    - Try longer
    - Check different NN architecture
    - Keep doing until fit train set well
    
- High variance : It is related to validation or test set performance.
    - Get more data
    - Use regularization ( reduce overfiting )
    - Check different NN architecture

But in the modern deep learning, big data era, getting a bigger network almost always just reduces the bias without necessarily hurtin the variance as long as we regularize appropriately. And getting more data pretty muchy always reduces the variance and doesn't hurt the bias much.

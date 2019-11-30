
# Three Major Architectures of Deep Networks


1. Unsupervised Pretrained Networks (UPNs)
2. Convolutional Neural Networks (CNNs)
3. Sequence Model
4. Deep reinforcement learning


## 1. Unsupervised Pretrained Networks (UPNs)


1.1 Autoencoders                
1.2 Deep Belief Networks(DBNs)                
1.3 Generative Adversarial Networks(GANs)             

### 1.1 Autoencoders

Autoencoders are used to learn compressed representations of datasets. Typically, we use them to reduce dataset's dimensionality by training the network to ignore sigmal 'noise'. The output of the autoencoder network is a reconstruction of the input data in the most efficient form. There are few reasons doing this may be useful: dimensionality reduction can decrease training time and using latent feature representations may enhance model performance.

This kind of network is composed of two parts:
- 1. Encoder: This is the part of the network that compreseed the input into a latent-space representation. It can be represented by an encoding function h = f(x)
- 2. Decoder: This part aims to reconstruct the input from the latent space representation. It can be represented by a decoding function r = g(h)

### 1.2 Deep Belief Networks(DBNs)

To understand Deep Belief Networks, we need to understand two important caveat of DBN
- 1. Belief Net 
- 2. Restriced Boltzmann Machine (RBM)

### 1.3 Generative Adversarial Networks (GANs)


It is generation model, that create plausible new data similar to the actual data. 

![title](Images/GAN1.png)

In the generation process, GANs adversarially develop two different models with competing. One is generator and the other is discriminator. The purpose of generator is to deceive discriminator by creating palusible data, and the purpose of discriminator is to discriminate the actual data from the fake data created by the generator.

Application of GANs:
1. GANs for Image Editing(https://arxiv.org/pdf/1611.06355.pdf) : 
![title](Images/GAN2_ImageEditing.png)

2. Using GANs for Security
3. Generating Data with GANs(https://arxiv.org/pdf/1612.07828.pdf) :
![title](Images/GAN3_GeneratingData.png)
4. GANs for Attention Prediction (https://arxiv.org/pdf/1701.01081) : When we see an image, we tend to focus on a particular part rather than the entire image as a whole. This is called attention and is an important human trait. Knowing where a person would look beforehand would certainly be a useful feature for business, as they can optimize and position their products better.
5. GANs for 3D Object Generation (https://github.com/maxorange/pix2vox) : GANs are quite popular in the gaming industry, because GANs can be used to automate the entire process of recreating 3D avatars and backgrounds.

## 2. Convolutional Neural Network ( CNN )

CNN is from cortex study of our brain, and it has used to image classification problems from 1980s. MLP has problem in image classification because it has many parameters to optimize in big image. The important components of CNN are convolutional layer. This layer has a filter that extracts features, an activation function that convert filtered features to non-linear values.

![title](Images/cnn.png)

Filter image can be represented as matrix and be applied like this: input image matirx * filter matrix = result matrix. If input image has features like filter, then result will have matrix that have high values, otherwise result will have matrix that mostly have 0. So, filter detect whether input image has features like filter. We apply many filters to each input images. 

![title](Images/cnn_filter.png)

When apply filters, since filters have smaller size than input images, we apply filters every few strides, and the result matrix after applying filter is called feature map or activation map.

![title](Images/cnn_stride.gif)

After applying filter, our result matrixs become smaller than original input images, and this means there will be information loss. Since we will apply several convolutional layer, we don't want information loss. So we artificially increase the size of our input images so that the result size becomes same as the original input size. It is called pedding.

![title](Images/cnn_process.png)

After extract features, we apply activation function(usually use relu function) to featured maps.

Afther then, since we don't need all features, we use sub sampling or pooling that artificially reduce size of featured maps. There are many methods to do pooling, but max pooling is usually used.

![title](Images/cnn_maxpooling.png)

Filters, activation function(ReLU), and pooling are repeatedly combined to extract features. After extract features, we input these features to our deep neural network to classify our images.

CNN is well suited to object recognition with images and consistently top image classification competitions. They can identify many aspects of visual data. CNNs tend to be most useful when there is some structure to the input data. So, CNNs also have been used in other tasks such as analyzing words and sound.

Developmet :                 
2014: R-CNN - An early application of CNNs to object detection.                  
2015: Fast R-CNN - Speeding up and simplifying R-CNN                        
2016: Faster R-CNN - Speeding up region proposal                          
2017: Mask R-CNN - Extending faster R-CNN for pixel level segmentation.

## 3. Sequence Model

3.1 Recurrent Neural Network                 
3.2 Recursive Neural Network

Sequence data are are inherently ordered and context sensitive where later values depend on previous ones. So, our standard neural network have some problems with handling these kind of sequence data:
- In sequence data, input and output can be different lengths in different examples. So our standard neural network can't handle these data.
- Sequence data don't share featres learned across different positions of text.

So, in many domains, we need other models that can handle sequence data.

Sequence model Example :
- speech recognition : audio clip - text transcript (many to many)
- music generation : empty or integer - music sequence (one to many) 
- sentiment classification : phrase(about movie) - sentiment(about rating) (many to one)
- DNA sequence analysis : DNA information sequence - correspondent protein (many to one)
- machine translation : sentence - sentence (many to many)
- video activity recognition : sequence of video frame - recognize the activity (many to one)
- name entity recongintion : sentence - identify people name in the sentence (many to many)

Recursive neural network is similar to recurrent network in that it can handle sequence data. The primary difference is that recursive network have the ability to model the hierarchial structures in the training dataset.




```python

```



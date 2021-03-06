# ASU-CSE569

A.	Bayesian Decision Theory & The Curse of Dimensionality

In the first part of this project, we will implement a minimum error rate classifier on a 2-class problem. In this project, we use a modified subset of the MNIST data containing images of the digit “0” and the digit “1” for ease of implementation and learning. 
Our dataset consists of 5923 training images for digit “0” and 6742 training images for digit “1”. Similarly, for testing purposes, we have 980 and 1135 images of digits “0” and “1” respectively. These images are stored as a matrix of the dimension 28 x 28 each. 
Our final goal is to perform minimum error classification using Bayesian Decision Theory. However, it is difficult to use Bayesian Decision Theory in a 784-d space. Therefore, it is essential to perform dimensionality reduction first. Another essential requirement for performing minimum error classification is to know the underlying parameters of the distributions for each class. After visualizing the distribution, we’ll use maximum likelihood estimation technique to estimate the underlying parameters of these distributions. Finally, we’ll use the Bayesian Decision Theory for optimal classification. 


B.	Shallow & Deep Learning in Neural Networks

The field of artificial neural networks have gained quite a popularity for long time now. Neural Networks like Multi-layered Perceptrons and Deep Neural Networks like Convolutional Neural Networks are called Universal Learners because they can learn any underlying non-linear functions of the data samples. In this project we will explore the workings of Multi-layered Perceptron by implementing from scratch. We will use a custom dataset to learn and test our neural network. In the later sections, we once again explore the MNIST dataset using CNNs. 
This part of the project again comprises of two components. First is the implementation, training, and testing of a “shallow” 3-layer MLP. We perform a 2-class classification task using a dataset consisting of 2000 training samples for each class. The first 1500 samples are used for training purpose and the rest 500 are set aside for validation for the neural network. We train the network until the learning error doesn’t decrease any further for the validation set or 1000 epochs are reached. Once the loss/error is minimized, we test the neural network on our testing set which contains 1000 samples of each class. 
The second part is the task of image classification using deep learning with Convolutional Neural Networks. For this task, we use the MNIST dataset which contains 70,000 images of handwritten digits, divided into 60,000 training images and 10,000 testing images. 

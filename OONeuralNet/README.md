# *Generalized application for Neural Network creation and training*

## **my personal project**

### What will the application do?

In its final form, the application will provide the users with a GUI for creating and training various types of neural networks without needing language-specific knowledge (in contrary to Tensorflow, Keras, Pytorch). For now, my goal is for Multilayer Perceptron, Convolutional Neural Network, and Recurrent Neural Network be easily created and trained via supervised learning.

The application is coded in such a way that the neural network architecture is stripped down to a rather abstract level. Each data point (of arbitrary type, most notably double) is represented by a Synapse object. Neuron contains Matrix objects which holds synapses and also contains weight and bias matrices. Each neuron is equipped with a Reception, Combination, and Activation function. A neural network is then composed of layers which are composed of neurons. Backpropagation is computed via automatic differentiation to facilitate training using arbitrary linear / non-linear functions.

By the end of the development, the user should be able to simply upload a txt. file (or even image files) with the necessary data and the application will use that for training and / or validation. The user should also be able to automatically and manually customize connectivity of neurons, ranging from fully connected layers to sparsely connected layers.


### **Who will use it?**

* (hopefully) data scientist and machine learning engineers, especially for conveniently testing which neural network models are most appropriate for some given data sets.
* people casually interested in machine learning, due to the ease of use and lack of language-specific programming experience
* me (and hopefully other science students) for playing around with data and possibly using it for other projects

### *Why is this project of interest to you?*

I was working on a data-driven personal project involving fourier transform on human voice sound files, and was slightly annoyed with having to code my own functions for signal analysis, and needing to use various libraries (NumPy, SciPy) while also somehow knowing how they all interact with each other. I was about to learn Pytorch / Tensorflow to feed the data in but thought to myself, "wouldn't it be convenient if all the data analysis and machine learning tools are available in a nice-looking, easily accessible GUI?". I programmed a Multilayer Perceptron from scratch for password strength classification in Python during highschool as part of a Math project, and figured that this project should be doable. So anyway, here goes.


### **User Stories**

1. As a user, I want to be able to add layers to a neural network.
2. As a user, I want to be able to add vector data to a list of vector data.
3. As a user, I want to be able to train a neural network to classify vector data.
4. As a user, I want to be able to choose among at least 2 different activation functions for each layer of the neural network.
5. As a user, I want to be able to test and see the result of a feed-forward propagation when I input a vector data.
6. As a user, I want to be able to input data by typing data points separated by commas.
7. As a user, I want to be able to see how the loss function value changes with each training iteration for each data.
8. As a user, I want to be able to set the number of training epochs.
9. As a user, I want to be able to set the learning rate.
10. As a user, I want to be able to save the weights and biases of my neural network to file
11. As a user, I want to be able to be able to load the weights and biases of my neural network from file


### **Phase 4: Task 2**
Fri Apr 01 03:51:04 PDT 2022 Added layer of size 4 to network <br>
Fri Apr 01 03:51:04 PDT 2022 Added layer of size 5 to network <br>
Fri Apr 01 03:51:04 PDT 2022 Added layer of size 3 to network <br>
Fri Apr 01 03:51:04 PDT 2022 Added layer of size 4 to network <br>
Fri Apr 01 03:51:04 PDT 2022 Added layer of size 5 to network <br>
Fri Apr 01 03:51:04 PDT 2022 Added layer of size 7 to network <br>
Fri Apr 01 03:51:04 PDT 2022 Parameters reset with neurons reconnection <br>
Fri Apr 01 03:51:15 PDT 2022 Added layer of size 2 to network <br>
Fri Apr 01 03:51:15 PDT 2022 Added layer of size 10 to network <br>
Fri Apr 01 03:51:15 PDT 2022 Added layer of size 2 to network <br>
Fri Apr 01 03:51:15 PDT 2022 Parameters reset with neurons reconnection <br>
Fri Apr 01 03:51:19 PDT 2022 Parameters reset with neurons reconnection <br>
Fri Apr 01 03:51:19 PDT 2022 Parameters reset with neurons reconnection <br>
Fri Apr 01 03:51:22 PDT 2022 Added layer of size 4 to network <br>
Fri Apr 01 03:51:22 PDT 2022 Parameters reset with neurons reconnection <br>
Fri Apr 01 03:51:24 PDT 2022 Added layer of size 4 to network <br>
Fri Apr 01 03:51:24 PDT 2022 Parameters reset with neurons reconnection <br>
Fri Apr 01 03:51:25 PDT 2022 Parameters reset with neurons reconnection <br>
Fri Apr 01 03:51:31 PDT 2022 Added layer of size 2 to network <br>
Fri Apr 01 03:51:31 PDT 2022 Added layer of size 10 to network <br>
Fri Apr 01 03:51:31 PDT 2022 Added layer of size 2 to network <br>
Fri Apr 01 03:51:31 PDT 2022 Parameters reset with neurons reconnection <br>
Fri Apr 01 03:51:41 PDT 2022 Input to Network: 0, 1 <br>
Fri Apr 01 03:51:41 PDT 2022 Output of Network: <br>
[ 0.9999967796767902 ] <br>
[ 3.220323209829117E-6 ] <br>

Fri Apr 01 03:51:45 PDT 2022 Input to Network: 1, 0 <br>
Fri Apr 01 03:51:45 PDT 2022 Output of Network: <br>
[ 0.9999957198747038 ] <br>
[ 4.280125296073632E-6 ] <br>

Fri Apr 01 03:52:00 PDT 2022 Input to Network: 0, 0 <br>
Fri Apr 01 03:52:00 PDT 2022 Output of Network: <br>
[ 1.9760207453595206E-6 ] <br>
[ 0.9999980239792546 ] <br>

Fri Apr 01 03:52:03 PDT 2022 Input to Network: 1, 1 <br>
Fri Apr 01 03:52:03 PDT 2022 Output of Network: <br>
[ 3.572193888888411E-6 ] <br>
[ 0.9999964278061111 ] <br>
<br>

### **Phase 4: Task 3**
1. I feel like there are too many unnecessary points of control and dependency between classes. For example, I should redesign the model such that Network uses Process, but Process does not use
Neuron nor Layer, since there is already a top-down (one-sided) association relationship between Network, Neuron and Layer anyway. 
2. I feel the main method, which is used for running the GUI, should be refactored into different classes for several components (e.g. different frames) of the GUI. This is for better code organization and to avoid
unintended dependencies between the GUI components which should be isolated from one another. For example, opening frame and the main GUI frame should be separate classes, both extending JFrame.
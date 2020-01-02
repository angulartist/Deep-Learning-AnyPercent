# Deep Learning Any%.
Logging my progress during my Deep Learning Speed Run

### Day 1
- [x] Learnt what an image is / what pixels are.
- [x] Learnt RGB and Grayscale color spaces, channels
- [x] Played with imgages as 3D NumPy multidimensional arrays
- [x] Played with the image aspect ratio (OpenCV)
- [x] Did some basic image processing (Flipping, Rotation, Translation, Masking...)
- [x] Plotted distribution of pixel intensities (Histograms)

### Day 2
- [x] Discovered image classification basics and its challenges (factors of variation, semantic gap...)
- [x] Went through types of learning (mainly supervised learning)
- [x] Learnt image classification steps (gathering data, splitting dataset, features extraction, training network, evaluating network)
- [x] Learnt how to quantify an image via features extraction (image descriptors, feature descriptors, feature vectors)
- [x] Applied Color Channel Statistics and Color Histograms algorithms (extracting global color distribution from an image)
- [x] Learnt and applied Hu Moments, LBP and HOG to extract shapes and textures

> N1: It's very important to understand features extraction when building CNNs to avoid the "black/magic-box" effect.

### Day 3
- [x] Built a simple image classifier using the k-Nearest Neighbor (k-NN) algorithm
- [x] Fine-tunned my k-NN hyperparameters by testing different distance metrics (Euclidian, Manhattan) and adjusting the number of neighbors k via the validation dataset
- [x] Learnt linear classification basics and parameterized learning (scoring function, loss function (Multi-Class SVM, Cross-entropy, weights)

### Day 4
- [x] Discovered Gradient Descent (optimization algorithm, finding optimal values for our hyperparameters) and implemented a simple version
- [x] Learnt Stochastic Gradient Descent (SGD) and the concept of mini-batches, [reducing training time, lowering loss, increasing accuracy] + learnt Momentum (in comparison to Nesterov's acceleration)
- [x] Learnt about underfitting, overfitting, generalization and how to avoid them via the regularization method

### Day 5
Before deep diving into deep learning and neural networks :

- [x] Learnt about Logistic Regression, SVM, Random forests and Decision Trees, did simple implementations
- [x] Applied various local invariant descriptors such as SURF, RootSIFT 
- [x] Discovered binary descriptors extraction

### Day 6
- [x] Learnt neural network basics, activation functions (Step, Sigmoid, ReLU, Leaky...) their pros/cons
- [x] Discovered the perceptron algorithm and implemented one basic
- [x] Learnt Feedforward architecture and Backpropagation technique (perceptron)

### Day 7
- [x] Learnt different techniques of weights initialization (uniform, normal, LeCun, Glorot...)
- [x] Deep dived into Convolutional Neural Networks (CNNs): Understood that the concept of "convolution" is pretty straightforward as it's an element-wise multiplication between two matrices followed by a sum
- [x] Learnt the concept of Kernels (It was quite hard to understand why this reduces the dimension of the input volume -> https://www.youtube.com/watch?v=C_zFhWdM4ic / https://github.com/vdumoulin/conv_arithmetic)
- [x] Learnt building blocks composing CNNs such as CONV layer, POOL layer, Batch Norm Layer, Dropout...
- [x] Discovered some famous CNNs architectures and patterns

### Day 8
- [x] Continued working with Backpropagation and took a refresher about SGD, kernels...
- [x] Learnt how to detect/monitor underfitting/overfitting by exploring the behiavour of  train loss and val loss
- [x] Played with some regularization methods to reduce overfitting : Learning rate schedulers (step-based decay, standard),  Data Augmentation, Batch Normalization (more stable convergence) and Dropout (higher test acc, lower train acc)
- [x] Learnt how to checkpoint models whenever they make progress (ie: loss/acc improves) for later predictions
- [x] Implemented LeNet on CIFAR-10

### Day 9
- [x] Deep dived into more theorical knowledge about backprop, optimizers and activation functions (I recommend that course https://www.coursera.org/learn/machine-learning#syllabus which covers some basic stuff)
- [x] Tried advanced optimization methods (adaptive learning rate => adagrad, rmsprop, adam, nadam), played with conv+strides and pool layers to see how it affects the input volume dimensions
- [x] Implemented AlexNet on a custom dataset while monitoring under/over-fitting and driving the expriment in aim to reach a higher accuracy
- [x] Just read about ensemble methods which is a great machine learning technique to improve classification accuracy and kind of easy to implement, however this requires some heavy computation power and I'll need to rely on cloud services
- [x] Started to study transfert learning features extraction/fine-tuning

### Day 9
- [x] Got hands dirty by playing with HDF5 (binary data format for storing huge volume of data)
- [x] Extracted features on a custom dataset using VGG16 net with ImageNets weights
- [x] Used these features to train a Logistic Regression classifier, reaching 96% accurracy
- [x] Took a whiteboard and just tried to give the easiest explanation on how neural networks work (including feedforward and backpropagation) + then tried to explain how ConvNets work, was a great refresher
- [x] Played with matrices and vectors (dot product...)

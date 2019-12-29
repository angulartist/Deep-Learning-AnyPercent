# Learning Deep Learning for Computer Vision in 69 days.
Logging my progress during my 69 days Deep Learning for CV journey.

### Day 1
[X] Learnt what an image is / what pixels are.
[X] Learnt RGB and Grayscale color spaces, channels
[X] Played with imgages as 3D NumPy multidimensional arrays
[X] Played with the image aspect ratio (OpenCV)
[X] Did some basic image processing (Flipping, Rotation, Translation, Masking...)
[X] Plotted distribution of pixel intensities (Histograms)

### Day 2
[X] Discovered image classification basics and its challenges (factors of variation, semantic gap...)
[X] Went through types of learning (mainly supervised learning)
[X] Learnt image classification steps (gathering data, splitting dataset, features extraction, training network, evaluating network)
[X] Learnt how to quantify an image via features extraction (image desctriptors, feature descriptors, feature vectors)
[X] Applied Color Channel Statistics and Color Histograms algorithms (extracting global color distribution from an image)
[X] Learnt and applied Hu Moments, LBP and HOG to extract shapes and textures

> N1: We skip features extraction when building CNNs as they are end-to-end models relying on raw image pixel intensities. But I found it very important to avoid the "black/magic-box" effect.

### Day 3
[X] Built a simple image classifier using the k-Nearest Neighbor (k-NN) algorithm
[X] Fine-tunned my k-NN hyperparameters by testing different distance metrics (Euclidian, Manhattan) and adjusting the number of neighbors k via the validation dataset
[X] Learnt linear classification basics and parameterized learning (scoring function, loss function (Multi-Class SVM, Cross-entropy, weights)

### Day 4
[X] Discovered Gradient Descent (optimization algorithm, finding optimal values for our hyperparameters) and implemented a simple version
[X] Learnt Stochastic Gradient Descent (SGD) and the concept of mini-batches, [reducing training time, lowering loss, increasing accuracy] + learnt Momentum (in comparison to Nesterov's acceleration)
[X] Learnt about underfitting, overfitting, generalization and how to avoid them via the regularization method

### Day 5
Before deep diving into deep learning and neural networks :

[X] Learnt about Logistic Regression, SVM, Random forests and Decision Trees, did simple implementations
[X] Applied various local invariant descriptors such as SURF, RootSIFT 
[X] Discovered binary descriptors extraction

### Day 6
[X] Learnt neural network basics, activation functions (ReLU, Leaky...)
[X] Discovered the perceptron algorithm and implemented one basic
[X] Learnt Feedforward and Backpropagation

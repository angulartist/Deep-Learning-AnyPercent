# Learning Deep Learning for Computer Vision in 69 days.
Logging my progress during my 69 days Deep Learning for CV journey.

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
- [x] Learnt how to quantify an image via features extraction (image desctriptors, feature descriptors, feature vectors)
- [x] Applied Color Channel Statistics and Color Histograms algorithms (extracting global color distribution from an image)
- [x] Learnt and applied Hu Moments, LBP and HOG to extract shapes and textures

> N1: We skip features extraction when building CNNs as they are end-to-end models relying on raw image pixel intensities. But I found it very important to avoid the "black/magic-box" effect.

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

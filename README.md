# Fashion-MNIST-Neural-Network-Scratch
Overview

This project implements a fully-connected neural network from scratch (using only NumPy) to classify images from the Fashion MNIST dataset.
The network is trained using the Adam optimizer and uses ReLU activation in hidden layers and softmax in the output layer.

We also include a prediction pipeline for the test dataset to evaluate accuracy and visualize predictions.

1. Dataset

Training set: /kaggle/input/fashionmnist/fashion-mnist_train.csv

Shape: 60000 samples, 28x28 pixels flattened into 784 features + 1 label column.

Test set: /kaggle/input/fashionmnist/fashion-mnist_test.csv

Shape: 10000 samples, same format as training data.

Preprocessing:

Normalize pixel values to [0,1].

Convert labels to one-hot encoding for training.

2. Neural Network Architecture

Input layer: 784 neurons (28x28 pixels)

Hidden layer 1: 512 neurons, ReLU activation

Hidden layer 2: 256 neurons, ReLU activation

Output layer: 10 neurons, softmax activation (for 10 classes)

3. Key Functions
3.1 Forward Propagation

Computes linear combinations and activations for each layer.

Formulas:

Hidden layer 1: 
    𝑍1 = 𝑊1 * 𝑋 + 𝑏1,   𝐴1 = 𝑅𝑒𝐿𝑈(𝑍1)

Hidden layer 2: 
    𝑍2 = 𝑊2 * A1 + 𝑏2,   𝐴2 = 𝑅𝑒𝐿𝑈(𝑍2)
Output layer: 
    𝑍3 = 𝑊3 * A2 + 𝑏3,   𝐴1 = softmax(𝑍3)

3.2 Cost Function

Cross-entropy loss:
     J=−m1​i=1∑m​j=1∑10​Yj,i​log(A3j,i​)

3.3 Backward Propagation

Computes gradients of weights and biases with respect to the cost.

Uses chain rule and derivatives of ReLU and softmax.

3.4 Parameter Updates

Uses Adam optimizer:

Tracks first moment (v) and second moment (s) for each parameter.

Applies bias correction to stabilize learning.

Updates weights with adaptive learning rate.

4. Training Procedure

Initialize parameters using He initialization for weights and zeros for biases.

Shuffle dataset each epoch.

Mini-batch gradient descent:

Batch size = 128

Forward pass → compute cost → backward pass → Adam update

Track training accuracy at the end of each epoch.

Typical training results after 50 epochs:

Epoch 50/50 - Cost: 0.0497 - Accuracy: 98.17%

5. Prediction on Test Set

Load test dataset, normalize, and one-hot encode labels.

Perform forward pass using trained weights.

Convert output probabilities to predicted classes using argmax.

Compute accuracy:

test_acc = np.mean(predictions_test == Y_test) * 100


Visualize sample predictions:

Displays first 5 test images with predicted and true labels.

Example Test Accuracy:

Test Accuracy: 88-90% (varies depending on training)

6. How to Run

Load the training and test CSV files from Kaggle.

Execute cells sequentially in Jupyter Notebook:

Data preprocessing

Parameter initialization

Forward/backward propagation

Adam updates

Training loop

Test predictions and visualization

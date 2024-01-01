Training a deep neural network

The training process for a deep neural network iterations, called epochs. For the first epoch, you start by assigning random initialization values for the weight (w) and bias (b) values. Then the process is as follows:

1. Features for data observations with known lagbel values are submitted to the input layer. Generally, these observations are grouped into "batches" (often referred to as `mini-batches`).

2. The neurons then apply their function, and if activated, pass the result onto the next layer until the output layer produces a prediction.

3. The prediction is compared to the actual known value, and the amount of variance between the predicted and true values (which we call the loss) is calculated. """A loss function is used to calculate the difference""".

4. Based on the results, revised values for the weights and bias values are calculated to reduce the loss, and these adjustments are `backpropagated` to the neurons in the network layers.

5. The next epoch repeats the batch training forward pass with the revised weight and bias values, hopefully improving the accuracy of the model (by reducing the loss).

### Steps

1. Explore the data using Pandas or another tool of your choice.
   1. Load your training dataset into Pandas (excluding rows with null values)
   2. Check your data for inaccuracies
      - Clean your data
   3. Deep Learning models work best when they are on the same scales. Implement some custom normalization to resolve this. For example, if you have kilograms in 1 feature, but grams in another feature, it is best to normalize them to grams or kilograms.
   4. Ensure dataset you are using meets minimum recommended requirements.
   - (Optional) Sample the dataset
   5. Split the dataset into training and test data using a 70/30 split.
   6. Review the first 25 training features and corresponding labels
2. Prepare the environment
   1. Install and import the Pytorch or Tensorflow libraries
3. Prepare the dataset
   1. Create a dataset and loader for the training data and labels
   2. Create a dataset and loader for the test data and labels
4. Define a Neural Network
   1. Define number of hidden layer models
   2. Define the neural network (as a class)
      1. Setup Initialization method
         1. Setup your Input layer
         2. Setup your Hidden layer
         3. Setup your Output layer
      2. Define your Activation functions in a Forward method
      3. Create a model instance from the network
5. Train the model
   1. Set the model to training mode
      1. For each batch of tensors, perform the following:
         1. Run it through the feedforward procedure
            1. Store the data and target (x and y) from each tensor
            2. Reset the gradients by zeroing them out at the start of each loop
            3. Store the output value from the activation function
            4. Aggregate the loss
         2. Run it through the backpropagate procedure
            1. Compute the gradients of the loss function with respect to the model parameters.
            2. Upate the model parameters based on the gradients computed by the loss.backward() function in previous step.
   1. Evaluation (test) mode - to prevent backpropagation
      1. For each batch of tensors, perform the following:
         1. Store the data and target (x and y) from each tensor
         2. Get the predictions
         3. Calculate the loss
         4. Calculate the accuracy
      2. Calculate the average loss and total accuracy for this epoch
      3. Return the average loss for the epoch

- Specify the loss criteria (using CrossEntropyLoss for multi-class classification in this example)
- Use an "Adam" optimizer to adjust weights
  (see https://pytorch.org/docs/stable/optim.html#algorithm for details of supported algorithms)
- We'll track metrics for each epoch in these arrays
- Train over 50 epochs
  - For each epoch:
    - print the epoch number
    - Feed training data into the model to optimize the weights
    - Feed the test data into the model to check its performance
    - Log the metrics for this epoch

---

### Bing help

How do I know when the dataset is too small or what factors should I consider?

Bing:
The size of the dataset required for deep learning depends on various factors, such as the complexity of the problem, the number of features, the quality of the data, and the architecture of the model. There is no definitive answer to this question, but some general guidelines are:

- The dataset should have at least an order of magnitude more examples than trainable parameters in the model. This means that if your model has 10,000 parameters, you should have at least 100,000 examples.
- The dataset should be reliable, meaning that it has few errors, noise, or outliers, and that it is properly filtered and labeled for the problem at hand.
- The dataset should be representative, meaning that it captures the diversity and distribution of the real-world data that the model will encounter.
- The dataset should be balanced, meaning that it has a similar number of examples for each class or outcome that the model is trying to predict.

If your dataset is too small, you may encounter problems such as overfitting, underfitting, or high variance in the model performance. To overcome these issues, you may try some techniques such as:

- Data augmentation, which is the process of creating new examples from existing ones by applying transformations such as rotation, scaling, cropping, flipping, or adding noise.
- Data synthesis, which is the process of generating new examples from scratch using methods such as generative adversarial networks (GANs), variational autoencoders (VAEs), or simulation.
- Data oversampling, which is the process of duplicating or creating new examples from the minority class to balance the dataset.
- Data undersampling, which is the process of removing or selecting a subset of examples from the majority class to balance the dataset.

However, these techniques may not always work well or may introduce some biases or artifacts in the data. Therefore, the best way to ensure a good dataset size for deep learning is to collect more real-world data whenever possible. You can also use some tools to search for existing datasets that are relevant to your problem, such as [Google Dataset Search](^1^) or [Kaggle Datasets](^2^).

Source: Conversation with Bing, 12/31/2023
(1) Impact of Dataset Size on Deep Learning Model Skill And Performance .... https://machinelearningmastery.com/impact-of-dataset-size-on-deep-learning-model-skill-and-performance-estimates/.
(2) The Size and Quality of a Data Set | Machine Learning | Google for .... https://developers.google.com/machine-learning/data-prep/construct/collect/data-size-quality.
(3) How does Dataset Size impact Deep Learning Models? - Medium. https://medium.com/@focaloidtechnologies/how-does-dataset-size-impact-deep-learning-models-d4c5b4ab8d90.
(4) How Much Data Is Needed For Machine Learning? | Graphite Note. https://graphite-note.com/how-much-data-is-needed-for-machine-learning.
(5) machine learning - How to get the data set size required for neural .... https://stats.stackexchange.com/questions/233512/how-to-get-the-data-set-size-required-for-neural-network-training.

Me:
The page describes the dataset as being too small to be useful for deep learning, so they performed an oversample to increase it's size. What should I be considering 'too small' or 'large enough' when viewing the dataset?

Bing:
The size of the dataset required for deep learning depends on various factors, such as the complexity of the problem, the number of features, the quality of the data, and the architecture of the model. There is no definitive answer to this question, but some general guidelines are:

- The dataset should have at least an order of magnitude more examples than trainable parameters in the model. This means that if your model has 10,000 parameters, you should have at least 100,000 examples.
- The dataset should be reliable, meaning that it has few errors, noise, or outliers, and that it is properly filtered and labeled for the problem at hand.
- The dataset should be representative, meaning that it captures the diversity and distribution of the real-world data that the model will encounter.
- The dataset should be balanced, meaning that it has a similar number of examples for each class or outcome that the model is trying to predict.

If your dataset is too small, you may encounter problems such as overfitting, underfitting, or high variance in the model performance. To overcome these issues, you may try some techniques such as:

- Data augmentation, which is the process of creating new examples from existing ones by applying transformations such as rotation, scaling, cropping, flipping, or adding noise.
- Data synthesis, which is the process of generating new examples from scratch using methods such as generative adversarial networks (GANs), variational autoencoders (VAEs), or simulation.
- Data oversampling, which is the process of duplicating or creating new examples from the minority class to balance the dataset.
- Data undersampling, which is the process of removing or selecting a subset of examples from the majority class to balance the dataset.

However, these techniques may not always work well or may introduce some biases or artifacts in the data. Therefore, the best way to ensure a good dataset size for deep learning is to collect more real-world data whenever possible. You can also use some tools to search for existing datasets that are relevant to your problem, such as [Google Dataset Search](^1^) or [Kaggle Datasets](^2^).

Source: Conversation with Bing, 1/1/2024
(1) Impact of Dataset Size on Deep Learning Model Skill And Performance .... https://machinelearningmastery.com/impact-of-dataset-size-on-deep-learning-model-skill-and-performance-estimates/.
(2) The Size and Quality of a Data Set | Machine Learning | Google for .... https://developers.google.com/machine-learning/data-prep/construct/collect/data-size-quality.
(3) How does Dataset Size impact Deep Learning Models? - Medium. https://medium.com/@focaloidtechnologies/how-does-dataset-size-impact-deep-learning-models-d4c5b4ab8d90.
(4) How Much Data Is Needed For Machine Learning? | Graphite Note. https://graphite-note.com/how-much-data-is-needed-for-machine-learning.
(5) machine learning - How to get the data set size required for neural .... https://stats.stackexchange.com/questions/233512/how-to-get-the-data-set-size-required-for-neural-network-training.

Me:
In this page, during the 'train' function, there is a call to optimizer.zero_grad(). What is this and what does it do?

Bing:
The `optimizer.zero_grad()` function is used to reset the gradients of the model parameters before each training iteration. This is necessary because PyTorch accumulates the gradients for each parameter as they are computed, so without zeroing them out, the gradients would reflect the sum of the gradients over all training iterations rather than the gradient of the current iteration. By calling `optimizer.zero_grad()`, we ensure that the gradients are only calculated for the current batch of data.

Me:
I want to take an educated guess on what the loss.backward() function performs during the backpropagate. First let me explain it, then gauge my understanding and then elaborate on what the loss.backward() does in more detail.

My understanding is that I runs a calculation of some type 0.5(y - y^)^2 (if we are using mean squared) through the hidden layers in reverse order and aggregates the value of how 'off' the prediction is. I am not sure about the optimizer.step() though.

Bing:
Your understanding of the loss.backward() function is partially correct. The loss.backward() function computes the gradients of the loss function with respect to the model parameters, which is used to update the parameters during training. The loss function measures how well the model is performing on the training data, and the gradients indicate the direction and magnitude of the change needed to improve the performance.

The loss function can be any function that maps the predicted values of the model to the actual values of the training data. For example, if we are using mean squared error (MSE) as the loss function, then the formula for the loss is:

$$L(y, \hat{y}) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

where $y$ is the true value of the target variable, $\hat{y}$ is the predicted value of the target variable, and $n$ is the number of examples in the training data. The loss.backward() function computes the gradients of this loss function with respect to the model parameters, such as the weights and biases of the neural network.

The optimizer.step() function, on the other hand, updates the model parameters based on the gradients computed by the loss.backward() function. The optimizer is an algorithm that determines how the model parameters should be updated to minimize the loss function. There are many different types of optimizers, such as stochastic gradient descent (SGD), Adam, and RMSprop, each with their own update rules and hyperparameters.

In summary, the loss.backward() function computes the gradients of the loss function with respect to the model parameters, and the optimizer.step() function updates the model parameters based on these gradients. By iteratively computing the gradients and updating the parameters, the model learns to make better predictions on the training data.

Source: Conversation with Bing, 1/1/2024
(1) machine learning - pytorch - connection between loss.backward() and .... https://stackoverflow.com/questions/53975717/pytorch-connection-between-loss-backward-and-optimizer-step.
(2) Backpropagation in a Neural Network: Explained | Built In. https://builtin.com/machine-learning/backpropagation-neural-network.
(3) Backpropagation - Wikipedia. https://en.wikipedia.org/wiki/Backpropagation.
(4) Optimizing Model Parameters — PyTorch Tutorials 2.2.0+cu121 documentation. https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html.
(5) Optimizers in Deep Learning: A Comprehensive Guide - Analytics Vidhya. https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-on-deep-learning-optimizers/.
(6) Optimizer in deep learning - Medium. https://medium.com/@ipradeepkumarlodhi/optimizer-in-deep-learning-7c2cc72c8f7e.
(7) Optimizers in Deep Learning. What is an optimizer? - Medium. https://medium.com/mlearning-ai/optimizers-in-deep-learning-7bf81fed78a0.
(8) OPTIMIZERS IN DEEP LEARNING. What is optimizer ? | by arshad ... - Medium. https://medium.com/analytics-vidhya/optimizers-in-deep-learning-36549ea9c59f.

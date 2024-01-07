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

### Summary of hyperparameter tuning

Most machine learning problems require a lot of hyperparameter tuning. Unfortunately, we can't provide concrete tuning rules for every model. Lowering the learning rate can help one model converge efficiently but make another model converge much too slowly. You must experiment to find the best set of hyperparameters for your dataset. That said, here are a few rules of thumb:

- Training loss should steadily decrease, steeply at first, and then more slowly until the slope of the curve reaches or approaches zero.
- If the training loss does not converge, train for more epochs.
- If the training loss decreases too slowly, increase the learning rate. Note that setting the learning rate too high may also prevent training loss from converging.
- If the training loss varies wildly (that is, the training loss jumps around), decrease the learning rate.
- Lowering the learning rate while increasing the number of epochs or the batch size is often a good combination.
- Setting the batch size to a very small batch number can also cause instability. First, try large batch size values. Then, decrease the batch size until you see degradation.
- For real-world datasets consisting of a very large number of examples, the entire dataset might not fit into memory. In such cases, you'll need to reduce the batch size to enable a batch to fit into memory.

#### Remember: the ideal combination of hyperparameters is data dependent, so you must always experiment and verify.

#### Overfitting tip - Training / Testing Split Results

We'd expect a lower precision on the test set

### Addressing Potential issues during the Data Split for Validation Sets and Test Sets

In the [Google Collab]('https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/validation_and_test_sets.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=validation_tf2-colab&hl=en#scrollTo=hMqWDc_m6rUC'), it is highlighted that one of the common issues in Machine Learning is that when you split data, you can end up getting different results for the `loss curves`. "No matter how you split the training set and the validation set, the loss curves differe significantly. Evidently, the data in the training set isn't similar enough to the data in the validation set".

I want to highlight something else and that is the fact that you must include your data as being part of the issue you are not getting the results expected after changing various hyperparameters. After all, your model will only be as good as your data. A good model can not fix your data as it's based on the data itself.

In the training data example, the data is the issue because the data is sorted by one of it's features. By splitting the data using 'order' (example, the 'first 60% is training' and the 'last 40% is testing') you will inadvertently be introducing what I would call a pre-estimation (terrible term, thinking of a better one later).

The `collab` states to fix this is to reindex the data by shuffling it. This is one way to do it with `Pands`. This would be good to check if this is handled by `Pytorch` at all.

```
# Use the reindex method of your dataframe and create a new variable for the results
shuffled_train_df = train_df.reindex(np.random.permutation(train_df.index))
```

At that point, you can retrain your model. I am quoting the `Collab's` "Step 2" as a Note in the code.

```
2. Pass `shuffled_train_df` (instead of train_df) as the second argument to `train_model` (in the code call associated with Task 1) so that the call becomes as follows:
epochs, rmse, history = train_model(my_model, shuffled_train_df, my_feature, my_label, epochs, batch_size, validation_split)
```

### Feature Engineering

75% of your time will be used in this area alone.

Tips:
String values like 'street names' can be handled using `one-hot` encoding. By using a one-hot encoding; the one hot encoding has a unique coefficient for each possible string we might see. For example, if we have a one hot encoding for street names, we'll have a unique coefficient for each possible street. If we have a one-hot encoding for street names, we will have a unique coefficient for each possible street. 'And then when we see the street we have, say Main Street, we'll put a one in the coefficient for Main Street, and a zero everywhere else. We will use that as our feature vector for representing strings. One-hot encodings are extremely handy for this sort of sparse, categorical data.' << What does this mean? I thought we were mapping each street name with a different integer.>>

#### What makes a Feature a 'Good Feature'.

1. **A feature should occur with a non-zero value at least a handful of times or more in our data set**. If a feature occurs with a non-zero value only extremely rarely, or maybe only once, it's probably not a great feature to use and should maybe be filtered out in a pre-processing step.

2. **Our feature should have a clear and obvious meaning**. For example, it's much easier to have a house age in units of years than a house age in seconds since Unix epoch. It will be much easier to debug and reason about.

3. **Features shouldn't take on magic values**. So for example it would be a bad idea if we had a feature whose job it was to tell us how many days a house had been on the market. Such as trying to use a special value of 'negative 1' to show that the house had never been on the market. Instead a better idea would be to have an indicator feature that would take on a boolean value, showing whether our days on the market feature was defined or not. So we can have a boolean (feature), 1-0 to show is that other feature defined. This allows the `original feature`, 'days_on_the_market', to maintain its natural units of 0 through n.

4. **Our feature values shouldn't change over time**.

5. **A feature shouldn't have crazy outlier values**. For example, in our California housing data, if we create a synthetic feature (`synthetic feature` is a feature you create yourself by computing or combining features and is not included in the original dataset), like 'rooms_per_person', where we take the total rooms and divide it by the total population, we get some reasonable values between 0 and 3 or 4 rooms per person for most city blocks. But in some city blocks, we've got values as high as 50, which does not make very much sense in relation to the other data. What we can do there is maybe cap or transform our feature so that we get rid of those crazy outlier values.

Tip: Binning
If we think about the effect of latitude on the housing prices in California, there is no sort of linear relationship from North to South that maps directly to housing prices. But within any particular latitude there often is a pretty strong correlation. (sounds like grouping together similar values to increase visibility).

### Good Habits

Know your data. Don't ever treat your data like a 'black box'.

- Visualize your data using historgrams or scatter plots, various rank metrics.
- You can do various data debugging, that ends up consuming a lot of time from the development looking for "duplicate values", "missing values", anything that looks like an "outlier".
- Using something that could work like a 'dashboard' that you can interact with your data can be incredibly helpful.
- Monitor your data over time. Just because your data sources were good yesterday does not mean they will be good tomorrow. Anything you can do to add monitoring to the stability of our features over time, will increase the robustness of our systems.

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

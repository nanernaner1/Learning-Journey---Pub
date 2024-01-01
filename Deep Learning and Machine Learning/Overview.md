Training a deep neural network

The training process for a deep neural network iterations, called epochs. For the first epoch, you start by assigning random initialization values for the weight (w) and bias (b) values. Then the process is as follows:

1. Features for data observations with known lagbel values are submitted to the input layer. Generally, these observations are grouped into "batches" (often referred to as `mini-batches`).

2. The neurons then apply their function, and if activated, pass the result onto the next layer until the output layer produces a prediction.

3. The prediction is compared to the actual known value, and the amount of variance between the predicted and true values (which we call the loss) is calculated. """A loss function is used to calculate the difference""".

4. Based on the results, revised values for the weights and bias values are calculated to reduce the loss, and these adjustments are `backpropagated` to the neurons in the network layers.

5. The next epoch repeats the batch training forward pass with the revised weight and bias values, hopefully improving the accuracy of the model (by reducing the loss).

### Steps

1. Explore the data.
   1. Load your training dataset (excluding rows with null values)
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

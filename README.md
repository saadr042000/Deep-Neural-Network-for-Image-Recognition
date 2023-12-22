<a name="br1"></a> 

COMP 551 Project 2

Saad Rahman

**Abstract**

In this project, we aimed to classify image data using Multilayer Perceptrons (MLP) and

Convolutional Neural Networks (CNN). Through various experiments, we analyzed the impact of

weight initialization, network depth, activations, regularization techniques, and different model

architectures on image classification accuracy. This paper summarizes our main findings on the

Fashion MNIST and CIFAR-10 datasets. I implemented the MLP model **from scratch,** ie no

pre-implemented model was used from frameworks like Tensorflow, Sci-kit learn etc.

**Introduction**

Image classification remains a fundamental problem in the domain of machine learning. Utilizing

the provided datasets, Fashion MNIST and CIFAR-10, this project explores the capabilities of

Multilayer Perceptrons and Convolutional Neural Networks for image classification. Through

multiple experiments, we aimed to understand how different hyperparameters and

configurations affect model performance, thus providing insights into best practices for neural

network model tuning.

**Datasets**

Fashion MNIST is a dataset of Zalando's article images, with 10 classes, 60,000 training

images, and 10,000 test images, each of size 28x28. CIFAR-10 consists of 60,000 32x32 color

images across 10 classes, with 6,000 images per class. There are 50,000 training images and

10,000 test images. We performed an exploratory analysis on both datasets, noting the

distribution of each class and visualizing representative images from each class. Both datasets

were vectorized and normalized to work with the model.

**Results**

**1. Weight Initialization:**

Weight Initialization

Zero Initialization

Training Accuracy in %

10\.0%

Testing Accuracy in %

10\.0%

Uniform [-1, 1]

Gaussian (0,1)

Xavier

85\.58%

84\.83%

98\.22%

81\.87%

81\.12%

88\.79%



<a name="br2"></a> 

Kaiming

98\.01%

88\.40%

On Fashion MNIST, Xavier and Kaiming initializations outperformed other methods in terms of

convergence speed and final accuracy, as shown by the previous table. However, zero

initialization resulted in poor accuracy scores.

The training progress for three different models are shown here, which is consistent with the

results shown above. Hence, for all the following experiments, we have used Kaiming weight

initialization for the model. Furthermore, trying different learning rates, we can see that a



<a name="br3"></a> 

learning rate of 0.01 is optimal for our model, which results in minimum loss as seen above from

the plot on the right. Moving forward, this is our choice of the hyperparameter for learning rate.

**2. Model Architecture and Complexity:**

Number of hidden layers

Training Accuracy

87\.70%

Test Accuracy

84\.23%

0

1

2

98\.07%

88\.80%

98\.43%

88\.02%

Increasing depth by adding more hidden layers improved accuracy. The 2-layer MLP had a

better performance than its shallower counterparts on Fashion MNIST. From the table, we can

see that the model with 2 hidden layers outperforms the models with lower number of hidden

layers. These results are consistent with our expectation of an MLP with 2 hidden layers being

better than the ones with a lower number of hidden layers.

**3. Activation Functions:**

From the following table, we can see that both the hyperbolic tangent and leaky ReLU activation

functions led to very good models, resulting in very high training and testing accuracies.

Compared to the model with ReLU activations above, the following two models had very similar

training accuracies of approximately 98%. However, the model with ReLU activations had a

slightly higher testing accuracy of around 88% compared to ~87% for the two models. Looking

at the performances, I would conclude that the ReLU activation is better, even though the

difference is very small.

Activation Function

Training Accuracy

Testing Accuracy

tanh

98\.28

98\.30

87\.20

87\.81

Leaky ReLU

**4. Regularization:**

Regularization Type

Training Accuracy

89\.73%

Testing Accuracy

86\.90%

L2

L1

66\.32%

65\.33%



<a name="br4"></a> 

Firstly, we can see both forms of regularization significantly reduces the gap between training

and testing accuracies, thus removing any possibility of overfitting. However, the results indicate

that L2 regularization is better for our model compared to L1 since it has a much higher training

and testing accuracy for the dataset.

**5. Data Normalization:**

Training with unnormalized images led to lower accuracy as seen from the following table. This

emphasizes the importance of normalization. The absence of input normalization can result in

gradient values that are either too small (vanishing gradients) or too large (exploding gradients).

This can make training unstable and hinder convergence.

Training Accuracy

10\.0%

Testing Accuracy

10\.0%

**6. Convolutional Neural Networks:**

CNN accuracy on Fashion MNIST: 75%. This is slightly lower compared to MLP with Xavier

weight initialization, which had testing accuracy of 88.79%. This can be attributed to the

restrictions set on the CNN. We are confident that, given a free-hand in hyperparameters, the

CNN would end up doing better in classifying the Fashion MNIST.

**7. MLP vs CNN on CIFAR-10:**

Model

MLP

Training Accuracy

Testing Accuracy

32\.86%

47\.0%

\-

CNN

54%

As we can see, there was an increase of almost 21% in testing accuracy for the CNN model,

which isn't trivial, meaning that for this dataset, it is easy to see how CNN fairs better. There can

be multiple reasons why this is the case for CIFAR-10, one reason is that CIFAR-10 has 3

channels (as opposed to 1), and CNN tends to work better with these kinds of data. Another

reason is that the photos are very blurry and it is hard to detect patterns, but CNN is more

specialized and capable to do that than a normal MLP, making it have more accuracy in such a

task.

**8. Optimizer Investigation:**



<a name="br5"></a> 

SGD with 0,0.1,0.5,0.7,0.9 had these accuracies respectively: 38%, 35%, 47%, 48%, and 54%.

They also had the following times to converge respectively (in seconds): 131.9, 135.7, 137.6,

137\.7, 133.8.

As for the Adam optimizer, it had 54% accuracy and converged in 182.8 seconds. We see that it

had the same accuracy as SGD with 0.9 (the highest of all tested) momentum, but took longer

to converge, leading us to conclude that for this specific situation using SGD with high

momentum would be most optimal. We see a steady increase in accuracy for all momentum, but

as for time to converge, We see that it starts low (131.9), steadily increases, until the last

momentum where it converges faster than all before it, except the first one (133.8). These

results are best shown by the graphs below:

**Conclusion**

This project provided a comprehensive exploration into image classification using neural

networks and reinforced the importance of various factors in neural network training. Key

takeaways include the importance of weight initialization, the benefits of deeper networks,

choice of activation function and the effectiveness of CNNs over MLPs for certain datasets.

Regularization and data normalization further augment training dynamics. Our findings provide

guidance for best practices in neural network model tuning for image classification.

Future explorations can delve deeper into other influential factors such as learning rate

schedules, data augmentation techniques, and ensemble methods to further optimize model

performance.

**References**

<https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>

<https://www.cs.mcgill.ca/~isabeau/COMP551/F23/slides/9-backprop.pdf>


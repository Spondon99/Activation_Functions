# Theoretical Understanding
## What is an Activation Function?
An activation function is a mathematical function which decides whether a particular neuron will be activated or not. It is specifically required to map the weighted sum of inputs from a linear space to a non-linear space. This allows the neural network to learn from complex non-linear datasets, which is why activation functions are a core component of a neural network.
There are many types of activation functions. The most commonly used ones are,

i)	Sigmoid or Logistic function

ii)	Hyperbolic Tangent or tanh function

iii)	Rectified Linear Unit or ReLU

iv)	Leaky ReLU

v)	Softmax

The selection of activation functions depends on the use case or the dataset that the neural network is trying to learn from. 
In this document, the Hyperbolic Tangent or tanh function is focused on and explained in detail.
## Hyperbolic Tangent Function (tanh)
The hyperbolic tangent function is one of the most common activation function used in the hidden layers of a neural network. The function takes input and produces an output that ranges from -1 to 1, in other words the range of the output is (-1, 1). Hyperbolic tangent function can be mathematically expressed as follows,

tanh(x) = (ex – e-x) / (ex + e-x)

Here,

X = the input value

e = mathematical constant which has the value of approximately 2.718

A graph of the tanh function has been shown below, on a series of random input values,

![image](https://github.com/Spondon99/Activation_Functions/assets/63509836/32786a0e-c822-48fe-9058-31de317ee272)

Fig.1. Hyperbolic Tangent function graphical representation

From the graph, it can be seen that regardless of how large or small the input value is, the tanh function will always produce an output in the range of (-1, 1). This is important because it introduces non-linearity to the learning process of a neural network, and keeps the value at a scaled level.

# Mathematical Exploration
## Derivation of the Hyperbolic Tangent function formula
For the construction of a neural network, an activation function needs to be chosen for the hidden layers that are differentiable. This is due to the backpropagation error signal calculation, which is used to determine the optimized weights and biases for the neural network layers. Tanh is one of the most commonly used function in this case. The derivative or the gradient of the tanh function can be calculated as follows,

f’(x) = \frac{d}{dx}tanh(x) = \frac{d}{dx}\frac{sinh(x)}{cosh(x)} = 1 – tanh2(x)

The derivative of the tanh function is function of feed-forward activation, which is evaluated at x value. Here, the tanh function is first expressed in terms of sinh and cosh functions, to simplify the derivation.  
The significance of the derivative of the tanh function comes when dealing with largely negative values. Unlike the sigmoid function, tanh function can deal more effectively with negative values. Some major points that make the derivative of the activation function significant are,
	
 Non-linearity: A differentiable activation function enables the neural networks to learn complex non-linear patterns in the data. During backpropagation, the derivative of the tanh function allows the calculation of gradients, which allow to update and optimize the parameters of the neural network.
	
 Bounded output: The average output of the tanh function is closer to zero, which helps in the convergence of the training of the network. It also prevents the saturation problem and reduces the vanishing gradient problem.
	
 Smoothness: The tanh function is smooth, so it is differentiable at every given value. This allows for efficient gradient-based optimization techniques, mainly backpropagation. So, the gradients are calculated at every value of the input and minimizes the loss according to that.

# Programming Exercise
The hyperbolic tangent function is implemented in Python and shown in code snippets below,

![image](https://github.com/Spondon99/Activation_Functions/assets/63509836/6aac5df7-a756-4f03-ae18-bbf3485843e0)

![image](https://github.com/Spondon99/Activation_Functions/assets/63509836/a10b8d04-f932-4408-8f57-fd44ae37faf8)

![image](https://github.com/Spondon99/Activation_Functions/assets/63509836/653014a1-ae99-45dc-8490-e7e301a6b306)

![image](https://github.com/Spondon99/Activation_Functions/assets/63509836/1cda0738-e9ee-479b-bdfc-ab8a9e49dd0e)

The code was done in a Jupyter Notebook, which is an interactive Python platform suitable for deep learning. In the first cell of the notebook, all the required libraries are imported. The math library contains mathematical functions and constants. The numpy or numerical Python library helps to create and manipulate arrays in Python. And the matplotlib library deals with the visualization of graphs and plots in Python.
In the next cell, a Python function is defined, which will take an array of numbers as an argument and return a list called ‘data’ by processing the array using the tanh activation function.
Another function is defined which will be used to plot the activation function graph. Here, a default parameter is added so that even if there is no argument passed while calling the plot_tanh() function, there is a default array to work with.
Finally, two graphs are plotted. One with an argument passed to the function, and one with the default parameter used. The resulting graphs are as following,

![image](https://github.com/Spondon99/Activation_Functions/assets/63509836/31bb50f5-c939-4f9c-a2b3-b1c7d378f69c)

![image](https://github.com/Spondon99/Activation_Functions/assets/63509836/1ba69d30-c500-4c43-ba35-7caee69d751e)

Fig.2. Graphical demonstration of tanh function.

# Analysis
## Advantages and Disadvantages
The advantages of using the hyperbolic tangent function as the activation function are,

a)	Zero-centered output: The output of the tanh function is centered around zero, which can help mitigate issues such as the vanishing gradient problem or the saturation problem. This helps particularly in networks with many layers. This is also useful when it comes to faster convergence during the training period.

b)	Similarity to Sigmoid: Tanh is closely related to the sigmoid function but is symmetric around the origin. This can sometimes be advantageous in certain scenarios, specifically when dealt with zero-centered data.

c)	Differentiability: Since the tanh function is smooth, so it is differentiable at each value. This ensures that the gradients can be computed at every step of the training process, allowing for weight updates.

Some disadvantages of the tanh function are,

a)	Vanishing gradient: While tanh can help to reduce the vanishing gradient problem in comparison to sigmoid, it might still suffer from the issue. This can happen frequently in very deep networks and when the weights are poorly initialized.

b)	Bias towards negative values: Since the range of the tanh function is (-1, 1). Its outputs are biased towards negative values. This can lead to bias which may not be desirable for the task.

c)	Computationally expensive: Computing the tanh function involves exponentiation, which can be computationally expensive compared to simpler activation functions like ReLU.


Impact on Gradient Descent and Vanishing Gradient Problem
The choice of activation function in a neural network can have a great impact on the behaviour of gradient descent and the vanishing gradient problem. The main ways in which the activation function impacts this are,

a)	Gradient Calculation: The activation function directly affects the calculation of gradients during backpropagation. This is used to compute the gradients of the loss function with respect to the network parameters. The form of the derivative can vary greatly between each activation function used.

b)	Vanishing Gradients: Certain activation functions are prone to the vanishing gradient problem. In this, the gradients become very small as they propagate backward through many layers of the network. This can slow down or even prevent learning in deep networks. So, this problem needs to be identified when choosing the activation function depending on the use case.

c)	Smoothness and Continuity: The smoothness and continuity of the activation function can impact the behaviour of gradient descent. Smooth activation functions such as tanh produce smooth loss surfaces. This can help gradient descent converge more smoothly and efficiently. Discontinuous activation functions like ReLU can introduce non-convexities or sharp edges in the loss surface, which can make the optimization more difficult.


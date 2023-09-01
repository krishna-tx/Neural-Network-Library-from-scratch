# Custom Neural Network Library from Scratch (PyTorch Clone)

I created this project because I wanted to understand how neural networks work. I developed the mathematics behind feed-forward neural networks with multivariable calculus and some basic linear algebra and applied it through code using Python and NumPy (for computational uses). The jupyter notebook goes through simple linear regression, a 2 layer neural network for binary classification, and an n-layer neural network for multi-class classification. I have also copied the custom library classes from the notebook into the "Classes" folder so that it can be used similar to the nn library in PyTorch.

# Forward and Backward Propagation
![nn_forward_backward_prop](https://github.com/krishna-tx/Neural-Network-Library-from-scratch/assets/16388863/25405e5c-6134-43b4-aa48-f9e0d2caad07)

On the left side of the above image, there is a diagram of an example neural network. This is the model that the math was written out for, but the same ideas can be generalized to simpler or more advanced feed-forward neural networks. 

## Forward Propagation
Under the diagram is the implementation of the forward propagation. It shows the equations with the appropriate variables being used for each layer in the network, as well as the shapes (dimensions) of the output matrices. The $z$ matrix is the linear output (weighted sum) of the input and weights/biases. The $a$ matrix is the output of the nonlinearity of the $z$ matrix. The loss is finally calculated at the end of the forward propagation based on the labels ($y$) and the prediction, in this case: $a_3$. This particular example uses the Mean Squared Error Loss, but the same idea would be true for any Loss function depending on the model and use cases.

## Backward Propagation
To the far right of the image, there is a dependency tree (that was what my multivariable calculus professor called it) that shows what the loss value depends on. Of course, the inputs ($X$) and labels ($y$) are out of our control, and we can only modify the weights and biases in the network's layers, so backpropagation will be done to calculate the gradients (partial derivatives of Loss w.r.t weights and biases). The calculus for this is shown in the middle of the image - starting from the 3rd layer down to the 1st layer since partial derivatives from further layers will be needed to calculate the gradients at previous layers by the chain rule.

# Matrix Dimensions
![nn_dims](https://github.com/krishna-tx/Neural-Network-Library-from-scratch/assets/16388863/1b4c3368-fd58-4ae9-bae0-864534577020)

This image shows how to implement the calculus of backpropagation in our vectorized neural network so that the dimensions/shapes of the matrices match. It starts from the 3rd layer down to the 1st layer and goes through the weights and bias matrices in each layer and how their gradients should be shaped to avoid incompatible dimensions during gradient descent.

## Resources I used to gain knowledge in this field:

1. Professor Andrew Ng's [Neural Network Playlist](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0)
2. PadhAI [Deep Learning](https://padhai.onefourthlabs.in/courses/dl-feb-2019)

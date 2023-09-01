# Custom Neural Network Library from Scratch (PyTorch Clone)

Shape of $X$: (n, 3) where n is the number of samples in the data and 3 is the number of features

Shape of $W_1$: (3, 4) - (# of neurons in input layer, # of neurons in first hidden layer)

Shape of $b_1$: (1, 4) - numpy broadcasting will take care of first dimension

Shape of $W_2$: (4, 2) - (# of of neurons in first hidden layer, # of neurons in second hidden layer)

Shape of $b_2$: (1, 2) - numpy broadcasting will take care of first dimension

Shape of $W_3$: (2, 1) - (# of neurons in second hidden layer, # of neurons in output layer)

Shape of $b_3$: (1, 1) - numpy broadcasting will take care of first dimension

$A_0 = X $

## Forward Propagation
$Z_1 = A_0 \cdot W_1 + b_1$ | Shape of $Z_1$: (n, 4)

$A_1 = \sigma(Z_1)$ | Shape of $A_1$: (n, 4)

$Z_2 = A_1 \cdot W_2 + b_2$ | Shape of $Z_2$: (n, 2)

$A_2 = \sigma(Z_2)$ | Shape of $A_2$: (n, 2)

$Z_3 = A_2 \cdot W_3 + b_3$ | Shape of $Z_3$: (n, 1)

$A_3 = \sigma(Z_3)$ | Shape of $A_3$: (n, 1)

## Backward Propagation

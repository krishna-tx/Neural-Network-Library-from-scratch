# Custom Neural Network Library from Scratch (PyTorch Clone)

Shape of $X$: (n, 3) where n is the number of samples in the data and 3 is the number of features

Shape of $W_1$: (3, 4) - (# of neurons in input layer, # of neurons in first hidden layer)

Shape of $b_1$: (1, 4) - numpy broadcasting will take care of first dimension

Shape of $W_2$: (4, 2) - (# of of neurons in first hidden layer, # of neurons in second hidden layer)

Shape of $b_2$: (1, 2) - numpy broadcasting will take care of first dimension

Shape of $W_3$: (2, 1) - (# of neurons in second hidden layer, # of neurons in output layer)

Shape of $b_3$: (1, 1) - numpy broadcasting will take care of first dimension

## Forward Propagation
$ a_0 = X $

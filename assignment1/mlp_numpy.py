################################################################################
# MIT License
#
# Copyright (c) 2024 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2024
# Date Created: 2024-10-28
################################################################################
"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import LinearModule, SoftMaxModule, CrossEntropyModule, ELUModule


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        alpha = 1

        ## The input layers

        self.inputlayer = LinearModule(n_inputs, 3072, True)
        self.inputELU = ELUModule(alpha)

        ## The hidden layers

        n_layers = [3072] + n_hidden
        self.layer = n_layers
        self.LinearModules = [LinearModule(n_layers[i], n_layers[i + 1], False) for i in range(len(n_layers) - 1)]
        self.ELUModules = [ELUModule(alpha) for i in range(len(n_layers) - 1)]

        ## The final output layer

        self.transformerlayer = LinearModule(n_hidden[-1], n_classes, False)
        self.SoftMaxModules = SoftMaxModule()
        self.CrossEntropyModules = CrossEntropyModule()
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        x = self.inputlayer.forward(x)
        ## Question: Do I need a ELU layer here


        for h in range(len(self.LinearModules)):
            x = self.ELUModules[h].forward(self.LinearModules[h].forward(x))


        out = self.SoftMaxModules.forward(self.transformerlayer.forward(x))

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss

        TODO:
        Implement backward pass of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        dx = self.transformerlayer.backward(self.SoftMaxModules.backward(dout))

        n_hidden = len(self.LinearModules)

        for h in range(n_hidden):
            dx = self.LinearModules[n_hidden - h - 1].backward(self.ELUModules[n_hidden - h - 1].backward(dx))
        
        #######################
        # END OF YOUR CODE    #
        #######################

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass from any module.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Iterate over modules and call the 'clear_cache' function.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        #self.inputlayer.clear_cache()
        for h in range(len(self.LinearModules)):
            self.LinearModules[h].clear_cache()
            self.ELUModules[h].clear_cache()
        self.transformerlayer.clear_cache()
        self.SoftMaxModules.clear_cache()

        pass
        #######################
        # END OF YOUR CODE    #
        #######################

#!/usr/bin/python3

import constants as constants

class bitBrick():
    """ Defining a class for a bitBrick """
    def __init__(self, inputs, weights, ID):
        ##
        self.inputs = inputs
        self.weights = weights
        self.ID = ID
        assert ((self.inputs >> 2) == 0), "Inputs are wider than 2 bits"
        assert ((self.weights >> 2) == 0), "Weights are wider than 2 bits"

    def displayContents(self):
        ##
        print("The input pixel value is {}".format(self.inputs))
        print("The weight pixel value is {}".format(self.weights))
        print("The ID of the pixel is {}".format(self.ID))

    def mult2(self, inputs, weights):
        ##
        output = self.inputs * self.weights
        latency = constants.latency_bitBrick
        return [output, latency]

    def displayOutputs(self):
        [output, latency] = self.mult2(self.inputs, self.weights)
        print("The output of multiplication using bit-bricks is {}".format(self.output))
        print("The latency for multiplication using bit-bricks is {}".format(self.latency))

##
if __name__ == '__main__':
    ##
    bitBrick_0_0 = bitBrick(2, 3, '0_0')
    bitBrick_0_0_Op = bitBrick_0_0.mult2(2, 3)
    bitBrick_0_0.displayContents()


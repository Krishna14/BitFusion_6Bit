#!/usr/bin/python3

import re
import constants

# Check the .py file for the first layer and extract the following information
#   1. Input image dimensions (M x N)
#   2. Number of filters (C)
#   3. Kernel Size
#   4. Activation type
#   5. Whether pooling is being done? If yes, what type of pooling - max, average
layerNumber = constants.layer_to_be_accelerated
obuf_size = 256
weight_resolution = constants.weightResolution
input_resoltion = constants.inputPixelResolution

def parser(inputFile):
    # inputFile  = open("keras_implementation.py", "r")
    outFile = open("ISA_implementation.txt", "w")
    outFile2 = open("keras_intermediate.txt", "w")
    
    # Values to be read in for modifying the code
    
    # Definite patterns that could be extracted from the code!
    count_pattern1 = 0
    count_pattern2 = 0
    count_pattern3 = 0
    count_pattern4 = 0
    count_pattern5 = 0
    
    # We can do it!
    pattern1 = "layers"
    pattern2 = "Conv2D"
    pattern3 = "AveragePooling"
    pattern4 = "Flatten"
    pattern5 = "Dense"
    pattern6 = "poolingtype"
    
    # Computing the total number of layers, conv2D, 
    # averagePooling, Flattening, Dense and the poolingType
    # of all the different layers
    for line in open("keras_implementation.py", "r"):
        if re.search(pattern1, line):
            count_pattern1 += 1
            if re.search(pattern2, line):
                count_pattern2 += 1
            elif re.search(pattern3, line):
                count_pattern3 += 1
            elif re.search(pattern4, line):
                count_pattern4 += 1
            elif re.search(pattern5, line):
                count_pattern5 += 1
    
    outFile2.write("There are " + str(count_pattern1) + " layers in the design \n")
    # print(count_pattern1)
    outFile2.write("Amongst the " + str(count_pattern1) + " layers, there are " + \
            str(count_pattern2) + " Conv2D layers in the design \n")
    # print(count_pattern2)
    outFile2.write("Amongst the " + str(count_pattern1) + " layers, there are " + \
            str(count_pattern3) + " pooling layers in the design \n")
    # print(count_pattern3)
    outFile2.write("Amongst the " + str(count_pattern1) + " layers, there are " + \
            str(count_pattern4) + " flattening layers in the design \n")
    # print(count_pattern4)
    outFile2.write("Amongst the " + str(count_pattern1) + " layers, there are " + \
            str(count_pattern5) + " flattening layers in the design \n")
    # print(count_pattern5)
    
    # Now that we have computed the different types of layers, 
    # let's compute the image dimensions, kernel size and activations
    
    kernelSize_pattern = "kernel_size"
    pooling_pattern = "Pooling2D"
    filterCount_pattern = "filters"
    activation_pattern = "activation"
    inputShape_pattern = "input_shape"
    model_pattern = "Sequential"
    
    kernelSize_values = []
    filterCount_values = []
    activation_values = []
    inputShape_values = []
    poolingType_values = []
    modelType_values = []
    
    found_kernelSize = False
    found_filterCount = False
    found_activation = False
    found_inputShape = False
    found_poolingType = False
    found_modelPattern = False
    
    # Extract kernelSize information for all the layers
    for line in open("keras_implementation.py", "r"):
        if not found_kernelSize:
            counter_kernelSize = 0
            if re.search(kernelSize_pattern, line):
                for part in line.split('( | )'):
                    counter_kernelSize += 1
                    if kernelSize_pattern in part:
                        # We will need two values
                        kernelSize_values.append(line.split()[counter_kernelSize])
                        kernelSize_values.append(line.split()[counter_kernelSize + 1])
    
    found_kernelSize = True
    # print(kernelSize_values)
    
    # Extract the required information!
    kernelSize_val = ""
    for value in kernelSize_values:
        kernelSize_val += str(value)
    
    # Extract filterCount information for all the layers
    for line in open("keras_implementation.py", "r"):
        if not found_filterCount:
            counter_filter = 0
            if re.search(filterCount_pattern, line):
                for part in line.split('='):
                    if filterCount_pattern in part:
                        filterCount_values.append(line.split()[counter_filter])
    
    found_filterCount = True
    # print(filterCount_values)
    
    # Extract activation information for all the layers
    for line in open("keras_implementation.py", "r"):
        if not found_activation:
            counter_activation = 0
            if re.search(activation_pattern, line):
                for part in line.split('='):
                    if activation_pattern in part:
                        index = line.split('=').index(part)
                        activation_values.append(line.split('=')[index+1])
    
    found_activation = True
    # print(activation_values)
    
    # Extract the input dimensions of the image that needs to be used -
    for line in open("keras_implementation.py", "r"):
        if not found_inputShape:
            counter_inputShape = 0
            if re.search(inputShape_pattern, line):
                for part in line.split('='):
                    if inputShape_pattern in part:
                        index = line.split('=').index(part)
                        inputShape_values.append(line.split('=')[index+1])
    
    found_inputShape = True
    # print(inputShape_values)
    
    # Extract the type of pooling that's happening in the layers
    for line in open("keras_implementation.py", "r"):
        if not found_poolingType:
            if re.search(pooling_pattern, line):
                poolingType_values.append(line.split('.')[-1])
    
    found_poolingType = True
    # print(poolingType_values)
    
    
    # Start checking for the pattern "keras.Sequential() or keras.Functional"
    for line in open("keras_implementation.py", "r"):
        if not found_modelPattern:
            if re.search(model_pattern, line):
                modelType_values.append(model_pattern)
                found_modelPattern = True
    
    # print(modelType_values)
    
    if modelType_values[0] == "Sequential":
        outFile.write("setup " + str(constants.inputPixelResolution) + " " + str(constants.weightResolution))
    
    outFile.close()
    
    line_number = []
    with open("keras_implementation.py") as myfile:
        for num, line in enumerate(myfile, 1):
            if re.search(pattern2, line):
                line_number.append(num)
    
    count = len(line_number)
    # print(line_number)
    
    for i in range(count):
        len_line_number = len(line_number)
        line_number.append(0)
        line_number[i+2:] = line_number[i+1:len_line_number]
        line_number[i+1] = line_number[i] + 1
   
    # Post processing 
    inputShape_values = str(str(inputShape_values).rstrip('\n'))
    # inputShape_values = list(inputShape_values)
    inputShape = inputShape_values[9] + ',' + inputShape_values[3:5] + ',' + inputShape_values[6:8]
    #
    print( ( (int(inputShape.split(',')) )

    # Computation of the output!
    # kernelSize = str(kernelSize_values[0] + kernelSize_values[1]).split('=')[1].split(',')[0:2]
    # print(kernelSize_values[0])
    # print(kernelSize_values[0])
    # kernelSize = ((int(kernelSize[0].split('(')[1]), int(kernelSize[1].split(')')[0])))

    # Activation
    activate = activation_values[0].split(',')[0]

    # FilterCount_values
    filterCount = int(filterCount_values[0].split('(')[2].split('=')[1].split(',')[0])
    
    # Compute the newly computed values!
    return [ inputShape, kernelSize, activate, filterCount ]


if __name__=='__main__':
    fileName = "keras_implementation.py"
    print(parser(fileName))


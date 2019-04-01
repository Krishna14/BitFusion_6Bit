#!/usr/bin/python3

import re
import constants

inFile = open("keras_implementation.py", "rt")
outFile = open("keras_modified_accelerated.txt", "w")

contents = inFile.readlines()
inFile.close()

pattern = "Conv2D"

line_numbers = []
# After reading the file line by line
for i in range(len(contents)):
    if re.search(pattern, contents[i]):
        line_numbers.append((i+1, i+2))    # Line number = index_number + 1

print(line_numbers)

layerNumber = constants.layer_to_be_accelerated

lineNumber_mapped = line_numbers[layerNumber - 1]
print(lineNumber_mapped)
index = -2
for i in range(len(contents)):
    if ( re.search(pattern, contents[i]) or (i == index+1) ):
        index = i
        outFile.write("##################")
        continue
    else:
        outFile.write(contents[i])

#!/usr/bin/python3

# import array
# import numpy as np

class shiftAdd():
    """Class to implement a shiftAdd"""
    def __init__(self, name, input_objs, parent_name, level=0):
        """Constructor to initialize the shiftAmts and inputs"""
        # print("shiftAdd.py <-__init__: inputs="+"self:"+str(self)+",")
        self.name = name
        # input_bb_obj = [[x,y],[a,b]] is the format
        assert type(input_objs) == list, 'shiftAdd - expecting input_obj to be a list'
        self.inputObjs = input_objs
        self.outputs = []
        self.shiftAddLevel = level
        # self.outputObj = output_obj
        # fu_name in case of level 0 and 1 and colName in level 2
        self.fuName = parent_name
        self.status = 'free'
        # shift amounts are represented here in format
        # [[shift for 0_0, shift_for_0_1],
        #  [shift_for 1_0, shift_for_1_1]]
        self.level1_shifts = [[4,0],[8,4]]

    def execShiftAdd(self):
        assert len(self.inputObjs) == 2, 'shiftAdd - expecting inputObjs to be arranged as matrix'
        max_len_output_inputObjs = max(len(self.inputObjs[0][1].outputs),
                                       len(self.inputObjs[0][0].outputs),
                                       len(self.inputObjs[1][1].outputs),
                                       len(self.inputObjs[1][0].outputs))
        output = [0 for x in range(max_len_output_inputObjs)]

        if self.shiftAddLevel == 0:
            for x in range(len(self.inputObjs[0][1].outputs)):
                output[x] += (self.inputObjs[0][1].outputs.pop(0) << 0)

            for x in range(len(self.inputObjs[0][0].outputs)):
                output[x] += (self.inputObjs[0][0].outputs.pop(0) << 2)

            for x in range(len(self.inputObjs[1][1].outputs)):
                output[x] += (self.inputObjs[1][1].outputs.pop(0) << 2)

            for x in range(len(self.inputObjs[1][0].outputs)):
                output[x] += (self.inputObjs[1][0].outputs.pop(0) << 4)
        elif self.shiftAddLevel == 1:
            for x in range(len(self.inputObjs[0][1].outputs)):
                output[x] += (self.inputObjs[0][1].outputs.pop(0) << self.level1_shifts[0][1])

            for x in range(len(self.inputObjs[0][0].outputs)):
                output[x] += (self.inputObjs[0][0].outputs.pop(0) << self.level1_shifts[0][0])

            for x in range(len(self.inputObjs[1][1].outputs)):
                output[x] += (self.inputObjs[1][1].outputs.pop(0) << self.level1_shifts[1][1])

            for x in range(len(self.inputObjs[1][0].outputs)):
                output[x] += (self.inputObjs[1][0].outputs.pop(0) << self.level1_shifts[1][0])


        self.outputs = output
        self.status = 'complete'

    def execAdd(self):
        # input_obj will be a list of length equal to #FU in a column
        assert self.shiftAddLevel == 2, 'shiftAdd - execAdd can only be run for level2'
        output = 0
        # would indicate how many shiftAdds had output
        # TODO could check the status of inputObj as well
        output_produced_count = 0
        for x in self.inputObjs:
            if len(x.shiftAddList_l1[0].outputs) != 0:
                output_produced_count += 1
                output += x.shiftAddList_l1[0].outputs.pop(0)

        if (output_produced_count != 0):
            self.outputs.append(output)
            self.status = 'complete'

    def computeSum(self, shiftAmts, inputs):
        """Returns the sum based on the inputs"""
        print("shiftAdd.py<-computeSum: inputs="+"self:"+str(self)+",")
        Sum = 0
        for shiftAmt in shiftAmts:
            index = shiftAmts.index(shiftAmt)
            Sum = Sum + (inputs[index] << shiftAmt)
        return Sum

    def displayAttributes(self):
        print("shiftAdd.py <- displayAttributes of {}-{} level:{}, status:{}, outputs:{}".\
            format(self.fuName, self.name, self.shiftAddLevel, self.status, self.outputs))



if __name__ == '__main__':
    SA0 = shiftAdd([0, 2, 2, 4], [2, 2, 4, 2], 0)
    SA1 = shiftAdd([0, 4, 4, 8], [2, 2, 4, 2], 1)

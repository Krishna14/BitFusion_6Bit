import numpy as np

class bitBrickCommands():
    #def __init__(self):

    #def load8(self, arguments):
        #load8 <mem location to read from> <#bytes to read>

    def mul2(self, arguments):
        #mul2 <dst reg> <src1> <src2>
        assert type(arguments) == list, 'mul2 - arguments not of type list'
        assert len(np.array(arguments)) == 2, 'mul2 - need 2 arguments'
        assert abs(arguments[0]) <= 3, 'mul2 - arguments[0] greater than 2 bits'
        assert abs(arguments[1]) <= 3, 'mul2 - arguments[1] greater than 2 bits'
        return arguments[0]*arguments[1]

    def mul4(self, arguments):
        #mul2 <dst reg> <src1> <src2>
        assert type(arguments) == list, 'mul2 - arguments not of type list'
        assert len(np.array(arguments)) == 2, 'mul2 - more than 2 arguments passed'
        assert abs(arguments[0]) <= 3, 'mul2 - arguments[0] greater than 2 bits'
        assert abs(arguments[1]) <= 3, 'mul2 - arguments[1] greater than 2 bits'
        return arguments[0]*arguments[1]

    def mul8(self, arguments):
        #mul2 <dst reg> <src1> <src2>
        assert type(arguments) == list, 'mul2 - arguments not of type list'
        assert len(np.array(arguments)) == 2, 'mul2 - more than 2 arguments passed'
        assert abs(arguments[0]) <= 3, 'mul2 - arguments[0] greater than 2 bits'
        assert abs(arguments[1]) <= 3, 'mul2 - arguments[1] greater than 2 bits'
        return arguments[0]*arguments[1]

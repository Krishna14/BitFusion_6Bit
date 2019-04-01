#!/usr/bin/python3
# import array
import numpy as np
from bitBrickCommands import *

class bitBrick():
    """ Class to implement a BitBrick """
    def __init__(self, name, fu_name):
        print("BitBrick.py <- __init__ {}-{}".format(fu_name, name))
        self.name = name
        self.fuName = fu_name
        self.status = "free"
        self.commands = []
        self.outputs = []

    def addCommand(self, command):
        print("bitBrick.py <- addCommand {}-{}, command:{}".format(self.fuName, self.name, command))
        self.commands.append(command)
        self.status = 'busy'

    def parseCommand(self, command):
        command_blocks = command.split()
        assert len(np.array(command_blocks)) == 3, 'bitBrick - malformed command received'

        operation = command_blocks[0]
        input_1 = np.int8(command_blocks[1])
        input_2 = np.int8(command_blocks[2])
        return [operation, input_1, input_2]



    def execCommand(self):
        while (len(self.commands) != 0):
            current_command = self.commands.pop(0)
            print("bitBrick.py <- execCommand in {}-{}: Executing: {}".\
                    format(self.fuName, self.name, current_command))
            command_blocks = self.parseCommand(current_command)

            if (command_blocks[0] == 'mul2'):
                self.outputs.append(bitBrickCommands.mul2(self,command_blocks[1:]))
        self.status = 'free'


    def displayAttributes(self):
        """ Function to print the attributes of an object of BitBrick class"""
        # print("BitBrick.py <- displayAttributes: inputs="+"self:"+str(self)+",")
        print("bitBrick.py <- displayAttributes: {}-{}, status:{}, commands:{}, outputs:{}".\
              format(self.fuName, self.name, self.status, self.commands, self.outputs))


if __name__ == "__main__":
    BB0 = bitBrick('BB0', 'FF0')
    BB0.addCommand("mul2 2 -2")
    BB0.addCommand("mul2 1  3")
    BB0.addCommand("mul2 3 3")
    BB0.displayAttributes()
    BB0.execCommand()
    # print(BB0.outputs)
    BB0.displayAttributes()

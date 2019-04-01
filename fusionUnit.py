#!/usr/bin/python3

import bitBrick
import shiftAdd
import numpy as np
import re
from memory import *
import utils as utils

class fusionUnit():
    """Class for representing the fusionUnit"""
    def __init__(self, name, wbuf_name, wbuf_obj, \
                 ibuf_name, ibuf_obj, \
                bit_width):
        """ Constructor class for the Fusion Unit """

        self.name = name
        # provide connected buf names
        self.wBufName = wbuf_name
        self.wBufObj = wbuf_obj
        self.iBufName = ibuf_name
        self.iBufObj = ibuf_obj
        # self.oBufName = obuf_name
        # self.oBufObj = obuf_obj
        self.bitWidth = bit_width
        self.commands = []
        self.outputs = []
        # TODO make these rows customizable for experiments
        self.rows = 4
        self.cols = 4
        self.BB_list = [[0 for x in range(self.rows)] for y in range(self.cols)] # Used to store the BB objects
        # level 0 shiftAdd
        self.shiftAddList = [[0 for x in range(int(self.rows/2))] for y in range(int(self.cols/2))]
        # shiftAdd of 1st level which take input from other shiftAdd
        self.shiftAddList_l1 = []

        for i in range(self.rows):
            for j in range(self.cols):
                BB_name = "BB_"+str(i)+"_"+str(j)
                self.BB_list[i][j] = bitBrick.bitBrick(BB_name, self.name)

        # TODO make this automatic along with different levels
        # assign input bitBricks for each shiftAdd
        temp_row = 0
        for i in range(int(self.rows/2)):
            temp_col = 0

            for j in range(int(self.cols/2)):
                shiftAdd_name = "SA_"+str(i)+"_"+str(j)
                input_bb_list = []
                input_bb_list.append([self.BB_list[temp_row][temp_col], self.BB_list[temp_row][temp_col+1]])
                input_bb_list.append([self.BB_list[temp_row+1][temp_col], self.BB_list[temp_row+1][temp_col+1]])

                temp_col += 2
                self.shiftAddList[i][j] = shiftAdd.shiftAdd(shiftAdd_name, input_bb_list, self.name, 0)
            temp_row += 2

        input_sa_list = []
        input_sa_list.append([self.shiftAddList[0][0], self.shiftAddList[0][1]])
        input_sa_list.append([self.shiftAddList[1][0], self.shiftAddList[1][1]])
        self.shiftAddList_l1.append(shiftAdd.shiftAdd("SA_2_0", input_sa_list, self.name, 1))


    def addCommand(self, command):
        if type(command) == list:
            self.commands += command
        else:
            self.commands.append(command)

    # BB_0_1:mul2 A/N 0x400
    def parseCommand(self, command):
        command_type = ""
        if "shconfig" in command:
            command_type = "l1_shiftAdd_config"
            command_blocks = command.split()
            assert command_blocks[0] == 'shconfig', 'fusionUnit - misleading shconfig command received'
            l1_shiftAdd_pattern_list = [[int(command_blocks[1]), int(command_blocks[2])],
                                        [int(command_blocks[3]), int(command_blocks[4])]]
            return [command_type, l1_shiftAdd_pattern_list]
        elif "mul2" in command:
            command_type = 'mul2'
            command_blocks = command.split(':')
            assert len(command_blocks) == 2, 'fusionUnit - malformed command received'
            pattern = re.compile('^BB_(\d+)_(\d+)$')
            matches = pattern.match(command_blocks[0])
            assert matches, 'fusionUnit - malformed target BB name received'

            bb_command = command_blocks.pop(1).split()
            assert len(bb_command) == 3, 'fusionUnit - malformed command for bitBrick received'
            command_blocks.append(bb_command[0])
            command_blocks.append(bb_command[1])
            command_blocks.append(bb_command[2])
            # return [<BB row num> <BB col num> <op> <memLoc of operand1> <memLoc of operand2>]
            return [command_type, matches.group(1), matches.group(2)] + command_blocks[1:]


    def sendCommand(self):
        # TODO reduce buf accesses by making most of data read in case of
        # following dict is to imitate mux distribution pattern of a buf access
        # so, from single access, data can be divided into chunks depending on bitWidth
        ibuf_data = {}
        wbuf_data = {}

        while len(self.commands) !=0:
            command_blocks = self.parseCommand(self.commands.pop(0))
            command_type = command_blocks.pop(0)
            if command_type == 'l1_shiftAdd_config':
                self.shiftAddList_l1[0].level1_shifts = command_blocks[0]
            elif command_type == 'mul2':
                bb_row = int(command_blocks[0])
                bb_col = int(command_blocks[1])
                bb_op = command_blocks[2]

                ibuf_byte = command_blocks[3].split('-') #of the form '0x400-2'
                ibuf_bit_start = int(ibuf_byte[1])
                ibuf_byte = int(ibuf_byte[0], 16) #of the form '0x400'

                wbuf_byte = command_blocks[4].split('-') #of the form '0x400-2'
                wbuf_bit_start = int(wbuf_byte[1])
                wbuf_byte = int(wbuf_byte[0], 16) #of the form '0x400'

                if ibuf_byte not in ibuf_data.keys():
                    ibuf_data[ibuf_byte] = self.iBufObj.load_mem(ibuf_byte, 1)[0]

                if wbuf_byte not in wbuf_data.keys():
                    wbuf_data[wbuf_byte] = self.wBufObj.load_mem(wbuf_byte, 1)[0]

                ibuf_operand = (ibuf_data[ibuf_byte] >> ibuf_bit_start) & 0x3
                wbuf_operand = (wbuf_data[wbuf_byte] >> wbuf_bit_start) & 0x3
                # ibuf_operand = 0
                # if (ibuf_bit_start + self.bitWidth - 1) <= 8:
                #     ibuf_operand = ((ibuf_data[ibuf_byte] << ibuf_bit_start) & 0xff) >> (8 - self.bitWidth)
                # else:
                #     if (ibuf_byte + 1) not in ibuf_data.keys():
                #         ibuf_data[ibuf_byte+1] = self.iBufObj.load_mem(ibuf_byte+1, 1)[0]
                #
                #     ibuf_operand = ((ibuf_data[ibuf_byte]  << ibuf_bit_start) & 0xff) >> ibuf_bit_start
                #     ibuf_operand = ibuf_operand << (self.bitWidth - 8 + ibuf_bit_start)
                #     ibuf_operand = ibuf_operand | (ibuf_data[ibuf_byte+1] >> (8 - self.bitWidth + (8 - ibuf_bit_start)))

                # wbuf_operand = 0
                # if (wbuf_bit_start + self.bitWidth - 1) <= 8:
                #     wbuf_operand = ((wbuf_data[wbuf_byte] << wbuf_bit_start) & 0xff) >> (8 - self.bitWidth)
                # else:
                #     if (wbuf_byte + 1) not in wbuf_data.keys():
                #         wbuf_data[wbuf_byte+1] = self.wBufObj.load_mem(wbuf_byte+1, 1)[0]
                #
                #     wbuf_operand = ((wbuf_data[wbuf_byte]  << wbuf_bit_start) & 0xff) >> wbuf_bit_start
                #     wbuf_operand = wbuf_operand << (self.bitWidth - 8 + wbuf_bit_start)
                #     wbuf_operand = wbuf_operand | (wbuf_data[wbuf_byte+1] >> (8 - self.bitWidth + (8 - wbuf_bit_start)))

                self.BB_list[bb_row][bb_col].addCommand(bb_op + " " + str(ibuf_operand) + " " + str(wbuf_operand))

    # executes the commands in bitBrick and they are now placed in it's output
    def execCommand(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if (self.BB_list[i][j].status == 'busy'):
                    self.BB_list[i][j].execCommand()

        # level 1 shiftAdd
        for i in range(int(self.rows/2)):
            for j in range(int(self.cols/2)):
                self.shiftAddList[i][j].execShiftAdd()
                self.shiftAddList[i][j].displayAttributes()

        self.shiftAddList_l1[0].execShiftAdd()
        self.shiftAddList_l1[0].displayAttributes()

    def getBusyBitBricks(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if (self.BB_list[i][j].status == 'busy'):
                    print("fusionUnit.py <- getBusyBitBricks of {} has {} busy with commands: {}".\
                          format(self.name, self.BB_list[i][j].name, self.BB_list[i][j].commands))



    def displayAttributes(self):
        print("fusionUnit.py<-displayAttributes: inputs="+"self:"+str(self))
        print("The input pixel's value is {}".format(self.input_pixel))
        print("The input weight's value is {}".format(self.weight))
        print("The input pixel's resolution is {}".format(self.input_pixel_resolution))
        print("The input weight's resolution is {}".format(self.weight_resolution))


if __name__=="__main__":
    wbuf_obj = memory('wbuf0',1024)
    wbuf_obj.store_mem(0x0, [9,1,2,3,4,5,6,7])

    ibuf_obj = memory('ibuf0', 1024)
    ibuf_obj.store_mem(0x0, [15,1,1,1,1,1,1,1])

    #obuf_obj = memory('obuf0', 1024)

    FF0 = fusionUnit('FF0', 'wbuf0', wbuf_obj, 'ibuf0', ibuf_obj, 2)
    # W = 165, I = 231, 231*165
    commands = []
    commands.append("BB_0_0:mul2 0x0-0 0x0-6")
    commands.append("BB_0_1:mul2 0x0-0 0x0-4")
    commands.append("BB_0_2:mul2 0x0-0 0x0-2")
    commands.append("BB_0_3:mul2 0x0-0 0x0-0")
    commands.append("BB_1_0:mul2 0x0-2 0x0-6")
    commands.append("BB_1_1:mul2 0x0-2 0x0-4")
    commands.append("BB_1_2:mul2 0x0-2 0x0-2")
    commands.append("BB_1_3:mul2 0x0-2 0x0-0")
    commands.append("BB_2_0:mul2 0x0-4 0x0-6")
    commands.append("BB_2_1:mul2 0x0-4 0x0-4")
    commands.append("BB_2_2:mul2 0x0-4 0x0-2")
    commands.append("BB_2_3:mul2 0x0-4 0x0-0")
    commands.append("BB_3_0:mul2 0x0-6 0x0-6")
    commands.append("BB_3_1:mul2 0x0-6 0x0-4")
    commands.append("BB_3_2:mul2 0x0-6 0x0-2")
    commands.append("BB_3_3:mul2 0x0-6 0x0-0")
    FF0.addCommand(commands)
    #FF0.addCommand(["BB_2_2:mul2 0x0-0 0x2-4", "BB_3_3:mul2 0x4-4 0x3-4"])
    FF0.sendCommand()
    FF0.getBusyBitBricks()
    FF0.execCommand()


    #FF0.computeProdLatency()

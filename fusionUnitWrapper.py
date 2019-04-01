from bitBrick import *
from memory import *
from fusionUnit import *
import utils as utils
from shiftAdd import *
from pprint import pprint

class fusionUnitWrapper():
    def __init__(self, ibuf_size, wbuf_size, obuf_size):
        # TODO decide how many fusion unit should be there in a row
        self.fuRows = 16
        self.fuCols = 16

        self.fuData = {}
        # stores obj of obuf of a column
        self.obuf_obj = []
        self.obufSize = obuf_size

        self.ibufSize = ibuf_size
        self.wbufSize = wbuf_size
        # which addr to store the output of level2 add in obuf
        # making it a list of lists with each list representing 1 cycle
        self.obuf_write_addr = []
        self.commands = []
        # stores objs of fu in a column
        self.col_fu_obj = [[0 for x in range(self.fuRows)] for y in range(self.fuCols)]
        # stores objs of adders of each column
        self.shiftAdd_l2_objs = []


        for i in range(self.fuRows):
            for j in range(self.fuCols):
                fu_name = "FU_"+str(i)+"_"+str(j)
                fu_wbuf_name = 'WBUF_'+str(i)+"_"+str(j)
                fu_ibuf_name = 'IBUF_'+str(i)+"_"+str(j)

                fu_ibuf_obj = memory(fu_ibuf_name, self.ibufSize)
                fu_wbuf_obj = memory(fu_wbuf_name, self.wbufSize)
                fu_obj = fusionUnit(fu_name, fu_wbuf_name, fu_wbuf_obj,\
                                    fu_ibuf_name, fu_ibuf_obj, 2)

                self.fuData[fu_name] = {}
                self.fuData[fu_name]['fu_obj'] = fu_obj
                self.fuData[fu_name]['fu_ibuf_obj'] = fu_ibuf_obj
                self.fuData[fu_name]['fu_wbuf_obj'] = fu_wbuf_obj


        for j in range(self.fuCols):
            for i in range(self.fuRows):
                self.col_fu_obj[j][i] = self.fuData[utils.getNameString('FU',i, j)]['fu_obj']

        for j in range(self.fuCols):
            col_sa_obj = shiftAdd('SA_x_'+str(j), self.col_fu_obj[j], 'BitFusion', 2)
            self.shiftAdd_l2_objs.append(col_sa_obj)

            obuf_name = 'OBUF_x_'+str(j)
            obuf_obj = memory(obuf_name, self.obufSize)
            self.obuf_obj.append(obuf_obj)
            self.obuf_write_addr.append(-1)

    # FU_0_1:BB_0_1:mul2 0x400-0 0x420-0
    # staddr obuf_x_0 0x400
    def parseCommand(self, command):
        command_type = ""
        if re.match('^#', command) is not None:
            command_type = 'comment'
            return [command_type]
        elif "staddr OBUF_x_" in command:
            command_type = 'staddr'
            pattern = re.compile('^staddr OBUF_x_(\d+) 0x(.*)$')
            matches = pattern.match(command)
            assert matches, 'fusionUnitWrapper - malformed stdaddr OBUF_x_ command received'

            return [command_type, matches.group(1), matches.group(2)]
        elif "mul2" in command:
            command_type = 'mul2'
            command_blocks = command.split(':')
            print(command)
            assert len(command_blocks) == 3, 'fusionUnitWrapper - malformed command received'
            command_blocks[1] += ":"+command_blocks.pop(2)
            pattern = re.compile('^FU_(\d+)_(\d+)$')
            matches = pattern.match(command_blocks[0])
            assert matches, 'fusionUnitWrapper - malformed target FU name received'

            # return [<BB row num> <BB col num> <op> <memLoc of operand1> <memLoc of operand2>]
            return [command_type, matches.group(1), matches.group(2)] + command_blocks[1:]
        elif "shconfig" in command:
            command_type = 'l1_shiftAdd_config'
            command_blocks = command.split(":")
            assert len(command_blocks) == 2, 'fusionUnitWrapper - maformed shconfig command received'
            pattern = re.compile('^FU_(\d+)_(\d+)$')
            matches = pattern.match(command_blocks[0])
            assert matches, 'fusionUnitWrapper - malformed target FU name received'

            return [command_type, matches.group(1), matches.group(2), command_blocks[1]]
        else:
            assert 0 > 1, 'fusionUnitWrapper - command not yet handled'

    def addCommand(self, command):
        if type(command) == list:
            self.commands += command
        else:
            self.commands.append(command)

    def sendCommand(self):
        while len(self.commands) != 0:
            command_blocks = self.parseCommand(self.commands.pop(0))
            command_type = command_blocks.pop(0)

            if command_type == 'comment':
                continue
            elif command_type == 'staddr':
                col = int(command_blocks[0])
                addr = int('0x'+command_blocks[1], 16)
                self.obuf_write_addr[col] = addr
                continue
            elif command_type in ['mul2', 'l1_shiftAdd_config']:
                fu_row = command_blocks[0]
                fu_col = command_blocks[1]
                fu_command = command_blocks[2]

                self.fuData[utils.getNameString('FU',fu_row,fu_col)]['fu_obj'].addCommand(fu_command)
                self.fuData[utils.getNameString('FU', fu_row, fu_col)]['fu_obj'].sendCommand()
            else:
                assert 0 > 1, 'fusionUnitWrapper - command not yet handled'

        # if command_type == 'mul2':
        #     print("rohit")
        #     for i in range(self.fuRows):
        #         for j in range(self.fuCols):
        #             self.fuData[utils.getNameString('FU', i, j)]['fu_obj'].sendCommand()

    def execCommand(self):
        for i in range(self.fuRows):
            for j in range(self.fuCols):
                self.fuData[utils.getNameString('FU',i,j)]['fu_obj'].execCommand()

        for j in range(self.fuCols):
            self.shiftAdd_l2_objs[j].execAdd()
            self.shiftAdd_l2_objs[j].displayAttributes()

            if len(self.shiftAdd_l2_objs[j].outputs) != 0:
                assert self.obuf_write_addr[j] != -1, 'fusionUnitWrapper - write address of this obuf is wrong'
                print(self.shiftAdd_l2_objs[j].outputs)
                self.obuf_obj[j].store_mem(self.obuf_write_addr[j], \
                                           utils.align_num_to_byte(self.shiftAdd_l2_objs[j].outputs.pop(0)))




    def getBusyBitBricks(self):
        for i in range(self.fuRows):
            for j in range(self.fuCols):
                self.fuData[utils.getNameString('FU',i,j)]['fu_obj'].getBusyBitBricks()

if __name__ == "__main__":
    DD = fusionUnitWrapper(256, 128, 1024)

    input_quantization = 6
    weight_quantization =  6

    input_image = list(np.ones((1,5,5), dtype=int).flatten() * ((255 & 0xff) >> (8 - input_quantization)))
    kernel = list(np.ones((1,2,2), dtype=int).flatten() * ((127 & 0xff) >> (8 - weight_quantization)))

    for rows in range(DD.fuRows):
        for cols in range(DD.fuCols):
            DD.fuData[utils.getNameString('FU', rows, cols)]['fu_ibuf_obj'].store_mem(0x0, input_image)
            DD.fuData[utils.getNameString('FU', rows, cols)]['fu_wbuf_obj'].store_mem(0x0, kernel)

    instr_file_pattern = re.compile('^instr_cycle(\d+).txt')

    count = 1
    while os.path.exists("./cycle_instr_dir/"+'instr_cycle'+str(count)+".txt"):
        print("FUSIONUNITWRAPPER: EXECUTING INSTRUCTIONS FOR CYCLE:"+str(count))
        with open("./cycle_instr_dir/"+'instr_cycle'+str(count)+".txt") as f:
            count += 1
            command = f.read().splitlines()
            DD.addCommand(command)
            DD.sendCommand()
            DD.getBusyBitBricks()
            DD.execCommand()

    # with open('instr_2.txt') as f:
    #     command = f.read().splitlines()
    # command = ["staddr OBUF_x_0 0x10",
    #            "FU_0_0:shconfig 0 0 0 0",
    #            "FU_0_0:BB_0_0:mul2 0x0-6 0x0-0",
    #            "FU_0_0:BB_0_1:mul2 0x0-6 0x0-2",
    #            "FU_0_0:BB_0_2:mul2 0x0-6 0x0-4",
    #            "FU_0_0:BB_0_3:mul2 0x0-6 0x0-6",
    #            "FU_0_0:BB_1_0:mul2 0x0-4 0x0-0",
    #            "FU_0_0:BB_1_1:mul2 0x0-4 0x0-2",
    #            "FU_0_0:BB_1_2:mul2 0x0-4 0x0-4",
    #            "FU_0_0:BB_1_3:mul2 0x0-4 0x0-6",
    #            "FU_0_0:BB_2_0:mul2 0x0-2 0x0-0",
    #            "FU_0_0:BB_2_1:mul2 0x0-2 0x0-2",
    #            "FU_0_0:BB_2_2:mul2 0x0-2 0x0-4",
    #            "FU_0_0:BB_2_3:mul2 0x0-2 0x0-6",
    #            "FU_0_0:BB_3_0:mul2 0x0-0 0x0-0",
    #            "FU_0_0:BB_3_1:mul2 0x0-0 0x0-2",
    #            "FU_0_0:BB_3_2:mul2 0x0-0 0x0-4",
    #            "FU_0_0:BB_3_3:mul2 0x0-0 0x0-6",
    #            "staddr OBUF_x_1 0x0",
    #            "FU_15_1:BB_1_1:mul2 0x0-4 0x0-2"]
    # DD.fuData[utils.getNameString('FU',0,0)]['fu_ibuf_obj'].store_mem(0x0, [231,1,1,1,1,1,1,1])
    # DD.fuData[utils.getNameString('FU',0,0)]['fu_wbuf_obj'].store_mem(0x0, [165,1,2,3,4,5,6,7])
    # DD.fuData[utils.getNameString('FU',15,1)]['fu_ibuf_obj'].store_mem(0x0, [231,1,1,1,1,1,1,1])
    # DD.fuData[utils.getNameString('FU',15,1)]['fu_wbuf_obj'].store_mem(0x0, [165,1,2,3,4,5,6,7])
    # DD.addCommand(command)
    # DD.sendCommand()
    # DD.getBusyBitBricks()
    # DD.execCommand()





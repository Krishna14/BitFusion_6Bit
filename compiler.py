#!/usr/bin/python3

import os
import argparse
import json
import numpy as np

class compiler():
    """ Defines a compiler class """

    def __init__(self, input_image, kernel_image, input_size, kernel_size, input_quantization=8,\
            weight_quantization=8, bitbrick_dim=(4, 4), fusionunit_dim=(16, 16), ibuf_size=256,\
            obuf_size=1024, wbuf_size=128):
        """ Used to initialize the compiler for the architecture """
        """ Inputs are the following - 
            1. Input image              ---> Input image
            2. Kernel image             ---> Kernel image
            3. Entire simulation data   ---> Simulation data
            4. Entire memory data       ---> Memory data
            5. Window Number            ---> Window number
            6. Cycles used              ---> Number of cycles consumed
            7. Input image shape        ---> Shape of the input image
            8. Kernel shape             ---> Shape of the input kernel
            9. bitFusion Rows           ---> bitFusion_dim[0]
           10. bitFusion Cols           ---> bitFusion_dim[1]
           11. bitBrick  Rows           ---> bitbrick_dim[0]
           12. bitBrick  Cols           ---> bitbrick_dim[1]
           13. Input buffer Size        ---> Input buffer
           14. Weight buffer Size       ---> Weight buffer
           15. Output buffer Size       ---> Output buffer
           16. Input Quantization       ---> Input quantization
           17. Weight Quantization      ---> Weight Quantization
           18. Bit Brick Consumed       ---> Indicates the % of bitbricks consumed
           19. Fusion Units Consumed    ---> Fusion units consumed! """
        ##
        self.input_image = input_image
        self.kernel_image = kernel_image
        ##
        self.entire_sim_data = {}
        self.entire_mem_data = {}
        ##
        self.window_num  = 0
        self.cycles_used = 0
        ##
        self.input_image_shape = list(self.input_image.shape)
        self.kernel_shape = list(self.kernel_image.shape)
        ##
        self.bitFusion_rows = fusionunit_dim[0]
        self.bitFusion_cols = fusionunit_dim[1]
        ##
        self.bitBrick_rows = bitbrick_dim[0]
        self.bitBrick_cols = bitbrick_dim[1]
        ##
        self.ibuf_size = ibuf_size
        self.wbuf_size = wbuf_size
        self.obuf_size = obuf_size
        ##
        self.inputQuantization = input_quantization
        self.weightQuantization = weight_quantization
        ##
        self.bitBricks_used_in_all_cycles = 0
        self.fusionUnits_used_in_all_cycles = 0
        ## We have a shared input buffer for the fusion units
        self.init_buffers_in_mem_db(self.bitFusion_rows, self.bitFusion_cols)

    def displayConfiguration(self):
        """ displayConfiguration - This is to be set in a different way """
        print("The systolic array architecture has the following features")
        print("The number of fusion units in the design are of the form {} x {}".format(self.bitFusion_rows, self.bitFusion_cols))
        print("The number of bit-bricks in each fusion unit are of the form {} x {}".format(self.bitBrick_rows, self.bitBrick_cols))
        print("The sizes of input buffer, weight buffer, and the output buffer are {}, {} and {}".format(self.ibuf_size, self.wbuf_size, self.obuf_size))
        print("The input quantization, weight quantization numbers are {} and {}".format(self.inputQuantization, self.weightQuantization))
        print("Bit-bricks used (%) is {}".format(self.bitBricks_used_in_all_cycles))
        print("Fusion units used (%) is {}".format(self.fusionUnits_used_in_all_cycles))

    def init_buffers_in_mem_db(self, fusionUnit_rows, fusionUnit_cols):
        """ This method initializes the input buffers in the fusion units """
        """ As per the architecture, the input buffers are shared, whereas the weight 
            buffers aren't shared amongst the different fusion units. """
        """ As we need to handle the movement of data into the buffer, we need to model the effects of the same """
        for c in range(fusionUnit_cols):
            ## Weight buffer name!
            wbuf_name = "WBUF_"+str(c)
            ibuf_name = "IBUF"
            self.entire_mem_data[ibuf_name] = {}
            self.entire_mem_data[ibuf_name]['all_lru'] = []

            input_image_pixel_size = self.input_image_shape[0] * self.input_image_shape[1]
            if len(self.input_image_shape) == 3:
                input_image_pixel_size *= self.input_image_shape[2]

            # Fill the entire the input buffer for N * 16 bytes ( N - fusionUnit_rows)
            self.entire_mem_data[wbuf_name] = {}
            self.entire_mem_data[wbuf_name]['all_lru'] = []
            
            # Based on the fusion Unit dimensions, we are filling the input buffers and the weight buffers!
            for halfword in range(int(self.ibuf_size / self.bitFusion_rows)):
                size = halfword*self.bitFusion_rows
                if input_image_pixel_size > halfword * self.bitFusion_rows:
                    self.entire_mem_data[ibuf_name]['mem'+str(size)] = {}
                    self.entire_mem_data[ibuf_name]['mem'+str(size)]['pix_byte'] = size
                    self.entire_mem_data[ibuf_name]['mem'+str(size)]['lru'] = 0
                    self.entire_mem_data[ibuf_name]['all_lru'].append(0)
                    self.entire_mem_data[ibuf_name]['pix'+str(size)] = size
                else:
                    self.entire_mem_data[ibuf_name]['mem'+str(size)] = {}
                    self.entire_mem_data[ibuf_name]['mem'+str(size)]['pix_byte'] = 0
                    self.entire_mem_data[ibuf_name]['mem'+str(size)]['lru'] = 0
                    self.entire_mem_data[ibuf_name]['all_lru'].append(0)
                    self.entire_mem_data[ibuf_name]['pix'+str(size)] = size

            self.entire_mem_data[wbuf_name] = {}
            self.entire_mem_data[wbuf_name]['all_lru'] = []
            input_kernel_pixel_size = self.kernel_shape[0] * self.kernel_shape[1]
            if len(self.kernel_shape) == 3:
                input_kernel_pixel_size *= self.kernel_shape[2]

            for halfword in range(int(self.wbuf_size/self.bitFusion_cols)):
                size = halfword*self.bitFusion_cols
                if input_image_pixel_size > halfword * self.bitFusion_rows:
                    self.entire_mem_data[wbuf_name]['mem'+str(size)] = {}
                    self.entire_mem_data[wbuf_name]['mem'+str(size)]['pix_byte'] = size
                    self.entire_mem_data[wbuf_name]['mem'+str(size)]['lru'] = 0
                    self.entire_mem_data[wbuf_name]['all_lru'].append(0)
                    self.entire_mem_data[wbuf_name]['pix'+str(size)] = size
                else:
                    self.entire_mem_data[wbuf_name]['mem'+str(size)] = {}
                    self.entire_mem_data[wbuf_name]['mem'+str(size)]['pix_byte'] = size
                    self.entire_mem_data[wbuf_name]['mem'+str(size)]['lru'] = 0
                    self.entire_mem_data[wbuf_name]['all_lru'].append(0)
                    self.entire_mem_data[wbuf_name]['pix'+str(size)] = size
        
        ##
        for col in range(fusionUnit_cols):
            obuf_name = "OBUF_x_"+str(col)
            self.entire_mem_data[obuf_name] = {}
            self.entire_mem_data[obuf_name]['next_byte'] = 0

    def init_cycle_in_db(self, cycle_num):
        ## 
        new_cycle_num = cycle_num
        new_cycle_name = 'cycle'+str(new_cycle_num)
        assert new_cycle_name not in self.entire_sim_data.keys(), 'compiler.py <---- init_cycle_in_db: Cycle already present'
        self.entire_sim_data[new_cycle_name] = {}

        for col in range(self.bitFusion_cols):
            col_name = "col"+str(col)
            self.entire_sim_data[new_cycle_name][col_name] = {}
            self.entire_sim_data[new_cycle_name][col_name]['nextFU'] = 'FU_0_' + str(col)
            self.entire_sim_data[new_cycle_name][col_name]['status'] = 'free'
            self.entire_sim_data[new_cycle_name][col_name]['command'] = 'nop'

            for row in range(self.bitFusion_rows):
                FU_name = "FU_"+str(row)+"_"+str(col)
                self.entire_sim_data[new_cycle_name][col_name][FU_name] = {}
                self.entire_sim_data[new_cycle_name][col_name][FU_name]['status'] = 'free'
                self.entire_sim_data[new_cycle_name][col_name][FU_name]['command'] = 'nop'

                for row_BB in range(self.bitBrick_rows):
                    for col_BB in range(self.bitBrick_cols):
                        BB_name = "BB_"+str(row_BB)+"_"+str(col_BB)
                        self.entire_sim_data[new_cycle_name][col_name][FU_name][BB_name] = {}
                        self.entire_sim_data[new_cycle_name][col_name][FU_name][BB_name]['status'] = 'free'
                        self.entire_sim_data[new_cycle_name][col_name][FU_name][BB_name]['command'] = 'nop'
    
    # How to generate interesting statistics from these values?
    def generate_interesting_stats(self, cycle):
        cycle_name = cycle
        #
        for FU_cols in range(self.bitFusion_cols):
            colName = "col"+str(FU_cols)
            #
            for FU_rows in range(self.bitFusion_rows):
                FUName = "FU_"+str(FU_rows)+"_"+str(FU_cols)
    
                # Increment the counter for cycles where FU is being used
                if self.entire_sim_data[cycle_name][col_name][FUName]['status'] != 'free':
                    self.fusionUnits_used_in_all_cycles += 1
                #
                for BB_row in range(self.bitBrick_rows):
                    for BB_col in range(self.bitBrick_cols):
                        BB_name = "BB_"+str(BB_row)+"_"+str(BB_col)
                        if self.entire_sim_data[cycle_name][col_name][FUName][BB_name]['status'] != 'free':
                            self.bitBricks_used_in_all_cycles += 1

    def clear_obuf_data(self, obuf_name):
        with open(obuf_name+".txt", "a+") as file:
            file.write(obuf_name+"\n")
            file.write(json.dumps(self.entire_mem_data[obuf_name], indent = 4))
            file.write("\n")

        obuf_keys = list(self.entire_mem_data[obuf_name].keys())
        for key in obuf_keys:
            if key == 'next_byte':
                continue
            else:
                self.entire_mem_data[obuf_name].pop(key, None)

    # Adds a new cycle to the DB into a file
    def add_new_cycle(self):
        if self.cycles_used > 0:
            cycle_remove = 'Cycle'+str(self.cycles_used)
            self.clear_cycle_from_sim_data(cycle_remove)

        self.cycles_used += 1
        self.init_cycle_in_db(self.cycles_used)


    def clear_cycle_from_sim_data(self, cycle_remove):
        self.generate_interesting_stats(cycle_remove)
        with open("entire_sim_data.txt", "a+") as file:
            file.write("\""+cycle_remove+"\": ")
            file.write(json.dumps(self.entire_sim_data[cycle_remove], indent=4))
            file.write("\n")
        self.entire_sim_data.pop(cycle_remove, None)

    # Gives a usable column for a Kernel to use -
    # Also generates address for accumulator of that column!
    def get_usable_bitfusion_col(self, window_num):
        current_cycle = self.cycles_used

        for colNum in range(self.bitFusion_cols):
            if self.entire_sim_data['cycle'+str(current_cycle)]['col'+str(colNum)]['status'] == 'free':
                self.entire_sim_data['cycle'+str(current_cycle)]['col'+str(colNum)]['status'] == 'used'

                self.gen_staddr_cmd(colNum, 'col'+str(colNum), cur_cycle)
                return cur_cycle, colNum

        # If reached here, it means next cycle is needed!
        self.add_new_cycle()
        cur_cycle = self.cycles_used
        for colNum in range(self.bitFusion_cols):
            if self.entire_sim_data['cycle'+str(cur_cycle)]['col'+str(colNum)]['status'] == 'free':
                self.entire_sim_data['cycle'+str(current_cycle)]['col'+str(colNum)]['status'] == 'used'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--input_image_shape', type=int, nargs='+',
                        help='input image shape, default: 1 28 28')
    parser.add_argument('--kernel_image_shape', type=int, nargs='+',
                        help='kernel image shape, default: 3 3 3')
    parser.add_argument('--padding', type=int, nargs='?', const=0,
                        help='Padding for each input, default: 0')
    parser.add_argument('--bitfusion_dim', type=int, nargs='+',
                        help='Bitfusion dimension, default: 16 16')
    parser.add_argument('--ibuf_size', type=int, nargs='?', const=256, help='Input buffer size (B) for all the fusion units (shared), default: 256')
    parser.add_argument('--wbuf_size', type=int, nargs='?', const=128,
                        help='Weight buffer size  (B) for each fusion unit column, default: 128')
    parser.add_argument('--obuf_size', type=int, nargs='?', const=1024,
                        help='Output buffer size (B) for each fusion unit column, default: 1024')
    parser.add_argument('--input_quant', type=int, nargs='?', const=8,
                        help='Inputs quantization 2/4/6/8, default:8')
    parser.add_argument('--weight_quant', type=int, nargs='?', const=8,
                        help='Outputs quantization 2/4/6/8, default:8')

    args = parser.parse_args()

    if args.input_image_shape == None:
        args.input_image_shape = [1, 28, 28]
    if args.kernel_image_shape == None:
        args.kernel_image_shape = [3, 3, 3]
    if args.bitfusion_dim == None:
        args.bitfusion_dim = [16, 16]
    if args.ibuf_size == None:
        args.ibuf_size = 256
    if args.wbuf_size == None:
        args.wbuf_size = 128
    if args.obuf_size == None:
        args.obuf_size = 16*1024
    if args.input_quant not in [2, 4, 6, 8]:
        parser.error("Input quantization can only be 2, 4, 6 or 8")
    if args.weight_quant not in [2, 4, 6, 8]:
        parser.error("Weight quantization can only be 2, 4, 6 or 8")

    # compiler = compiler(args.input_image_shape
    input_image  = np.zeros(tuple(args.input_image_shape), dtype=int) * 15
    kernel_image = np.ones(tuple(args.kernel_image_shape), dtype=int) * 15
    padding      = args.padding

    print("The input image shape is {}".format(args.input_image_shape))

    if padding == 1:
        # np.pad pads the input array with a default value of zero!
        input_image = np.pad(input_image, (1, 1), 'constant')
        if len(args.input_image_shape) == 3:
            shape = input_image.shape
            input_image = input_image[1:(shape[0]-1), :, :]
        elif len(args.input_image_shape) > 3:
            assert len(input_image_shape) <= 3, "Error! padding wouldn't be correct!"

    ##
    compiler = compiler(input_image, kernel_image, tuple(args.input_image_shape), tuple(args.kernel_image_shape),\
            args.input_quant, args.weight_quant, (4, 4), args.bitfusion_dim, args.ibuf_size, args.obuf_size, args.wbuf_size)
    ##
    print(compiler.displayConfiguration()) 


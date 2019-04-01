from memory import *
import os
import re
from fusionUnitWrapper import *

def entire_sim_data_parser(filename):
    # if os.path.exists('reconstruct_matrix_temp.txt'):
    #     os.remove('reconstruct_matrix_temp.txt')

    instr_file_pattern = re.compile('^instr_cycle(\d+).txt')

    for file in os.listdir("./cycle_instr_dir/."):
        print(file)
        if instr_file_pattern.match(file):
            print("removing file:{}".format(file))
            os.remove("./cycle_instr_dir/"+file)

    # temp_out_file = open('reconstruct_matrix_temp.txt', 'w')
    cycle_instr_file = ""

    cycle_max = 0
    with open(filename, 'r') as file:
        for line in file:
            line = line.rstrip()
            if "cycle" in line:
                pattern2 = re.compile('^"cycle(\d+)":')
                matches2 = pattern2.match(line)
                assert matches2, 'line is expected to match'
                cycle_max = int(matches2.group(1))

                if cycle_instr_file != "":
                    cycle_instr_file.close()

                cycle_instr_file = open("./cycle_instr_dir/"+'instr_cycle'+matches2.group(1)+".txt", 'w')
                # temp_out_file.write("#cycle"+matches2.group(1)+"\n")
                cycle_instr_file.write("#cycle"+matches2.group(1)+"\n")
            elif "command" in line:
                if "nop" not in line:
                    pattern = re.compile('^.*command": "(.*)"')
                    matches = pattern.match(line)

                    if matches:
                        # temp_out_file.write(matches.group(1)+"\n")
                        cycle_instr_file.write(matches.group(1)+"\n")

    # temp_out_file.close()
    if cycle_instr_file != "":
        cycle_instr_file.close()
    print("total cycles:{}".format(cycle_max))
    return cycle_max



if __name__ == "__main__":
    total_cycles = 0
    in_file_name = 'entire_sim_data.txt'
    if os.path.isfile(in_file_name):
        total_cycles = entire_sim_data_parser(in_file_name)

    print(total_cycles)
    obuf_data = {}
    bitfusion_cols = 16
    bitfusion_row = 16
    obuf_size = 256
    padding = 0
    input_shape = (1,8,8)
    kernel_shape = (1,4,4)
    # single image and kernel_shape = (2,2)
    calculated_output_shape = ((input_shape[0] + 2*padding - kernel_shape[0] + 1),\
                               (input_shape[1] + 2*padding - kernel_shape[1] + 1))
    if len(kernel_shape) == 3:
        calculated_output_shape = (kernel_shape[0],
                                   (input_shape[0] + 2*padding - kernel_shape[1] + 1), \
                                   (input_shape[1] + 2*padding - kernel_shape[2] + 1))

    # multiple images
    if len(input_shape) == 3:
        if len(kernel_shape) == 3:
            calculated_output_shape = (input_shape[0] * kernel_shape[0], \
                                    (input_shape[1] + 2*padding - kernel_shape[1] + 1),\
                                    (input_shape[2] + 2*padding - kernel_shape[2] + 1))
        elif len(kernel_shape) == 2:
            calculated_output_shape = (input_shape[0], \
                                       (input_shape[1] + 2*padding - kernel_shape[0] + 1), \
                                       (input_shape[2] + 2*padding - kernel_shape[1] + 1))


    total_windows = calculated_output_shape[0] * calculated_output_shape[1]
    if len(calculated_output_shape) == 3:
        total_windows *= calculated_output_shape[2]

    print("calculated_output_shape:{}".format(calculated_output_shape))
    print("total_windows:{}".format(total_windows))


    input_quantization = 6
    weight_quantization = 6

    input_image = [x for x in range(1,65)]
    print(np.array(input_image, dtype=int).reshape(input_shape))

    kernel = np.ones(kernel_shape, dtype=int) * 63
    print(kernel)

    print("After quantization")
    for x in range(len(input_image)):
        input_image[x] = input_image[x] >> (8 - input_quantization)

    kernel = list(kernel.flatten())
    for x in range(len(kernel)):
        kernel[x] = kernel[x] >> (8 - weight_quantization)

    print(np.array(input_image, dtype=int).reshape(input_shape))
    print(np.array(kernel, dtype=int).reshape(kernel_shape))

    # run the fusionUnitWrapper
    DD = fusionUnitWrapper(256, 128, 128)

    # input_image = list(np.ones((1, 5, 5), dtype=int).flatten() * ((255 & 0xff) >> (8 - input_quantization)))
    # kernel = list(np.ones((1, 2, 2), dtype=int).flatten() * ((127 & 0xff) >> (8 - weight_quantization)))

    for rows in range(DD.fuRows):
        for cols in range(DD.fuCols):
            DD.fuData[utils.getNameString('FU', rows, cols)]['fu_ibuf_obj'].store_mem(0x0, input_image)
            DD.fuData[utils.getNameString('FU', rows, cols)]['fu_wbuf_obj'].store_mem(0x0, kernel)

    instr_file_pattern = re.compile('^instr_cycle(\d+).txt')

    count = 1
    while os.path.exists("./cycle_instr_dir/" + 'instr_cycle' + str(count) + ".txt"):
        print("FUSIONUNITWRAPPER: EXECUTING INSTRUCTIONS FOR CYCLE:" + str(count))
        with open("./cycle_instr_dir/" + 'instr_cycle' + str(count) + ".txt") as f:
            count += 1
            command = f.read().splitlines()
            DD.addCommand(command)
            DD.sendCommand()
            DD.getBusyBitBricks()
            DD.execCommand()



    for buf_num in range(bitfusion_cols):
        obuf_data['OBUF_x_'+str(buf_num)] = {}
        obuf_data['OBUF_x_'+str(buf_num)]['mem_obj'] = memory('OBUF_x_'+str(buf_num), obuf_size, False)

    cycle = 0
    window_outputs = []
    # window_output = [[0 for x in range(input_shape[0] + 2*padding - kernel_shape[0] + 1)]
    #                  for y in range(input_shape[1] + 2*padding - kernel_shape[1] + 1)]

    for win in range(total_windows):
        buf_num_to_access = win % bitfusion_cols
        addr_to_load = 0x0 + (4*cycle)
        buf_name = 'OBUF_x_'+str(buf_num_to_access)

        # data will be a list
        data = obuf_data[buf_name]['mem_obj'].load_mem(addr_to_load, 4)
        accumulated_num = 0

        for x in range(4):
            accumulated_num += data[x] << (x*8)

        window_outputs.append(accumulated_num)
        if win == bitfusion_cols - 1:
            cycle += 1

    print(np.array(window_outputs).reshape(calculated_output_shape))


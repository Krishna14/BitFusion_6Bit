import os
import utils
import numpy as np


class mem_handlers():
    def __init__(self):
        utils.cprint("mem_handlers", "__init__", "inputs=self:" + str(self))
        self.mem_dir = 'memories/'

    def create_mem(self, name, size):
        if os.path.exists(self.mem_dir+name):
            utils.cprint("mem_handlers", "create_mem", "memory file "+name+" exists. Will replace!")
            os.remove(self.mem_dir+name)

        buf_fd = open(self.mem_dir+name, 'w+b')
        buf_fd.seek(size - 1)
        buf_fd.write(b'\0')
        buf_fd.close()

    # reads size bytes from addr space of memory name
    def read_mem(self, name, addr, size):
        assert os.path.exists(self.mem_dir+name), 'memory file '+str(name)+' missing'

        buf_fd = open(self.mem_dir+name, 'r+b')
        buf_fd.seek(addr)
        data = []
        access_byte_cnt = 1
        while access_byte_cnt < (size+1):
            temp = buf_fd.read(1)

            val_read = int.from_bytes(temp, byteorder='little')
            # if val_read < 0:
            #     val_read = utils.twos_complement(abs(val_read))

            #temp = bytearray(temp, 'utf-8')
            data.append(val_read)
            access_byte_cnt += 1
        data = np.array(data, dtype=np.int16)
        #utils.cprint("mem_handlers", "read_mem", str(data))
        buf_fd.close()
        return data.tolist()

    # stores data as list from starting address addr of memory name
    def write_mem(self, name, addr, data):
        assert os.path.exists(self.mem_dir+name), 'memory file ' + str(name) + ' missing'
        assert type(data) == list, 'write_mem expects data as list'

        buf_fd = open(self.mem_dir+name, 'rb+')
        buf_fd.seek(addr)
        buf_fd.write(bytes(data))
        buf_fd.close()


if __name__ == '__main__':
    mh = mem_handlers()
    mh.create_mem("rohit",0x8000)
    mh.write_mem("rohit", 0x400, [231, 11,12,13,14,15,16])
    #mh.read_mem("rohit", 8, 32)
    x = mh.read_mem("rohit", 0x400, 16)
    print(x)
    os.remove(mh.mem_dir+"rohit")

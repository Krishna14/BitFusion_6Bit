from mem_handlers import *


class memory(mem_handlers):

    def __init__(self, name, size=256, create_new=True):
        print("memory <- __init__: inputs=self:"+str(self))
        super(memory, self).__init__()
        self.mem_name = name
        self.mem_size = size
        if create_new == True:
            self.create_mem(self.mem_name, self.mem_size)

    def load_mem(self, addr, length):
        assert (addr + length) <= self.mem_size, 'load_mem access from ' + self.mem_name + 'exceeds its size'
        data = self.read_mem(self.mem_name, addr, length)
        # returns list type
        print("memory.py <- load_mem for {}: inputs=addr:{}, length:{}, data:{}".\
              format(self.mem_name,hex(addr),length,data))
        return data

    def store_mem(self, addr, data):
        print("memory.py <- store_mem for "+self.mem_name+" : inputs=addr:"+str(addr)+", data:"+str(data))
        assert (addr + len(data)) <= self.mem_size, 'store_mem access from ' + self.mem_name + 'exceeds its size'
        #assert type(data) == list, 'store_mem expects data as list'
        # TODO add check for integer overflows
        #for b in data:
        #    assert (b>>8) == 0, 'store_mem int8 integer overflow occurred'

        self.write_mem(self.mem_name, addr, data)





if __name__ == '__main__':
    mem_obj = memory("rohit", 12*1024) #12K
    data_to_store = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    mem_obj.store_mem(0x400, data_to_store)
    mem_obj.load_mem(0x0, 16)
    data = mem_obj.load_mem(0x400, 17)
    print(data)

    mem_obj2 = memory("gupta", 12*1024)
    mem_obj2.store_mem(0x0,data)
    print(mem_obj2.load_mem(0x0,8))

    mem_obj2.store_mem(0x0, utils.align_num_to_byte(1<<24))

    os.remove(mem_obj.mem_dir+"rohit")
    os.remove(mem_obj2.mem_dir+"gupta")


##This document contains the descriptions for the ISA that we have designed for this project.

ld-mem:
    Used for loading 4B from main memory to the buffers
    Operands width depends on the memory supported and include the target buffer as well

st-mem:
    Used for storing 4B from buffers to the main memory
    Operands width depends on the memory supported

ld-buf:
    Used for loading 1B from buffers - IBUF, WBUF
    Operands width depends on the memory supported

mul2:
    Used for multiplying 2bits in the Bit Brick
    Operands widths can vary depending on the size of IBUF, WBUF as they specify which address to get the bits from

staddr:
    Used for storing the data from column-based accumulator to an address in OBUF
    Operand can vary depending on size of OBUF

#!/usr/bin/python3

import re

pattern = r"Cookie"
sequence = "Cookie"

if re.match(pattern, sequence):
    print("Match!")
else:
    print("Not a Match!")

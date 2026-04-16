"""
Simple example showing inplace window summation

This example mutates the size of the base sequence by 
replacing a window of elements with their sum. This means 
each call to advance will grab the next N items and replace 
them with their sum. At the end you have a list of sums for 
each block of numbers.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from sliceview import sliceview

x = list(range(100))
y = sliceview(x, 0, 10)

while y:
    print(y, sum(y))
    y[:] = [sum(y)]
    y.advance(1)
print(x)
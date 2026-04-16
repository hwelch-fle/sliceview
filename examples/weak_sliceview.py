"""
Simple example of using a weak reference as the base of a sliceview

This example uses the code from sum_window, but deletes the base when 
the sum is 245 showing that the object the view is referencing no longer
exists. Can be useful for situations where you don't want a sliceview to
prevent data that isn't used from being cleaned up.

This script will raise a ReferenceError and you'll see that the proxy is marked
dead before the final iteration that raises the error
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from sliceview import sliceview
from weakref import proxy

class weaklist[T](list[T]): ...

x = weaklist(list(range(100)))
y = sliceview(proxy(x), 0, 10)

while y:
    print(sum(y))
    if sum(y) == 245:
        del x # type: ignore
    print(repr(y.base))
    y[:] = [sum(y)]
    y.advance(1)
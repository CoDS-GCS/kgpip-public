"""
finds the install path of DGL and comments out the pesky line that prints: "Using Backend: PyTorch"
somehow, this line gets printed over and over when auto-sklearn is running

"""

import dgl
import fileinput

script_path = dgl.__path__[0] + '/backend/__init__.py'

with fileinput.input(script_path, inplace=True) as f:
    for line in f:
        new_line = line.replace("print('Using backend: %s' % mod_name, file=sys.stderr)", 
                                "# print('Using backend: %s' % mod_name, file=sys.stderr)")
        print(new_line, end='')


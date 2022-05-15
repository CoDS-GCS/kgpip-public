# finds the install path of DGL and comments out the pesky line that prints: "Using Backend: PyTorch"
# somehow, this line gets printed over and over when auto-sklearn is running

import autosklearn
import fileinput

"""
This helper script goes to the root directory of auto sklearn and changes the default logging.yaml configuration 
to prevent it from printing to console.
"""

script_path = autosklearn.__path__[0] + '/util/logging.yaml'

with fileinput.input(script_path, inplace=True) as f:
    for line in f:
        new_line = line.replace("[console, file_handler]", "[]").replace("[file_handler]", "[]")\
                       .replace("[file_handler, console]", "[]").replace("[distributed_logfile]", "[]")
        new_line = new_line.replace('class: logging.StreamHandler', 'class: logging.FileHandler').replace('stream: ext://sys.stdout', 'filename: autosklearn.log')
        print(new_line, end='')


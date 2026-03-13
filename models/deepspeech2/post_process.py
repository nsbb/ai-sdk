#!/usr/bin/env python3
import numpy as np
import os
import re
import sys

def parse_args():
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('', '--tensor', dest="tensor_file", help="network name")
    parser.add_option('', '--dtype', dest="dtype", default='float32', help="dtype of tensor file: int8/int16/int32/float/float16/float32/float64")

    (options, args) = parser.parse_args()
    if options.tensor_file:
        return options

    parser.print_help()
    sys.exit(-1)

alphabets = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', "'"]

if __name__ == '__main__':
    options = parse_args()
    print()
    if os.path.exists(options.tensor_file):
        print(options.tensor_file)
        tensor = np.fromfile(options.tensor_file, dtype=getattr(np, options.dtype), sep='\n')
        tensor = tensor.reshape((-1, 29))
        print(tensor[0,:])
        tensor_argmax = np.argmax(tensor, axis=1)
        print(tensor_argmax)
        results = [alphabets[a] if a != len(alphabets) else '-' for a in tensor_argmax]
        resut_str = ''.join(results)
        print(resut_str.replace('-', ''))
        for i in range(1, len(alphabets)):
            resut_str = re.sub(pattern=r'{}+'.format(alphabets[i]), string=resut_str, repl='{}'.format(alphabets[i]))
        print(re.sub(pattern=r'(\s)+', string=resut_str.replace('-', ''), repl=r'\1'))

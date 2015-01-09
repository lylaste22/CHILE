#!/usr/bin/env python
# coding: utf-8

# import your requirements here (ex. import numpy)

import sys
import test_omp

def main():

    # receive the arguments
    if (len(sys.argv)<2):
         print "usage: %s arg1 arg2 arg3 ..." % sys.argv[0]
         sys.exit(1)
	
    # get the arguments into variables
    arg1 = sys.argv[1];
    arg2 = sys.argv[2];
    arg3 = sys.argv[3];

    # code here your stuff

    print "do some stuff with %s %s %s " % (arg1, arg2, arg3)

    test_omp.run();

# main function
if __name__ == "__main__":
	main()

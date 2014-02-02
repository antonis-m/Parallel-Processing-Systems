#!/usr/bin/python

import sys


f1 = open(sys.argv[1], 'r')
f2 = open(sys.argv[2], 'r')
f3 = open(sys.argv[3], 'r')

file_list = [f1,f2,f3]

sizes = [512, 1024, 2048]
threads = range(1,9)
for s in sizes:
    out = open(str(s)+".out",'w')
    out.write("Size: "+ str(s)+"\n")
    for t in threads:
        out.write(str(t)+"\t")
        val1 = float(f1.readline())
        val2 = float(f2.readline())
        val3 = float(f3.readline())
        average = round((val1+val2+val3) / 3, 4)
        out.write(str(val1)+"\t")
        out.write(str(val2)+"\t")
        out.write(str(val3)+"\t")
        out.write(str(average)+"\n")
    out.close()


f1.close()
f2.close()
f3.close()

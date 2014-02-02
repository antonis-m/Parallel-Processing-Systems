#!/usr/bin/python

f1 = open("512.out",'r')
f2 = open("1024.out",'r')
f3 = open("2048.out",'r')
out = open("output",'w')

f1.readline()
f2.readline()
f3.readline()

for t in range(1,9):
    val1 = ((f1.readline()).split("\t"))[4][:-1]
    if t==1:
        base1 = float(val1)
    val1 = base1 / float(val1)
    val2 = ((f2.readline()).split("\t"))[4][:-1]
    if t==1:
        base2 = float(val2)
    val2 = base2 / float(val2)
    val3 = ((f3.readline()).split("\t"))[4][:-1]
    if t==1:
        base3 = float(val3)
    val3 = base3 / float(val3)
    out.write("%s\t%s\t%s\t%s\n" % (t,val1, val2,val3))

out.close()
f1.close()
f2.close()
f3.close()

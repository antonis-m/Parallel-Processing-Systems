#!/usr/bin/python

import Gnuplot
from numpy import *

g = Gnuplot.Gnuplot()
g.title('OMP Results')
g.xlabel('#threads')
g.ylabel('SpeedUp')
g('set term png')
g('set out "output.png"')

databuff1= Gnuplot.File("results.txt", using='1:2',with_='linespoints', title="512 table size")
databuff2 = Gnuplot.File("results.txt", using='1:3',with_='linespoints', title="1024 table size")
databuff3 = Gnuplot.File("results.txt", using='1:4',with_='linespoints', title="2048 table size")
g.plot(databuff1, databuff2, databuff3)

#!/usr/bin/python

import Gnuplot
from numpy import *

g = Gnuplot.Gnuplot()
g.title('Execution times')
g.xlabel('Table size')
g.ylabel('Elapsed Time in seconds')
g('set term png')
g('set out "output.png"')

databuff1= Gnuplot.File("times", using='1:2',with_='linespoints', title="omp naive")
databuff2 = Gnuplot.File("times", using='1:3',with_='linespoints', title="omp tiled")
databuff3 = Gnuplot.File("times", using='1:4',with_='linespoints', title="naive")
databuff4 = Gnuplot.File("times", using='1:5',with_='linespoints', title="tiled")
databuff5 = Gnuplot.File("times", using='1:6',with_='linespoints', title="shmem_tiled")
g.plot(databuff1, databuff2, databuff3, databuff4, databuff5)

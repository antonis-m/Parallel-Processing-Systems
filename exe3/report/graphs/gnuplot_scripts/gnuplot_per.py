#!/usr/bin/python

import Gnuplot
from numpy import *

g = Gnuplot.Gnuplot()
g.title('Performance - size 8192')
g.ylabel('Medges/sec')
g('set term png')
g('set out "output.png"')
g('set style fill solid border -1')
g('set style data histogram')
g('set boxwidth 0.9')
g('set xtic scale 0')
g('set auto x')
g('unset key')
g('set yrange [0:8]')
databuff1= Gnuplot.File("8192", using='2:xtic(1) ti col fc rgb "#ff0000" ')
g.plot(databuff1)

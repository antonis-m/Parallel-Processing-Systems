CC=gcc -O3 -Wall -o 
OCC=gcc -O3 -fopenmp -Wall -o 

all: serial rec tiled

serial: lu_serial.c
	$(CC) serial lu_serial.c 
rec: lu_rec.c
	$(CC) rec lu_rec.c 
tiled: lu_tiled.c
	$(CC) tiled lu_tiled.c 

clean:
	rm serial rec tiled 


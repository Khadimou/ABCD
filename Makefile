#ICC = icc

CFLAGS=-std=c99 -Wall -pg -DNDEBUG
DFLAGS=-g3
LFLAGS=-lm
#IFLAGS = -mkl

NN_FLIST=nn.c 
all: nn filtre O2 #mkl

nn: main.o nn.o
	$(CC) $(CFLAGS) $(DFLAGS) -fopenmp -o $@ $^ $(LFLAGS) -g 

# mkl: main.o nn.o
# 	$(ICC) $(CFLAGS) -qopenmp main.c $(NN_FLIST) -o nn_bench_$@ $(IFLAGS) 

%.o: %.c
	$(CC) $(CFLAGS) $(DFLAGS) -fopenmp -c -o $@ $< $(LFLAGS) 

nn.o : protos.h
main.o : protos.h

filtre: filtre.o nn.o
	$(CC) $(CFLAGS) $(DFLAGS) -fopenmp -o $@ $^ $(LFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) $(DFLAGS) -c -fopenmp -o $@ $< $(LFLAGS)

filtre.o: protos.h

O2:
	$(CC) $(CFLAGS) $(DFLAGS) -$@ -fopenmp $(NN_FLIST) main.c -o nn_$@ $(LFLAGS)

clean:
	rm -f *.o nn filtre nn_O2 nn_bench_*

.PHONY: clean

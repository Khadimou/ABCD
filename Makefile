CFLAGS=-std=c99 -Wall
DFLAGS=-g3
LFLAGS=-lm

all: nn filtre

nn: main.o nn.o
	$(CC) $(CFLAGS) $(DFLAGS) -o $@ $^ $(LFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) $(DFLAGS) -c -o $@ $< $(LFLAGS)

nn.o : protos.h
main.o : protos.h

filtre: filtre.o nn.o
	$(CC) $(CFLAGS) $(DFLAGS) -o $@ $^ $(LFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) $(DFLAGS) -c -o $@ $< $(LFLAGS)

filtre.o: protos.h

clean:
	rm -f *.o nn filtre

.PHONY: clean

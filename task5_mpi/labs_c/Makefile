CC=mpicc
CCFLAGS=-dynamic
LIBS=-lpng


EXE1=heat
OBJS1=pngwriter.o heat_mpi.o main.o

all: $(EXE1)

pngwriter.o:	pngwriter.c pngwriter.h
heat_mpi.o:	heat_mpi.c heat_mpi.h
main.o:	main.c heat_mpi.h

.PHONY: $(EXE1)

$(EXE1): $(OBJS1) .ALWAYSEXEC
	$(CC) $(CCFLAGS) $(OBJS1) -o $@ $(LIBS)

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

clean:
	-/bin/rm -f $(EXE1) a.out *.o 

.ALWAYSEXEC:

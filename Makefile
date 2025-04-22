# Compilers
CC = gcc

# Flags
CFLAGS = -g -std=c11 -Wall -Wextra -O1 -march=native -funroll-all-loops -ffast-math -ftree-vectorize -fopenmp
TINY_LDFLAGS = -lm -mavx2
CG_LDFLAGS = -lm -lglfw -lGL -lGLEW -mavx2

TARGETS = headless head

# Files
C_SOURCES = wtime.c log_256.c xoroshiro128p.c photon.c 
C_OBJS = $(patsubst %.c, %.o, $(C_SOURCES))

headless: tiny_mc.o $(C_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(TINY_LDFLAGS)

head: cg_mc.o $(C_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(CG_LDFLAGS)

clean:
	rm -f $(TARGETS) *.o


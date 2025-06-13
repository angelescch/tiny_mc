# Compilers
CC = icx

CFLAGS =  -std=c11 -Wall -Wextra -DICX -O3 -march=native -funroll-loops -ffast-math -ftree-vectorize -funsafe-math-optimizations -fopenmp -mavx2
TINY_LDFLAGS = -lm -mavx2
CG_LDFLAGS = -lm -lglfw -lGL -lGLEW -mavx2

TARGETS = headless_23 head

# Files
C_SOURCES = wtime.c log_256.c xoroshiro128p.c photon.c 
C_OBJS = $(patsubst %.c, %.o, $(C_SOURCES))

headless_23: tiny_mc.o $(C_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(TINY_LDFLAGS)

head: cg_mc.o $(C_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(CG_LDFLAGS)

clean:
	rm -f $(TARGETS) *.o


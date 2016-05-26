current_dir = $(shell pwd)
include_dir = -I $(current_dir)/include
binary_dir = $(current_dir)/bin
source_dir = $(current_dir)/src

CC = gcc
CFLAGS = -g -Wall #-Werror
binary = kmeans

all: main.c $(source_dir)/*   
	$(CC) $(CFLAGS) -o $(binary_dir)/$(binary) $(include_dir) $(source_dir)/* main.c -lm
clean:
	$(RM) $(binary_dir)/$(binary)

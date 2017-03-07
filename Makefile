current_dir = $(shell pwd)
include_dir = $(current_dir)/include
binary_dir = $(current_dir)/bin
source_dir = $(current_dir)/src

MKDIR_P = mkdir -p
CC = gcc
CFLAGS = -g -Wall #-Werror
binary = kmeans

all: main.c $(source_dir)/cluster.c
	$(MKDIR_P) $(binary_dir)	
	$(CC) $(CFLAGS) -o $(binary_dir)/$(binary) -I $(include_dir) $(source_dir)/cluster.c main.c -lm
clean:
	$(RM) $(binary_dir)/$(binary)

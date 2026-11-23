NVCC = nvcc
NVCCFLAGS = -arch=sm_80 -O3 -std=c++17 -Xcompiler -Wall,-Wextra
INCLUDES = -Iinclude

TARGET = delaunay

SRCS = src/main.cu src/delaunay.cu src/io.cpp
OBJS = src/main.o src/delaunay.o src/io.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(OBJS) -o $(TARGET)

src/%.o: src/%.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

src/%.o: src/%.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(TARGET) src/*.o

.PHONY: all clean

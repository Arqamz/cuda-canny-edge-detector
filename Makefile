PROJECT_NAME_SERIAL = canny_serial
PROJECT_NAME_CUDA = canny_cuda

# Directory structure
BUILD_DIR = build
BIN_DIR = bin
SRC_DIR = src
INCLUDE_DIR = include
OBJ_DIR = $(BUILD_DIR)/obj

# Compiler settings
CC = gcc
NVCC = nvcc
CFLAGS_SERIAL = -Wall -O1 -g -I$(INCLUDE_DIR)/shared -I$(INCLUDE_DIR)/serial -lm
CFLAGS_CUDA = -Wall -O1 -g -I$(INCLUDE_DIR)/shared -I$(INCLUDE_DIR)/cuda -lm
NVCC_FLAGS = -O1 -g -I$(INCLUDE_DIR)/shared -I$(INCLUDE_DIR)/cuda
LDFLAGS = -lm
CUDA_LDFLAGS = -lm -lcudart

# Input image
PIC = input/pic_tiny.pgm

# Source files
SERIAL_SOURCES = $(wildcard $(SRC_DIR)/serial/*.c) $(wildcard $(SRC_DIR)/shared/*.c)
CUDA_C_SOURCES = $(wildcard $(SRC_DIR)/cuda/*.c) $(wildcard $(SRC_DIR)/shared/*.c)
CUDA_CU_SOURCES = $(wildcard $(SRC_DIR)/cuda/*.cu)

# Object files
SERIAL_OBJECTS = $(patsubst $(SRC_DIR)/%,$(OBJ_DIR)/%.o,$(basename $(SERIAL_SOURCES)))
CUDA_C_OBJECTS = $(patsubst $(SRC_DIR)/%,$(OBJ_DIR)/%.o,$(basename $(CUDA_C_SOURCES)))
CUDA_CU_OBJECTS = $(patsubst $(SRC_DIR)/%,$(OBJ_DIR)/%.o,$(basename $(CUDA_CU_SOURCES)))
CUDA_OBJECTS = $(CUDA_C_OBJECTS) $(CUDA_CU_OBJECTS)

# Default target
all: build_serial build_cuda

# Directory creation
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OBJ_DIR)/serial:
	mkdir -p $(OBJ_DIR)/serial

$(OBJ_DIR)/shared:
	mkdir -p $(OBJ_DIR)/shared

$(OBJ_DIR)/cuda:
	mkdir -p $(OBJ_DIR)/cuda

# Compilation rules for serial
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)/serial $(OBJ_DIR)/shared
	@mkdir -p $(@D)
	$(CC) $(CFLAGS_SERIAL) -c $< -o $@

# Compilation rules for CUDA .c files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)/cuda $(OBJ_DIR)/shared
	@mkdir -p $(@D)
	$(CC) $(CFLAGS_CUDA) -c $< -o $@

# Compilation rules for CUDA .cu files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)/cuda $(OBJ_DIR)/shared
	@mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Linking targets
build_serial: $(BIN_DIR)/$(PROJECT_NAME_SERIAL)

$(BIN_DIR)/$(PROJECT_NAME_SERIAL): $(SERIAL_OBJECTS) | $(BIN_DIR)
	$(CC) $^ -o $@ $(LDFLAGS)

build_cuda: $(BIN_DIR)/$(PROJECT_NAME_CUDA)

$(BIN_DIR)/$(PROJECT_NAME_CUDA): $(CUDA_OBJECTS) | $(BIN_DIR)
	$(NVCC) $^ -o $@ $(CUDA_LDFLAGS)

# Clean
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Run targets
run_serial: $(BIN_DIR)/$(PROJECT_NAME_SERIAL)
	$< $(PIC) 2.5 0.25 0.5

run_cuda: $(BIN_DIR)/$(PROJECT_NAME_CUDA)
	$< $(PIC) 2.5 0.25 0.5

# # Profiling with gprof
# gprof_serial: CFLAGS_SERIAL += -pg
# gprof_serial: LDFLAGS += -pg
# gprof_serial: clean build_serial
# 	echo "$(BIN_DIR)/$(PROJECT_NAME_SERIAL) $(PIC) 2.0 0.5 0.5" > lastrun.binary
# 	$(BIN_DIR)/$(PROJECT_NAME_SERIAL) $(PIC) 2.0 0.5 0.5
# 	gprof -b $(BIN_DIR)/$(PROJECT_NAME_SERIAL) > gprof_$(PROJECT_NAME_SERIAL).txt
# 	./run_gprof.sh $(PROJECT_NAME_SERIAL)

# CUDA profiling with nvprof (NVIDIA's profiler)
# Warning: nvprof is not supported on devices with compute capability 8.0 and higher.
# Use NVIDIA Nsight Systems for GPU tracing and CPU sampling and NVIDIA Nsight Compute for GPU profiling.
# Refer https://developer.nvidia.com/tools-overview for more details.
# nvprof: build_cuda
# 	nvprof --log-file nvprof_output.txt $(BIN_DIR)/$(PROJECT_NAME_CUDA) $(PIC) 2.5 0.25 0.5

# Phony targets
.PHONY: all clean build_serial build_cuda run_serial run_cuda serial cuda gprof_serial nvprof

# Shortcut targets
serial: build_serial
cuda: build_cuda

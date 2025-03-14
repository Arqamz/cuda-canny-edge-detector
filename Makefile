CC=gcc
LIBS= 
SOURCE_DIR= src
BUILD_DIR= build
BIN_DIR= bin
CFLAGS= -O1 -g
LDFLAGS= -lm 
OBJS=$(BUILD_DIR)/canny_edge.o $(BUILD_DIR)/hysteresis.o $(BUILD_DIR)/pgm_io.o
EXEC= canny
INCS= -I.
CSRCS= $(SOURCE_DIR)/canny_edge \
	$(SOURCE_DIR)/hysteresis.c \
	$(SOURCE_DIR)/pgm_io.c

# PIC=pics/pic_small.pgm
# PIC=pics/pic_medium.pgm
PIC=input/pic_large.pgm

all: canny

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

canny: $(BUILD_DIR) $(BIN_DIR) $(OBJS)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/$@ $(OBJS) $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SOURCE_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

run: $(EXEC) $(PIC)
	$(BIN_DIR)/$(EXEC) $(PIC) 2.5   0.25  0.5
# 			        sigma tlow  thigh

	
gprof:	CFLAGS +=  -pg
gprof:  LDFLAGS += -pg 
gprof:	clean all
	echo "$(BIN_DIR)/$(EXEC) $(PIC) 2.0 0.5 0.5" > lastrun.binary
	$(BIN_DIR)/$(EXEC) $(PIC) 2.0 0.5 0.5
	gprof -b $(BIN_DIR)/$(EXEC) > gprof_$(EXEC).txt
	./run_gprof.sh canny


clean:
	@-rm -rf $(BIN_DIR)/canny $(OBJS) gmon.out
	@-rm -rf $(BUILD_DIR)
	@-rm -rf $(BIN_DIR)

.PHONY: clean comp exe run all

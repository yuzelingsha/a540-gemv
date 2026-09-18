NDK ?= /home/reisa/mix2-gpu/android-ndk-r28b
API ?= 28
CXX := $(NDK)/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android$(API)-clang++
OPENCL_DIR ?= /home/reisa/mix2-gpu/opencl

CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -pthread -I$(OPENCL_DIR)/include
LDFLAGS := -L$(OPENCL_DIR) -ldl

TARGET := lmhead_microbench
SRC := main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean

# Makefile for libadrenollm_dlsym_hook.so
# Target: aarch64-linux-android (Qualcomm Snapdragon 835 / MSM8998)

NDK ?= $(HOME)/android-ndk-r25c
API ?= 28
CXX ?= $(NDK)/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android$(API)-clang++

CXXFLAGS := -std=c++17 -O3 -fPIC -Wall -Wextra -pthread
LDFLAGS  := -shared -ldl

TARGET := libadrenollm_dlsym_hook.so
SRC    := adrenollm_dlsym_hook.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	@if [ -x "$(CXX)" ]; then \
		echo "Building $(TARGET) with Android NDK Clang++..."; \
		$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS); \
	else \
		echo "Note: NDK compiler not found at $(CXX), fallback to local clang++ if available"; \
		clang++ -std=c++17 -O3 -fPIC $(SRC) -o $(TARGET) -shared -ldl 2>/dev/null || true; \
	fi

clean:
	rm -f $(TARGET)

.PHONY: all clean

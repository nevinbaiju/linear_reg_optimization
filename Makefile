CXX := g++
CXXFLAGS := -O3 -march=native -mavx

TARGETS := inference

.PHONY: all clean

all: $(TARGETS) inference.cpp

inference: inference.cpp
	$(CXX) $(CXXFLAGS) inference.cpp -o inference

run: 
	./inference

clean:
	rm -f $(TARGETS)
# clean_results:
# 	rm -rf results
# 	rm -rf plots
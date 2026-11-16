CXX      = g++
CXXFLAGS = -O2 -Wall -Wno-unused-result
LDLIBS   = -lpthread

Q1 = q1_p1 q1_p2 q1_p3 q1_p4
Q2 = q2_p1 q2_p2 q2_p3 q2_p4 q2_p5 q2_p6

all: $(Q1) $(Q2) gen_graph_ops

q1_p%: q1_mpsc.cpp
	$(CXX) $(CXXFLAGS) -DVERSION=$* -o $@ $< $(LDLIBS)

q2_p%: q2_graph.cpp
	$(CXX) $(CXXFLAGS) -DVERSION=$* -o $@ $< $(LDLIBS)

gen_graph_ops: gen_graph_ops.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(Q1) $(Q2) gen_graph_ops

.PHONY: all clean

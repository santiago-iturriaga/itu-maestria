CXXFLAGS =	-O2 -g -Wall -fmessage-length=0
LIBS =

MCTWRR: MCTWRR.cpp
	$(CXX) $(CXXFLAGS) -o MCTWRR MCTWRR.cpp $(LIBS)

MCTMakespan: MCTMakespan.cpp
	$(CXX) $(CXXFLAGS) -o MCTMakespan MCTMakespan.cpp $(LIBS)

SufferageWRR: SufferageWRR.cpp
	$(CXX) $(CXXFLAGS) -o SufferageWRR SufferageWRR.cpp $(LIBS)

SufferageMakespan: SufferageMakespan.cpp
	$(CXX) $(CXXFLAGS) -o SufferageMakespan SufferageMakespan.cpp $(LIBS)

MinMinMakespan: MinMinMakespan.cpp
	$(CXX) $(CXXFLAGS) -o MinMinMakespan MinMinMakespan.cpp $(LIBS)

MinMinWRR: MinMinWRR.cpp
	$(CXX) $(CXXFLAGS) -o MinMinWRR MinMinWRR.cpp $(LIBS)
	
all: MinMinMakespan MinMinWRR SufferageMakespan SufferageWRR MCTMakespan MCTWRR

clean:
	rm -f MinMinMakespan MinMinWRR SufferageMakespan SufferageWRR MCTMakespan MCTWRR

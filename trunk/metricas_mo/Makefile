CXXFLAGS =	-O2 -g -Wall -fmessage-length=0
LIBS =

all: FP_2obj FP_3obj FP_2obj_3col Spacing_2obj Spread_2obj

FP_2obj: FP_2obj.cpp
	$(CXX) $(CXXFLAGS) -o FP_2obj FP_2obj.cpp $(LIBS)

FP_2obj_3col: FP_2obj_3col.cpp
	$(CXX) $(CXXFLAGS) -o FP_2obj_3col FP_2obj_3col.cpp $(LIBS)

Spread_2obj: Spread_2obj.cpp
	$(CXX) $(CXXFLAGS) -o Spread_2obj Spread_2obj.cpp $(LIBS)
	
Spacing_2obj: Spacing_2obj.cpp
	$(CXX) $(CXXFLAGS) -o Spacing_2obj Spacing_2obj.cpp $(LIBS)

FP_3obj: FP_3obj.cpp
	$(CXX) $(CXXFLAGS) -o FP_3obj FP_3obj.cpp $(LIBS)

clean:
	rm -f FP_2obj FP_3obj Spread_2obj Spacing_2obj

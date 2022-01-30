# Makefile for test functions

CPP = g++
LC = g++
CPPFLAGS = `pkg-config --cflags liborthpol-1.0` -g -lgfortran
LCFLAGS = `pkg-config --libs liborthpol-1.0`

TARGETS = testRecur test5
TESTRECUR = testRecur.o
TEST5 = test5.o

testRecur : $(TESTRECUR);
	$(LC) $(TESTRECUR) -o testRecur $(LCFLAGS)

test5: $(TEST5);
	$(LC) $(TEST5) -o test5 $(LCFLAGS)

clean :
	rm -f *.o $(TARGETS)
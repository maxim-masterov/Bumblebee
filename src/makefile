# Make sure to use same compilers and flags as Trilinos
CXX=mpicxx

EXE_NAME=bumblebee.out

CXX_FLAGS=-std=c++11 -O3 -mavx2 -DBUMBLEBEE_USE_MAGIC_POWDER 
#-D_BUMBLEBEE_USE_OPENMP

BASE_DIR=/home/masterov/FoxBerry/SubModules

INCLUDE_DIRS=
#\
	-I$(SURFSARA_TRILINOS_INCLUDE)

LIBRARY_DIRS=
#\
	-L$(SURFSARA_TRILINOS_LIB) -L/usr/local/boost-1.64.0/stage/lib -lboost_serialization -lboost_system -lboost_filesystem -lboost_serialization -lfinitevolume -ltinydistributor -ltinydecomposer

LIBRARIES= -lstratimikos -lstratimikosbelos -lstratimikosaztecoo -lstratimikosamesos -lstratimikosml -lstratimikosifpack -lifpack2-adapters -lifpack2 -lanasazitpetra -lModeLaplace -lanasaziepetra -lanasazi -lamesos2 -lbelostpetra -lbelosepetra -lbelos -lml -lifpack -lzoltan2 -lpamgen_extras -lpamgen -lamesos -lgaleri-xpetra -lgaleri-epetra -laztecoo -lisorropia -lxpetra-sup -lxpetra -lthyratpetra -lthyraepetraext -lthyraepetra -lthyracore -lthyratpetra -lthyraepetraext -lthyraepetra -lthyracore -lepetraext -ltpetraext -ltpetrainout -ltpetra -lkokkostsqr -ltpetrakernels -ltpetraclassiclinalg -ltpetraclassicnodeapi -ltpetraclassic -ltpetraext -ltpetrainout -ltpetra -lkokkostsqr -ltpetrakernels -ltpetraclassiclinalg -ltpetraclassicnodeapi -ltpetraclassic -ltriutils -lzoltan -lepetra -lsacado -lrtop -lteuchoskokkoscomm -lteuchoskokkoscompat -lteuchosremainder -lteuchosnumerics -lteuchoscomm -lteuchosparameterlist -lteuchoscore -lteuchoskokkoscomm -lteuchoskokkoscompat -lteuchosremainder -lteuchosnumerics -lteuchoscomm -lteuchosparameterlist -lteuchoscore -lkokkosalgorithms -lkokkoscontainers -lkokkoscore -lkokkosalgorithms -lkokkoscontainers -lkokkoscore -lmpi

LINK_FLAGS=
#-nostartfiles

#just assuming that epetra is turned on.
DEFINES=-DMYAPP_EPETRA


default: print_info BumblebeeMPI.out

# Echo some info
print_info:
	@echo "\nBumblebee test with MPI "

BumblebeeMPI.out: BumblebeeMPI.o
	$(SCALASCA) $(CXX) $(CXX_FLAGS) $(INCLUDE_DIRS) BumblebeeMPI.cpp -o $(EXE_NAME) $(LINK_FLAGS) $(DEFINES) $(LIBRARY_DIRS) $(LIBRARIES)

BumblebeeMPI.o: BumblebeeMPI.cpp
	 $(SCALASCA) $(CXX) -c $(CXX_FLAGS) $(INCLUDE_DIRS) $(DEFINES) BumblebeeMPI.cpp

.PHONY: clean
clean:
	rm -f *.o *.a *.out 

//============================================================================
// Name        : BumblebeeMPI.cpp
// Author      : Maxim Masterov
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <mpi.h>
#include <stdio.h>
#include <vector>
#include <unordered_map>
#include <Epetra_MpiComm.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <sys/time.h>
#include <Epetra_MultiVector.h>
#include <Epetra_Time.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Export.h>
#include <Epetra_Import.h>
#include <AztecOO_config.h>
#include <AztecOO.h>
#include "ml_include.h"
#include "ml_MultiLevelPreconditioner.h"
#include "ml_epetra_utils.h"

#include "IterativeSolvers.h"

union _neighb {
    struct {
        double West;
        double South;
        double Bottom;
        double Cent;
        double Top;
        double North;
        double East;

        double do_not_touch_me;     // Dummy member to enforce alignment
    } name;

    double data[8];

    void setZero() {
        for(int n = 0; n < 8; ++n)
            data[n] = 0.;
    }
};

inline double getRealTime();
int StripeDecomposition(Epetra_Map *&Map, int _nx, int _ny, int _nz, int _size,
    const Epetra_MpiComm &comm);
int Decomposition3(Epetra_Map *&Map, int _size, const Epetra_MpiComm &comm);
int Decomposition4(Epetra_Map *&Map, int _size, const Epetra_MpiComm &comm);

/*
 * Mimics build of coefficients
 */
void BuildCoefficients(int i, int _ny, int _nz, int *MyGlobalElements, int NumGlobalElements, _neighb &weights) {

    double diag = 4.;
    double offd = -1.;

    if (_nz > 1)
        diag = 6.;

    int nynz = _ny * _nz;
    int l = 1;
    int coeff;

    if (_nz == 1)
        coeff = _ny;
    else
        coeff = _nz;

    weights.setZero();

    if (MyGlobalElements[i] >= nynz && MyGlobalElements[i] < NumGlobalElements) {
        weights.name.West = offd;
    }

    if (_nz > 1) {
        if (MyGlobalElements[i] >= _nz && MyGlobalElements[i] < NumGlobalElements) {
            l = i / nynz;
            if (i >= (nynz * l + _nz)) {
                weights.name.South = offd;
            }
        }
    }

    if ( MyGlobalElements[i] % coeff ) {
        weights.name.Bottom = offd;
    }

    weights.name.Cent = diag;

    if ( (MyGlobalElements[i] + 1) % coeff ) {
        weights.name.Top = offd;
    }

    if (_nz > 1) {
        if (MyGlobalElements[i] >= 0 && MyGlobalElements[i] < NumGlobalElements - _nz) {
            l = i / nynz;
            ++l;
            if (i < (nynz * l - _nz)) {
                weights.name.North = offd;
            }
        }
    }

    if (MyGlobalElements[i] >= 0 && MyGlobalElements[i] < NumGlobalElements - nynz) {
        weights.name.East = offd;
    }

//    std::cout << i << ": ";
//    for(int n = 0; n < 7; ++n)
//        std::cout << weights.data[n] << " ";
//    std::cout << "\n";
}

void AssembleMatrixGlob(int dimension, Epetra_Map *Map, int _nx, int _ny, int _nz, Epetra_CrsMatrix *A) {

    int NumMyElements = Map->NumMyElements();           // Number of local elements
    int *MyGlobalElements = Map->MyGlobalElements();    // Global index of local elements
    int NumGlobalElements = Map->NumGlobalElements();   // Number of global elements

    std::vector<double> Values(dimension + 1);
    std::vector<int> Indices(dimension + 1);
    _neighb weights;

    int nynz = _ny * _nz;

    for(int i = 0; i < NumMyElements; ++i) {

        int NumEntries = 0;

        BuildCoefficients(i, _ny, _nz, MyGlobalElements, NumGlobalElements, weights);

        if (fabs(weights.name.West) > DBL_EPSILON) {
            Indices[NumEntries] = MyGlobalElements[i] - nynz;
            Values[NumEntries] = weights.name.West;
            ++NumEntries;
        }

        if (fabs(weights.name.South) > DBL_EPSILON) {
            Indices[NumEntries] = MyGlobalElements[i] - _nz;
            Values[NumEntries] = weights.name.South;
            ++NumEntries;
        }

        if (fabs(weights.name.Bottom) > DBL_EPSILON) {
            Indices[NumEntries] = MyGlobalElements[i] - 1;
            Values[NumEntries] = weights.name.Bottom;
            ++NumEntries;
        }

        Indices[NumEntries] = MyGlobalElements[i];
        Values[NumEntries] = weights.name.Cent;
        ++NumEntries;

        if (fabs(weights.name.Top) > DBL_EPSILON) {
            Indices[NumEntries] = MyGlobalElements[i] + 1;
            Values[NumEntries] = weights.name.Top;
            ++NumEntries;
        }

        if (fabs(weights.name.North) > DBL_EPSILON) {
            Indices[NumEntries] = MyGlobalElements[i] + _nz;
            Values[NumEntries] = weights.name.North;
            ++NumEntries;
        }

        if (fabs(weights.name.East) > DBL_EPSILON) {
            Indices[NumEntries] = MyGlobalElements[i] + nynz;
            Values[NumEntries] = weights.name.East;
            ++NumEntries;
        }

        // Put in off-diagonal entries
        A->InsertGlobalValues(MyGlobalElements[i], NumEntries, Values.data(), Indices.data());
    }

    Values.clear();
    Indices.clear();

    A->FillComplete();
}

void UpdateMatrixGlob(int dimension, Epetra_Map *Map, int _nx, int _ny, int _nz, Epetra_CrsMatrix *A) {

    int NumMyElements = Map->NumMyElements();           // Number of local elements
    int *MyGlobalElements = Map->MyGlobalElements();    // Global index of local elements
    int NumGlobalElements = Map->NumGlobalElements();   // Number of global elements

    std::vector<double> Values(dimension + 1);
    std::vector<int> Indices(dimension + 1);
    _neighb weights;

    int nynz = _ny * _nz;


//    double *values = A->ExpertExtractValues();
//    Epetra_IntSerialDenseVector *indices = &A->ExpertExtractIndices();
//    Epetra_IntSerialDenseVector *offsets = &A->ExpertExtractIndexOffset();

    for(int i = 0; i < NumMyElements; ++i) {

        int NumEntries = 0;

        BuildCoefficients(i, _ny, _nz, MyGlobalElements, NumGlobalElements, weights);

        if (fabs(weights.name.West) > DBL_EPSILON) {
            Indices[NumEntries] = MyGlobalElements[i] - nynz;
            Values[NumEntries] = weights.name.West / 2.;
            ++NumEntries;
        }

        if (fabs(weights.name.South) > DBL_EPSILON) {
            Indices[NumEntries] = MyGlobalElements[i] - _nz;
            Values[NumEntries] = weights.name.South / 2.;
            ++NumEntries;
        }

        if (fabs(weights.name.Bottom) > DBL_EPSILON) {
            Indices[NumEntries] = MyGlobalElements[i] - 1;
            Values[NumEntries] = weights.name.Bottom / 2.;
            ++NumEntries;
        }

        Indices[NumEntries] = MyGlobalElements[i];
        Values[NumEntries] = weights.name.Cent / 2.;
        ++NumEntries;

        if (fabs(weights.name.Top) > DBL_EPSILON) {
            Indices[NumEntries] = MyGlobalElements[i] + 1;
            Values[NumEntries] = weights.name.Top / 2.;
            ++NumEntries;
        }

        if (fabs(weights.name.North) > DBL_EPSILON) {
            Indices[NumEntries] = MyGlobalElements[i] + _nz;
            Values[NumEntries] = weights.name.North / 2.;
            ++NumEntries;
        }

        if (fabs(weights.name.East) > DBL_EPSILON) {
            Indices[NumEntries] = MyGlobalElements[i] + nynz;
            Values[NumEntries] = weights.name.East / 2.;
            ++NumEntries;
        }

        // Put in off-diagonal entries
        A->ReplaceGlobalValues(MyGlobalElements[i], NumEntries, Values.data(), Indices.data());
//        memcpy(values+offsets->operator ()(i), Values.data(), sizeof(double)*NumEntries);
    }

    Values.clear();
    Indices.clear();
}

/* ********************************************************* */
/* ********************************************************* */
/* ********************************************************* */
/* ********************************************************* */
/* ********************************************************* */
/* ********************************************************* */
class MatrixDiagonal {
public:
    MatrixDiagonal() : imp(nullptr) { }
    ~MatrixDiagonal() { }

    /*
     * \brief Registers pointer on diagonal elements that should be sent in internal maps
     * Method fills internal maps and makes pointers to the diagonal elements and their row ids. Later
     * this data can be exported to the neighbor processes
     * \note Method has no effect if has already been called of for serial code (or with 1 process)
     */
    void RegisterDiagonal(Epetra_CrsMatrix *Matrix, const Epetra_MpiComm *comm);

    void Send(const MPI_Comm comm, std::vector<std::vector<double> > &rcv_data,
            std::vector<std::vector<int> > &rcv_ind);

    /*!
     * \brief Returns pointer to the vector of diagonal values that should be sent to proc \e pid
     * @param pid Receiving process
     */
    inline std::vector<double*> *GetDiagPointers(const int pid) {
        std::unordered_map<int, std::vector<double*> >::iterator it;
        it = snd_buf.find(pid);
        if (it == snd_buf.end()) {
            std::cout << "Error! Can't find PID " << pid
                    << " in the map of diagonal values." << std::endl;
            return nullptr;
        }
        return &it->second;
    }

    /*!
     * \brief Returns pointer to the vector of row ids that should be sent to proc \e pid
     * @param pid Receiving process
     */
    inline std::vector<int> *GetIndPointers(const int pid) {
        std::unordered_map<int, std::vector<int> >::iterator it;
        it = ids_buf.find(pid);
        if (it == ids_buf.end()) {
            std::cout << "Error! Can't find PID " << pid
                    << " in the map of diagonal values." << std::endl;
            return nullptr;
        }
        return &it->second;
    }

private:
    /*!
     * \brief Sorts indices and pointers form internal map based on the first one
     * Uses array of indices to sort in ascending order both indices and pointers of internal maps
     */
    void Sort(const int my_rank);

private:
    // why unordered? i don't know...
    std::unordered_map<int, std::vector<double*> > snd_buf;         // Pointers to the diagonal elements that should be sent
    std::unordered_map<int, std::vector<int> > ids_buf;             // Indices of diagonal elements that should be sent
    const Epetra_Import *imp;
};

void MatrixDiagonal::RegisterDiagonal(Epetra_CrsMatrix *Matrix, const Epetra_MpiComm *comm) {

    if (!snd_buf.empty() || !ids_buf.empty())
        return;

    const int my_rank = comm->MyPID();
    imp = Matrix->Importer();

    if (imp != nullptr) {
        double *values = Matrix->ExpertExtractValues();                              // Get array of values
        Epetra_IntSerialDenseVector *indices = &Matrix->ExpertExtractIndices();     // Get array of column indices
        Epetra_IntSerialDenseVector *offsets = &Matrix->ExpertExtractIndexOffset(); // Get array of offsets
        int number_exp_rows = imp->NumExportIDs();                                  // Get number of export rows
        int *exp_rows = imp->ExportLIDs();                                          // Get ids of export rows
        int *exp_procs = imp->ExportPIDs();                                         // Get ids of processes to which exp_rows should be exported

        /* Get diagonal values which should be exported */
        std::vector<double*> diag_values;
        diag_values.resize(number_exp_rows);
        for(int i = 0; i < number_exp_rows; ++i) {
            int loc_row = exp_rows[i];
            for(int j = offsets->operator ()(loc_row); j < offsets->operator ()(loc_row+1); ++j) {
                if (loc_row == indices->operator ()(j)) {
                    diag_values[i] = &values[j];
                }
            }
        }

        /*
         * Use two maps: one to store initially the size of individual messages that should be sent,
         * another one to store messages themselves. Keys are pids which should receive messages.
         */
        std::unordered_map<int, int> msg_size;                          // Local map helps to count how many data points should be sent to each process
        std::unordered_map<int, int>::iterator it_c, it_c_end;
        for(int i = 0; i < number_exp_rows; i++)
            msg_size[exp_procs[i]]++;

        /* Allocate memory for the sending map */
        it_c = msg_size.begin();
        it_c_end = msg_size.end();
        for(; it_c != it_c_end; ++ it_c) {
            snd_buf[it_c->first].resize(it_c->second);
            ids_buf[it_c->first].resize(it_c->second);
        }

        /* Assign data to the sending map */
        int offset = 0;
        std::unordered_map<int, std::vector<double*> >::iterator it_d, it_d_end;

        /* Original data consists of chunks which we need to extract and send separately to each neighbor */
        it_c = msg_size.begin();
        it_c_end = msg_size.end();
        it_d = snd_buf.begin();
        it_d_end = snd_buf.end();
        for(int n = 0, end = diag_values.size(); n < end; /* changed inside */) {
            int pid_loc = exp_procs[n];                                           // check out the pid
            it_d = snd_buf.find(pid_loc);
            if (it_d != it_d_end) {                                                             // if pid found we are safe to go
                uint32_t chunk_size = it_d->second.size();                                           // get number of elements in a chunk
                memcpy(ids_buf[pid_loc].data(), exp_rows + offset, sizeof(int) * chunk_size);           // copy particular chunk from the original array
                for(uint32_t m = 0; m < chunk_size; ++m) {
                    it_d->second[m] = diag_values[offset + m];
                }
                n += chunk_size;
                offset += chunk_size;
            }
            else {
                std::cerr << "Error! Can't find PID " << pid_loc
                        << " in the map of diagonal values. Rank " << my_rank << "." << std::endl;
            }
        }

        Sort(my_rank);
        // after this point we have two sorted vectors: vector of pointers on diagonal elements and
        // vector of corresponding row indices
        // TODO: wrap it in a single class (store map of pids and elements, map of pids and indices)
    }
}

void MatrixDiagonal::Sort(const int my_rank) {

    std::unordered_map<int, std::vector<double*> >::iterator it_d, it_d_end;
    std::unordered_map<int, std::vector<int> >::iterator it_i, it_i_end;

    it_d = snd_buf.begin();
    it_d_end = snd_buf.end();
    it_i_end = ids_buf.end();
    for(; it_d != it_d_end; ++it_d) {
        it_i = ids_buf.find(it_d->first);
        if (it_i == it_i_end) {
            std::cerr << "Error! Can't find PID " << it_d->first
                    << " in the map of row ids. Rank " << my_rank << "." << std::endl;
            break;
        }
        int size = it_i->second.size();
        // do stupid bubble sort of data and index vectors...
        for(int n = 0; n < size; ++n) {
            int max_ind = n;
            int my_value = it_i->second[max_ind];
            for(int m = n + 1; m < size; ++m) {
                if (it_i->second[m] < my_value) {
                    std::swap(it_i->second[m], it_i->second[max_ind]);
                    double *tmp = it_d->second[m];
                    it_d->second[m] = it_d->second[max_ind];
                    it_d->second[max_ind] = tmp;
                    max_ind = m;
                    my_value = it_i->second[max_ind];
                }
            }
        }
    }
}

void MatrixDiagonal::Send(const MPI_Comm comm, std::vector<std::vector<double> > &rcv_data,
        std::vector<std::vector<int> > &rcv_ind) {

    if (snd_buf.empty() || ids_buf.empty()) {
        std::cerr << "Error! Nothing to be sent." << std::endl;
        return;
    }

    int my_rank;
    int num_ngb = snd_buf.size();                       // number of neighbors
    MPI_Request send_request[num_ngb];
    MPI_Request recv_request[num_ngb];
    std::unordered_map<int, std::vector<double*> >::iterator it_d, it_d_end;
    std::unordered_map<int, std::vector<int> >::iterator it_i, it_i_end;

    MPI_Comm_rank(comm, &my_rank);
    rcv_data.resize(num_ngb);
    rcv_ind.resize(num_ngb);
    it_d = snd_buf.begin();
    it_d_end = snd_buf.end();
    it_i = ids_buf.begin();
    it_i_end = ids_buf.end();
    int counter = 0;
    int tag1 = 0;
    int tag2 = 0;
    for(; it_d != it_d_end; ++it_d) {
        int pid = it_d->first;
        it_i = ids_buf.find(pid);
        if (it_i == it_i_end)
            std::cerr << "Error! Can't find PID " << it_d->first
                    << " in the map of row ids. Rank " << my_rank << "." << std::endl;

        /* Assemble the message */
        int msg_size = it_d->second.size();
        std::vector<double> snd_msg(msg_size);
        std::cout << "(" << my_rank << "->" << pid << ")" << " snd_msg: " << num_ngb << " : " << " ";
        for(int n = 0, end = msg_size; n < end; ++n) {
            snd_msg[n] = *it_d->second[n];
            std::cout << snd_msg[n] << " ";
        }
        std::cout << "\n";
        MPI_Isend(snd_msg.data(), msg_size, MPI_DOUBLE, pid, tag1, comm, &send_request[counter]);
        MPI_Isend(it_i->second.data(), msg_size, MPI_INT, pid, tag2, comm, &send_request[counter]);

        /* Probe the message size and allocate memory */
        MPI_Status status;
        int rcv_size = 0;
        MPI_Probe(pid, tag1, comm, &status);
        MPI_Get_count(&status, MPI_DOUBLE, &rcv_size);
        rcv_data[counter].resize(rcv_size);
        rcv_ind[counter].resize(rcv_size);
        MPI_Irecv(rcv_data[counter].data(), rcv_size, MPI_DOUBLE, pid, tag1, comm, &recv_request[counter]);
        MPI_Irecv(rcv_ind[counter].data(), rcv_size, MPI_INT, pid, tag2, comm, &recv_request[counter]);
        ++counter;
    }

    MPI_Status status_arr[num_ngb];
    MPI_Waitall(num_ngb, send_request, status_arr);
    MPI_Waitall(num_ngb, recv_request, status_arr);
}
/* ********************************************************* */
/* ********************************************************* */
/* ********************************************************* */
/* ********************************************************* */
/* ********************************************************* */
/* ********************************************************* */

class MatrixDiagonal2 {
public:
    MatrixDiagonal2() { }
    ~MatrixDiagonal2() { }

    void RegisterDiagonal(Epetra_CrsMatrix *Matrix, const Epetra_MpiComm *comm, std::vector<int> &list_gid);

    void SendDiagonalCoefficients(const Epetra_MpiComm *comm,
            std::unordered_map<int, std::vector<double> > &rcv_data);

private:
    void Ping(const Epetra_MpiComm *comm, bool print = false);

    void MakePointersOnDiagonal(Epetra_CrsMatrix *Matrix);

private:
//    int CountMyNeighbors(const MPI_Comm comm);

private:
    // why unordered? I don't know...
//    std::unordered_map<int, std::vector<int> > remote_pids;             // Pointers to the diagonal elements that should be sent
    std::unordered_map<int, std::vector<int> > remote_lids;             // Indices of diagonal elements that should be received (from other proc)
    std::unordered_map<int, std::vector<int> > local_lids;              // Indices of diagonal elements that should be sent (from this proc)
    std::unordered_map<int, std::vector<double*> > local_diag;              // Indices of diagonal elements that should be sent (from this proc)
};

int CountMyNeighbors(const MPI_Comm comm) {

    int num_proc = 0;
    int my_rank = 0;
    int ngb = 0;
    int outd = 0;
    int wght = 0;
    int topology;

    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &num_proc);
    MPI_Topo_test(comm, &topology) ;

    switch (topology) {
        case MPI_UNDEFINED:
            ngb = num_proc;
            break;
        case MPI_CART:
            MPI_Cartdim_get(comm, &ngb);
            if (ngb == 2)
                ngb = 4;            // max 4 neighbors in 2d topology
            else if (ngb == 3)
                ngb = 6;            // max 6 neighbors in 3d topology
            else
                ngb = num_proc;     // for safety
            break;
        case MPI_GRAPH:
            MPI_Graph_neighbors_count(comm, my_rank, &ngb);
            break;
        case MPI_DIST_GRAPH:
            MPI_Dist_graph_neighbors_count(comm, &ngb, &outd, &wght);
            break;
    }

    return ngb;
}

void PrintList(int my_rank, std::vector<int> &list, const std::string name) {
    std::cout << "(" << my_rank << ") " << name << " : ";
    for(size_t n = 0; n < list.size(); ++n) {
        std::cout << list[n] << " ";
    }
    std::cout << "\n";
}
void MatrixDiagonal2::RegisterDiagonal(Epetra_CrsMatrix *Matrix, const Epetra_MpiComm *comm, std::vector<int> &list_gid) {

    if (!remote_lids.empty())
        return;

    int num_ids = list_gid.size();
    std::vector<int> list_pid(num_ids);
    std::vector<int> list_lid(num_ids);
    const int my_rank = comm->MyPID();
    const bool sendig_proc = list_gid.empty() ? false : true;

    /* Get PIDs and LIDs corresponding to GIDs */
    Matrix->Map().RemoteIDList(num_ids, list_gid.data(), list_pid.data(), list_lid.data());

    PrintList(my_rank, list_gid, "list_gid");
    PrintList(my_rank, list_pid, "list_pid");
    PrintList(my_rank, list_lid, "list_lid");
    return;

    /* Split obtained LIDs on groups based on unique PIDs */
    /* The map stores PID as a key and the number of corresponding LIDs as a value */
    std::unordered_map<int, int> pid_counter;
    std::unordered_map<int, int>::iterator it_c, it_c_end;
    for(int n = 0; n < num_ids; n++)
        pid_counter[list_pid[n]]++;

    /* Allocate memory for maps */
    it_c = pid_counter.begin();
    it_c_end = pid_counter.end();
    for(; it_c != it_c_end; ++ it_c) {
        remote_lids[it_c->first].resize(it_c->second);
    }

    /* Assign data to the sending map */
    int offset = 0;
    std::unordered_map<int, std::vector<int> >::iterator it_i, it_i_end;

    /* Obtained at step (1) data consists of chunks which should be split */
    it_i = remote_lids.begin();
    it_i_end = remote_lids.end();
    size_t data_type_size = sizeof(int);
    for(int n = 0; n < num_ids; /* changed inside */) {
        int pid_loc = list_pid[n];                                           // check out the pid
        it_i = remote_lids.find(pid_loc);
        if (it_i != it_i_end) {                                                             // if pid found we are safe to go
            uint32_t chunk_size = it_i->second.size();                                           // get number of elements in a chunk
            memcpy(remote_lids[pid_loc].data(), list_lid.data() + offset, data_type_size * chunk_size);           // copy particular chunk from the original array
            n += chunk_size;
            offset += chunk_size;
        }
        else {
            std::cerr << "Error! Can't find PID " << pid_loc
                    << " in the map of diagonal values. Rank " << my_rank << "." << std::endl;
        }
    }

    /* Ping remote processes and trigger */
    Ping(comm);
}

void MatrixDiagonal2::Ping(const Epetra_MpiComm *comm, bool print) {

    const int my_rank = comm->MyPID();
    bool sendig_proc = false;

    if (!remote_lids.empty())
        sendig_proc = true;

    /* Count process' neighbors */
    MPI_Comm loc_comm = comm->Comm();
    int my_ngb = CountMyNeighbors(loc_comm);

    /* 3) Send an array of LIDs to other process and inform it that we are waiting for some data from it */
    /*   (only for processes who has to collect remote data) */
    if (sendig_proc) {
        std::unordered_map<int, std::vector<int> >::iterator it_i, it_i_end;
        it_i = remote_lids.begin();
        it_i_end = remote_lids.end();
        for(; it_i != it_i_end; ++it_i)
            MPI_Send(it_i->second.data(), it_i->second.size(), MPI_INT, it_i->first, 0, loc_comm);           // <--- change PIDList[0]!!!
    }
    MPI_Barrier(loc_comm);            // !!! Absolutely necessary !!!

    /* 4) Check if process should receive any message */
    for(int n = 0; n < my_ngb; ++n) {
        int flag;
        MPI_Status status;
        MPI_Iprobe(MPI_ANY_SOURCE, 0, loc_comm, &flag, &status);
        if (flag == true) {
            /* If so - receive the message and put it into the map */
            int sender_id = status.MPI_SOURCE;
            if (print)
                std::cout << "PID " << my_rank << " has received a message from PID " << sender_id << "\n";

            int msg_size = 0;
            MPI_Get_count(&status, MPI_INT, &msg_size);
            local_lids[sender_id].resize(msg_size);
            MPI_Recv(local_lids[sender_id].data(), msg_size, MPI_INT, sender_id, 0, loc_comm, &status);

            std::cout << "from " << sender_id << " to " << my_rank << " : ";
            for(int n = 0; n < msg_size; ++n)
                std::cout << local_lids[sender_id][n] << " ";
            std::cout << std::endl;

            /* After this step we know what data should be sent from this process */
        }
        else {
            if (print)
                std::cout << "PID " << my_rank << " has not received any message\n";
        }
    }
}

void MatrixDiagonal2::MakePointersOnDiagonal(Epetra_CrsMatrix *Matrix) {

    /* Allocate memory */
    std::unordered_map<int, std::vector<int> >::iterator it_i, it_i_end;
    it_i = local_lids.begin();
    it_i_end = local_lids.end();
    for(; it_i != it_i_end; ++ it_i) {
        local_diag[it_i->first].resize(it_i->second.size());
    }

    /* Get diagonal values which should be exported */
    double *values = Matrix->ExpertExtractValues();                              // Get array of values
    Epetra_IntSerialDenseVector *indices = &Matrix->ExpertExtractIndices();     // Get array of column indices
    Epetra_IntSerialDenseVector *offsets = &Matrix->ExpertExtractIndexOffset(); // Get array of offsets
    std::unordered_map<int, std::vector<double*> >::iterator it_d, it_d_end;
    it_i = local_lids.begin();
    it_i_end = local_lids.end();
    for(; it_i != it_i_end; ++ it_i) {
        int loc_pid = it_i->first;
        int loc_size = it_i->second.size();
        it_d = local_diag.find(loc_pid);
        if (it_d == it_d_end) {

        }
        std::vector<double*> *loc_vec = &it_d->second;
        for(int lid = 0; lid < loc_size; ++lid) {
            int loc_row = it_i->second[lid];
            for(int j = offsets->operator ()(loc_row); j < offsets->operator ()(loc_row+1); ++j) {
                if (loc_row == indices->operator ()(j)) {
                    loc_vec->operator[](lid) = &values[j];
                }
            }
        }
    }
}

void MatrixDiagonal2::SendDiagonalCoefficients(const Epetra_MpiComm *comm,
        std::unordered_map<int, std::vector<double> > &rcv_data) {

    std::unordered_map<int, std::vector<double*> >::iterator it_d, it_d_end;

    it_d = local_diag.begin();
    it_d_end = local_diag.end();

    int neighbors = local_diag.size();
    MPI_Request snd_request[neighbors];
    MPI_Request rcv_request[neighbors];
    MPI_Status snd_status[neighbors];
    MPI_Status rcv_status[neighbors];
    int tag = 0;
    int counter = 0;
    for(; it_d != it_d_end; ++it_d) {
        int pid = it_d->first;
        uint32_t size = it_d->second.size();
        std::vector<double> data(size);
        for(uint32_t n = 0; n < size; ++n) {
            data[n] = *it_d->second[n];
        }
        MPI_Isend(data.data(), size, MPI_DOUBLE, pid, tag, comm->Comm(), &snd_request[counter]);

        rcv_data[pid].resize(remote_lids[pid].size());
        MPI_Irecv(rcv_data[pid].data(), size, MPI_DOUBLE, pid, tag, comm->Comm(), &rcv_request[counter]);
        ++counter;
    }

    MPI_Waitall(neighbors, snd_request, snd_status);
    MPI_Waitall(neighbors, snd_request, rcv_status);
}



class MatrixDiagonal3 {
public:
    MatrixDiagonal3() { }
    ~MatrixDiagonal3() { }

    void RegisterDiagonal(Epetra_CrsMatrix *Matrix, const Epetra_MpiComm *comm,
            std::vector<int> &list_gid);

    void ExchangeDiagonalCoefficients(const Epetra_MpiComm *comm,
            std::unordered_map<int, std::vector<double> > &rcv_data);
private:
    void MakePointersOnDiagonal(Epetra_CrsMatrix *Matrix, const Epetra_MpiComm *comm);

private:
    /*
     * The idea: use local_ids to assemble a vector of double from local_ptrs and send the vector to the neighbor.
     * Store received message in local_data
     */
    std::unordered_map<int, std::vector<int> > local_ids;           // local IDs which should be sent to the neighbor(s)
    std::unordered_map<int, std::vector<double> > local_data;       // local data that is received from the nighbor(s)
    std::unordered_map<int, std::vector<double*> > local_ptrs;      // local pointers on diagonal elements of the matrix
};

void MatrixDiagonal3::RegisterDiagonal(Epetra_CrsMatrix *Matrix, const Epetra_MpiComm *comm, std::vector<int> &list_gid) {

    const int my_rank = comm->MyPID();
    const Epetra_Import *importer = Matrix->Importer();
    bool print = false;

    if (importer != nullptr) {
        int number_exp_rows = importer->NumExportIDs();
        int *exp_procs = importer->ExportPIDs();
        MPI_Comm loc_comm = comm->Comm();
//        MPI_Request request_arr[number_exp_rows];
        int my_ngb = CountMyNeighbors(loc_comm);

        /* Send locally stored GIDs to export PIDs */
        int tag = 0;
        for(int n = 0; n < number_exp_rows; ++n) {
            if (!list_gid.empty())
                MPI_Send(list_gid.data(), list_gid.size(), MPI_INT, exp_procs[n], tag, loc_comm);
        }
        MPI_Barrier(loc_comm);

        /* Check if process should receive any message */
        for(int n = 0; n < my_ngb; ++n) {
            int flag;
            MPI_Status status;
            MPI_Iprobe(MPI_ANY_SOURCE, tag, loc_comm, &flag, &status);
            if (flag == true) {
                /* If so - receive the message and put it into the map */
                int sender_id = status.MPI_SOURCE;

                if (print)
                    std::cout << "1. PID " << my_rank << " has received a message from PID " << sender_id << "\n";

                int msg_size = 0;
                MPI_Get_count(&status, MPI_INT, &msg_size);
                std::unordered_map<int, std::vector<int> > local_list_gid;
                local_list_gid[sender_id].resize(msg_size);
                MPI_Recv(local_list_gid[sender_id].data(), msg_size, MPI_INT, sender_id, tag, loc_comm, &status);

//                std::cout << "from " << sender_id << " to " << my_rank << " : ";
//                for(int n = 0; n < msg_size; ++n)
//                    std::cout << local_list_gid[sender_id][n] << " ";
//                std::cout << std::endl;

                /* Iterate over received messages and check if GIDs from them are presented in this proc */
                std::unordered_map<int, std::vector<int> >::iterator it_i, it_i_end;
                it_i = local_list_gid.begin();
                it_i_end = local_list_gid.end();

                const Epetra_BlockMap *Map = &Matrix->Map();
                std::unordered_map<int, int> counter;
                for(; it_i != it_i_end; ++it_i) {
                    int loc_pid = it_i->first;
                    local_ids[loc_pid].resize(it_i->second.size());
                    counter[loc_pid] = 0;
                    int *prt_counter = &counter[loc_pid];
                    for(uint32_t n = 0, end = it_i->second.size(); n < end; ++n) {
//                        int loc_gid = it_i->second[n];
                        int lid = Map->LID(it_i->second[n]);
                        if (lid != -1) {
                            local_ids[loc_pid][*prt_counter] = lid;
                            (*prt_counter)++;
                        }
                    }
                }

                // Resize pre-allocated memory
                it_i = local_list_gid.begin();
                it_i_end = local_list_gid.end();
                for(; it_i != it_i_end; ++it_i) {
                    int loc_pid = it_i->first;
                    int size = counter[loc_pid];
                    if (size == 0)
                        local_ids.erase(loc_pid);
                    else
                        local_ids[loc_pid].resize(size);
                }

                /* After this step we know what data should be sent from this process */
            }
            else {
                if (print)
                    std::cout << "1. PID " << my_rank << " has not received any message\n";
            }
        }

//        MPI_Status status_arr[number_exp_rows];
//        MPI_Waitall(number_exp_rows, request_arr, status_arr);

//        if (local_ids.empty())
//            return;

//        std::unordered_map<int, std::vector<int> >::iterator it_i, it_i_end;
//        it_i = local_ids.begin();
//        it_i_end = local_ids.end();
//        std::cout << "(" << my_rank << ") ";
//        for(; it_i != it_i_end; ++it_i) {
//            std::cout << it_i->first << " : ";
//            for(uint32_t n = 0; n < it_i->second.size(); ++n)
//                std::cout << it_i->second[n] << " ";
//            std::cout << "\n    ";
//        }
//        std::cout << std::endl;
//
//        MPI_Barrier(loc_comm);
////
////        if (local_ids.empty())
////            return;
//
        /* Send back message indicating how many data points process should expect back */
        std::unordered_map<int, std::vector<int> >::iterator it_i, it_i_end;
        it_i = local_ids.begin();
        it_i_end = local_ids.end();
        tag = 1;
        for(; it_i != it_i_end; ++it_i) {
            int num_loc_points = it_i->second.size();
            MPI_Send(&num_loc_points, 1, MPI_INT, it_i->first, tag, loc_comm);           // <--- change PIDList[0]!!!
        }
        MPI_Barrier(loc_comm);

        for(int n = 0; n < my_ngb; ++n) {
            int flag;
            MPI_Status status;
            MPI_Iprobe(MPI_ANY_SOURCE, tag, loc_comm, &flag, &status);

            if (flag == true) {
                /* If so - receive the message and put it into the map */
                int sender_id = status.MPI_SOURCE;

                if (print)
                    std::cout << "2. PID " << my_rank << " has received a message from PID " << sender_id << "\n";

                int num_remote_points = 0;
                MPI_Recv(&num_remote_points, 1, MPI_INT, sender_id, tag, loc_comm, &status);

                local_data[sender_id].resize(num_remote_points);
            }
            else {
                if (print)
                    std::cout << "2. PID " << my_rank << " has not received any message\n";
            }
        }

        MPI_Barrier(loc_comm);

        /* Once finished - fill local data with zeros. It will be rewritten during the real communication */
        std::unordered_map<int, std::vector<double> >::iterator it_d, it_d_end;
        it_d = local_data.begin();
        it_d_end = local_data.end();
        for(; it_d != it_d_end; ++it_d)
            memset(it_d->second.data(), 0x00, sizeof(double) * it_d->second.size());

        MPI_Barrier(loc_comm);
    }

    MakePointersOnDiagonal(Matrix, comm);
}

void MatrixDiagonal3::MakePointersOnDiagonal(Epetra_CrsMatrix *Matrix, const Epetra_MpiComm *comm) {

    if (local_ids.empty())
        return;

    const int my_rank = comm->MyPID();

    /* Allocate memory */
    std::unordered_map<int, std::vector<int> >::iterator it_i, it_i_end;
    it_i = local_ids.begin();
    it_i_end = local_ids.end();
    for(; it_i != it_i_end; ++ it_i) {
        local_ptrs[it_i->first].resize(it_i->second.size());
    }

    /* Get diagonal values which should be exported */
    double *values = Matrix->ExpertExtractValues();                              // Get array of values
    Epetra_IntSerialDenseVector *indices = &Matrix->ExpertExtractIndices();     // Get array of column indices
    Epetra_IntSerialDenseVector *offsets = &Matrix->ExpertExtractIndexOffset(); // Get array of offsets
    std::unordered_map<int, std::vector<double*> >::iterator it_d, it_d_end;
    it_i = local_ids.begin();
    for(; it_i != it_i_end; ++ it_i) {
        int loc_pid = it_i->first;
        int loc_size = it_i->second.size();

        it_d = local_ptrs.find(loc_pid);

        if (it_d == it_d_end) {
            std::cerr << "Error! Can't find PID " << loc_pid
                    << " in the map of pointers on diagonal values. Rank " << my_rank << "."
                    << std::endl;
            break;
        }

        std::vector<double*> *loc_vec = &it_d->second;
        for(int lid = 0; lid < loc_size; ++lid) {
            int loc_row = it_i->second[lid];
            for(int j = offsets->operator ()(loc_row); j < offsets->operator ()(loc_row+1); ++j) {
                if (loc_row == indices->operator ()(j)) {
                    loc_vec->operator[](lid) = &values[j];
                }
            }
        }
    }

//    it_i = local_ids.begin();
//    for(; it_i != it_i_end; ++ it_i) {
//        int loc_pid = it_i->first;
//
//        it_d = local_ptrs.find(loc_pid);
//
//        std::cout << loc_pid << " (" << my_rank << ") : ";
//        for(uint32_t n = 0, end = it_d->second.size(); n < end; ++n)
//            std::cout << *it_d->second[n] << " ";
//        std::cout << "\n";
//    }
}

void MatrixDiagonal3::ExchangeDiagonalCoefficients(const Epetra_MpiComm *comm,
        std::unordered_map<int, std::vector<double> > &rcv_data) {

    int my_rank = 0;
    MPI_Comm_rank(comm->Comm(), &my_rank);
    std::unordered_map<int, std::vector<double*> >::iterator it_dp, it_dp_end;
    bool allocated = rcv_data.empty() ? false : true;

    it_dp = local_ptrs.begin();
    it_dp_end = local_ptrs.end();

    int neighbors = local_ptrs.size();
    MPI_Request request_snd[neighbors];
    MPI_Status status_snd[neighbors];
    int tag = 0;
    int counter = 0;
    for(; it_dp != it_dp_end; ++it_dp) {
        int pid = it_dp->first;
        uint32_t size = it_dp->second.size();
        std::vector<double> data(size);
        std::cout << my_rank << "(" << pid << ") Send : " << size << " : ";
        for(uint32_t n = 0; n < size; ++n) {
            data[n] = *it_dp->second[n];
            std::cout << data[n] << " ";
        }
        std::cout << std::endl;

        MPI_Isend(data.data(), size, MPI_DOUBLE, pid, tag, comm->Comm(), &request_snd[counter]);
        ++counter;
    }

    neighbors = local_data.size();
    MPI_Request request_rcv[neighbors];
    MPI_Status status_rcv[neighbors];
    std::unordered_map<int, std::vector<double> >::iterator it_d, it_d_end;
    it_d = local_data.begin();
    it_d_end = local_data.end();
    counter = 0;
    for(; it_d != it_d_end; ++it_d) {
        int pid = it_d->first;
        std::vector<double> *prt_out;

        // Allocate memory only if incoming map is empty
        if (!allocated) {
            rcv_data[pid].resize(local_data[pid].size());
        }
        prt_out = &rcv_data[pid];

        MPI_Irecv(prt_out->data(), prt_out->size(), MPI_DOUBLE, pid, tag, comm->Comm(), &request_rcv[counter]);
        ++counter;
    }

    MPI_Waitall(local_ids.size(), request_snd, status_snd);
    MPI_Waitall(local_data.size(), request_rcv, status_rcv);
}

int main(int argc, char** argv) {

    //  int NumMyElements = 1e+5;
    int _nx;
    int _ny;
    int _nz;
    int NumGlobalElements;

    if (argc > 1) {
        if (argc == 2) {
            _nx = atoi(argv[1]);
            _ny = _nz = 1;
        }
        else if (argc == 3) {
            _nx = atoi(argv[1]);
            _ny = atoi(argv[2]);
            _nz = 1;
        }
        else if (argc == 4) {
            _nx = atoi(argv[1]);
            _ny = atoi(argv[2]);
            _nz = atoi(argv[3]);
        }
        else {
            std::cout << "Too many arguments..." << std::endl;
            return 0;
        }
        NumGlobalElements = _nx * _ny * _nz;
    }
    else {
        _nx = _ny = 4;
        _nz = 1;
        NumGlobalElements = _nx * _ny * _nz;
//        std::cout << "Runned with 1 thread..." << std::endl;
    }

    MPI_Init(&argc, &argv);
    Epetra_MpiComm comm(MPI_COMM_WORLD);

    // Epetra_Comm has methods that wrap basic MPI functionality.
    // MyPID() is equivalent to MPI_Comm_rank, and NumProc() to
    // MPI_Comm_size.
    //
    // With a "serial" communicator, the rank is always 0, and the
    // number of processes is always 1.
    const int myRank = comm.MyPID();
    const int numProcs = comm.NumProc();
    Epetra_Time time(comm);
    double min_time, max_time;
    double time1, time2, full;

    if (myRank == 0) std::cout << "Problem size: " << NumGlobalElements << std::endl;

    Epetra_Map Map(NumGlobalElements, 0, comm);

    int NumMyElements = Map.NumMyElements();            // Number of local elements
    int *MyGlobalElements = Map.MyGlobalElements();    // Global index of local elements

//    std::cout << "NumMyElements: " << NumMyElements << " " << myRank << "\n";
    /*
     * Sparse matrix. 3 - is a guessed number of non-zeros per row
     */
    int dimension;
    if (_nz == 1)
        dimension = 4;
    else
        dimension = 6;

  Epetra_Map *myMap;  // if create do not forget do delete!
//  StripeDecomposition(myMap, _nx, _ny, _nz, NumGlobalElements, comm);
  Decomposition4(myMap, NumGlobalElements, comm);

  //  cout << *myMap << endl;

//  Epetra_CrsMatrix A(Copy, *myMap, dimension+1, false);

    /*
     * Lowlevel matrix assembly (row-by-row from left to right)
     */
    Epetra_CrsMatrix *A;
    A = new Epetra_CrsMatrix(Copy, *myMap, dimension+1, false);

//    std::cout << *myMap << "\n";
    time1 = time.WallTime();
    AssembleMatrixGlob(dimension, myMap, _nx, _ny, _nz, A);
    time2 = time.WallTime();
    MPI_Reduce(&time1, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, comm.Comm());
    MPI_Reduce(&time2, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm.Comm());
    if (myRank == 0) {
        full = max_time - min_time;
//        std::cout << "Assembly time: " << full << std::endl;
    }


//    time1 = time.WallTime();
//    UpdateMatrixGlob(dimension, &Map, _nx, _ny, _nz, &A);
//    time2 = time.WallTime();
//    MPI_Reduce(&time1, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, comm.Comm());
//    MPI_Reduce(&time2, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm.Comm());
//    if (myRank == 0) {
//        full = max_time - min_time;
//        std::cout << "Update time: " << full << std::endl;
//    }

    /*
     * Epetra vectors for Unknowns and RHS
     */
//    Epetra_Vector x(*myMap);
//    Epetra_Vector b(*myMap, false);
    Epetra_Vector x(Map);
    Epetra_Vector b(Map, false);

    double dx = 1./(_nx-1);
    b.PutScalar(1000. *dx * dx);

//    slv::Precond p;
//    p.build(*balanced_matrix, Jacobi_p);
//  p.print();

    time1 = time.WallTime();

//    /* **************************** */
//    /* **************************** */
//    /* **************************** */
//
//    // create a parameter list for ML options
//    Teuchos::ParameterList MLList;
//    // Sets default parameters for classic smoothed aggregation. After this
//    // call, MLList contains the default values for the ML parameters,
//    // as required by typical smoothed aggregation for symmetric systems.
//    // Other sets of parameters are available for non-symmetric systems
//    // ("DD" and "DD-ML"), and for the Maxwell equations ("maxwell").
//    ML_Epetra::SetDefaults("SA",MLList);
//    // overwrite some parameters. Please refer to the user's guide
//    // for more information
//    // some of the parameters do not differ from their default value,
//    // and they are here reported for the sake of clarity
//    // output level, 0 being silent and 10 verbose
//    MLList.set("ML output", 10);
//    // maximum number of levels
//    MLList.set("max levels",5);
//    // set finest level to 0
//    MLList.set("increasing or decreasing","increasing");
//    // use Uncoupled scheme to create the aggregate
//    MLList.set("aggregation: type", "Uncoupled");
//    // smoother is Chebyshev. Example file
//    // `ml/examples/TwoLevelDD/ml_2level_DD.cpp' shows how to use
//    // AZTEC's preconditioners as smoothers
//    MLList.set("smoother: type","Chebyshev");
//    MLList.set("smoother: sweeps",1);
//    // use both pre and post smoothing
//    MLList.set("smoother: pre or post", "both");
//  #ifdef HAVE_ML_AMESOS
//    // solve with serial direct solver KLU
//    MLList.set("coarse: type","Amesos-KLU");
//  #else
//    // this is for testing purposes only, you should have
//    // a direct solver for the coarse problem (either Amesos, or the SuperLU/
//    // SuperLU_DIST interface of ML)
//    MLList.set("coarse: type","Jacobi");
//  #endif
//    // Creates the preconditioning object. We suggest to use `new' and
//    // `delete' because the destructor contains some calls to MPI (as
//    // required by ML and possibly Amesos). This is an issue only if the
//    // destructor is called **after** MPI_Finalize().
//    double time3 = time.WallTime();
//    ML_Epetra::MultiLevelPreconditioner* MLPrec =
//      new ML_Epetra::MultiLevelPreconditioner(*A, MLList);
//
//    double time4 = time.WallTime();
//    MPI_Reduce(&time3, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, comm.Comm());
//    MPI_Reduce(&time4, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm.Comm());
//    if (myRank == 0) {
//        full = max_time - min_time;
//        std::cout << "ML time: " << "\t" << full << std::endl;
//    }
//
//    // verify unused parameters on process 0 (put -1 to print on all
//    // processes)
//    MLPrec->PrintUnused(0);
//  #ifdef ML_SCALING
//    timeVec[precBuild].value = MPI_Wtime() - timeVec[precBuild].value;
//  #endif
//    // ML allows the user to cheaply recompute the preconditioner. You can
//    // simply uncomment the following line:
//    //
//    // MLPrec->ReComputePreconditioner();
//    //
//    // It is supposed that the linear system matrix has different values, but
//    // **exactly** the same structure and layout. The code re-built the
//    // hierarchy and re-setup the smoothers and the coarse solver using
//    // already available information on the hierarchy. A particular
//    // care is required to use ReComputePreconditioner() with nonzero
//    // threshold.
//    // =========================== end of ML part =============================
//    // tell AztecOO to use the ML preconditioner, specify the solver
//    // and the output, then solve with 500 maximum iterations and 1e-12
//    // of tolerance (see AztecOO's user guide for more details)
//
//    /* **************************** */
//    /* **************************** */
//    /* **************************** */

//    // Create Linear Problem
//    Epetra_LinearProblem problem;
//
//    problem.SetOperator(A);
//    problem.SetLHS(&x);
//    problem.SetRHS(&b);
//
//    // Create AztecOO instance
//    AztecOO solver(problem);
//
////    solver.SetPrecOperator(MLPrec);
//    solver.SetAztecOption(AZ_conv, AZ_noscaled);
//    solver.SetAztecOption(AZ_solver, AZ_bicgstab);
////    solver.SetAztecOption(AZ_output, 0);
////    solver.SetAztecOption(AZ_precond, AZ_ilu);//AZ_dom_decomp);
////    solver.SetAztecOption(AZ_subdomain_solve, AZ_icc);
//    solver.SetAztecOption(AZ_precond, AZ_Jacobi);
//    solver.SetAztecOption(AZ_omega, 0.72);
////    solver.Iterate(30, 1.0E-8);
//
////    slv::BiCGSTAB solver;
////    solver.SetStopCriteria(RNORM);
////    solver.SetMaxIter(1000);
////    solver.SetTolerance(1e-8);
////    solver.PrintHistory(true, 1);
////    solver.solve(A, x, b, x);
//
//    /* time2 */
//    time2 = time.WallTime();
//    MPI_Reduce(&time1, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, comm.Comm());
//    MPI_Reduce(&time2, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm.Comm());

//    if (myRank == 0) {
//        full = max_time - min_time;
//        std::cout << numProcs << "\t" << full << std::endl;
//        std::cout << "Iterations: " << solver.NumIters() << "\n";
//        std::cout << "Residual: " << solver.TrueResidual() << "\n";
//        std::cout << "Residual: " << solver.ScaledResidual() << "\n";
//    }

//    std::cout << "\n\n\n" << std::endl;

    /* *************************************** */
    int entries = 0;
    double *values;
//    int *indices;

    int rows = A->NumMyRows();
    values = A->ExpertExtractValues();
    Epetra_IntSerialDenseVector *indices = &A->ExpertExtractIndices();
    Epetra_IntSerialDenseVector *offsets = &A->ExpertExtractIndexOffset();

//    for(int i = 0; i < rows; ++i) {
//        for(int j = offsets->operator ()(i); j < offsets->operator ()(i+1); ++j) {
//            if (i == indices->operator ()(j)) {
//                values[j] = i;
////                std::cout << "diag : " << values[j] << "\n";
//            }
//        }
//    }

    /* ************************************************** */
    /* 0) Create a map
     * 1) Extract diagonal
     * 2) import diagonal elements */
    /* ************************************************** */
//    int glob_el[2];
//    if (comm.MyPID() == 0) {
//        glob_el[0] = 15;
//        glob_el[1] = 12;
//    }
//    else {
//        glob_el[0] = 0;
//        glob_el[1] = 2;
//    }
//    Epetra_Map TargetMap(-1, 2, glob_el, 0, comm);
//    Epetra_Import Import(TargetMap, A->Map());
//    Epetra_CrsMatrix B(Copy, TargetMap, dimension+1);
//
//    Epetra_Vector diagA(A->Map());
//    Epetra_Vector diagB(TargetMap);
//
//    A->ExtractDiagonalCopy(diagA);
//    diagB.Import(diagA, Import, Insert);
//
//    std::cout << "diagB: " << "\n";
//    for(int n = 0; n < diagB.MyLength(); ++n)
//        std::cout << diagB[n] << "\n";
    /* ************************************************** */


//    std::cout << "error: " << A->ExtractGlobalRowView(1, entries, values, indices) << "\n";
//
//    A->ExtractDiagonalCopy(Diagonal)
//
//    std::cout << "\n";
//    std::cout << "entries: " << entries << "\n";
//    for(int n = 0; n < entries; ++n) {
//        std::cout << values[n] << " " << "\n";
//    }
//    std::cout << *A << "\n";
//    values[0] = -99999.;
//    std::cout << *A << "\n";

    int glob_el[2];
    if (comm.MyPID() == 0) {
        glob_el[0] = 15;
        glob_el[1] = 12;
    }
    else {
        glob_el[0] = 0;
        glob_el[1] = 2;
    }
    Epetra_Map TargetMap(-1, 2, glob_el, 0, comm);

//    std::cout << Map << "\n";
//    std::cout << TargetMap << "\n";

//    Epetra_Import Import(TargetMap, Map);
//    Epetra_Vector y(TargetMap);
//
//    for(int n = 0; n < y.MyLength(); ++n)
//        y[n] = 0;
//
//    for(int n = 0; n < x.MyLength(); ++n)
//        x[n] = n;

//    A->ExtractDiagonalCopy(x);

//    if (A->Filled())
//        std::cout << "************** Matrix A is filled\n";
//    else
//        std::cout << "-------------- Matrix A is not filled\n";
//    Epetra_Import Import(TargetMap, A->Map());
//    Epetra_CrsMatrix B(Copy, TargetMap, dimension+1);
//
//    Epetra_Vector diagA(A->Map());
//    Epetra_Vector diagB(TargetMap);
//
//    A->ExtractDiagonalCopy(diagA);
//    diagB.Import(diagA, Import, Insert);

//    std::cout << "diagB: " << "\n";
//    for(int n = 0; n < diagB.MyLength(); ++n)
//        std::cout << diagB[n] << "\n";

//    /* How to get a map of ghost elements */
//    const Epetra_Import *imp = A->Importer();
//    if (imp != nullptr) {
//        std::cout << "(" << myRank << ") nums: " << imp->NumSend() << " " << imp->NumRecv() << "\n";
//        std::cout << "(" << myRank << ") NumExportIDs: " << imp->NumExportIDs() << "\n";
//        int numexids = imp->NumExportIDs();
//        int *procs, *ids;
//        ids = imp->ExportLIDs();                // get ids which should be exported
//        procs = imp->ExportPIDs();              // get process ids to which those ids should be exported
//        std::cout << "(" << myRank << ") procs: ";
//        if (procs != nullptr)
//            for(int n = 0; n < numexids; ++n) {
//                std::cout << procs[n] << " ";
//            }
//        std::cout << "\n";
//        std::cout << "(" << myRank << ") ids: ";
//        if (ids !=  nullptr)
//            for(int n = 0; n < numexids; ++n) {
//                std::cout << ids[n] << " ";
//            }
//        std::cout << "\n";
//    }
//    /* ********************************* */

    MPI_Barrier(MPI_COMM_WORLD);
    if (A->Importer() != nullptr) {
//        std::cout << "(" << myRank << ") A "
//                    << "NumSend : " << A->Importer()->NumSend() << "\n"
//                    << "NumRecv : " << A->Importer()->NumRecv() << "\n"
//                    << "NumRemoteIDs : " << A->Importer()->NumRemoteIDs() << "\n"
//                    << "NumExportIDs : " << A->Importer()->NumExportIDs() << "\n";
//
//        std::cout << "(" << myRank << ") RemoteLIDs : ";
//        for(int n = 0; n <A->Importer()->NumRemoteIDs(); ++n)
//            std::cout << A->Importer()->RemoteLIDs()[n] << " ";
//        std::cout << "\n";
//
//        std::cout << "(" << myRank << ") ExportLIDs : ";
//        for(int n = 0; n <A->Importer()->NumExportIDs(); ++n)
//            std::cout << A->Importer()->ExportLIDs()[n] << " ";
//        std::cout << "\n";
//
//        std::cout << "(" << myRank << ") ExportPIDs : ";
//        for(int n = 0; n <A->Importer()->NumExportIDs(); ++n)
//            std::cout << A->Importer()->ExportPIDs()[n] << " ";
//        std::cout << "\n";
//
//        std::cout << "(" << myRank << ") PermuteFromLIDs : ";
//        for(int n = 0; n <A->Importer()->NumPermuteIDs(); ++n)
//            std::cout << A->Importer()->PermuteFromLIDs()[n] << " ";
//        std::cout << "\n";
//
//        std::cout << "(" << myRank << ") PermuteToLIDs : ";
//        for(int n = 0; n <A->Importer()->NumPermuteIDs(); ++n)
//            std::cout << A->Importer()->PermuteToLIDs()[n] << " ";
//        std::cout << "\n";
//        std::cout << std::endl;

//        /*
//         * Use two maps: one to store initially size of individual messages that should be sent,
//         * another one to store messages themselves. Keys are pids which should receive messages.
//         */
//        /* Count how many data points should be sent to each process */
//        std::unordered_map<int, int> count;
//        std::unordered_map<int, int>::iterator it_c, it_c_end;
//        for(int i = 0; i < A->Importer()->NumExportIDs(); i++)
//            count[A->Importer()->ExportPIDs()[i]]++;
//
////        it_c = count.begin();
////        it_c_end = count.end();
////        for(; it_c != it_c_end; ++it_c)
////            std::cout << "(" << myRank << ") Map : " << it_c->first << ": " << it_c->second << "\n";
////        std::cout << "\n";
//
//        int *data = A->Importer()->ExportLIDs();
//        int data_size = A->Importer()->NumExportIDs();
//        std::unordered_map<int, std::vector<int> > snd_buf;
//
//        // Allocate memory for the sending map
//        it_c = count.begin();
//        for(int n = 0; n < count.size(); ++n) {
//            snd_buf[it_c->first].resize(it_c->second);
//            ++it_c;
//        }
//
//        // Assign data to the sending map
//        int offset = 0;
//        std::unordered_map<int, std::vector<int> >::iterator it_d, it_d_end;
//        it_d = snd_buf.begin();
//        it_d_end = snd_buf.end();
//
//        // original data consists of chunks which we need to extract and send separately to each neighbor
//        it_c = count.begin();
//        it_c_end = count.end();
//        for(int n = 0; n < data_size; ++n) {
//            int pid = A->Importer()->ExportPIDs()[n];                                           // check out the pid
//            it_d = snd_buf.find(pid);
//            if (it_d != it_d_end) {                                                             // if pid found we are safe to go
//                int chunk_size = it_d->second.size();                                           // get number of elements in a chunk
//                memcpy(it_d->second.data(), data + offset, sizeof(int) * chunk_size);           // copy particular chunk from the original array
//                n += chunk_size;
//                offset += chunk_size;
//            }
//            else
//                std::cout << "Error!" << "\n";
//        }
//
////        std::sort(p.begin(), p.end(),
////            [&](std::size_t i, std::size_t j){ return compare(vec[i], vec[j]); });
//
//        it_d = snd_buf.begin();
//        it_d_end = snd_buf.end();
//        for(; it_d != it_d_end; ++it_d) {
//            std::sort(it_d->second.begin(), it_d->second.end());
//            std::cout << "(" << myRank << ") Snd buf : " << it_d->first << ": ";
//            for(int n = 0; n < it_d->second.size(); ++n)
//                std::cout << it_d->second[n] << " ";
//            std::cout << "\n";
//        }
        /* ******************** */

//        int *export_lids = A->Importer()->ExportLIDs();
//        std::vector<int> rcv(A->Importer()->NumRecv());
//        int rcv_counter = 0;
//
//        MPI_Request send_request[A->Importer()->NumSend()];
//        MPI_Request recv_request[A->Importer()->NumSend()];
//        for(int n = 0; n < A->Importer()->NumSend(); ++n) {
//            MPI_Isend(export_lids + n, 1, MPI_INT, A->Importer()->ExportPIDs()[n], 0, MPI_COMM_WORLD, &send_request[n]);
//            MPI_Irecv(rcv.data() + n, 1, MPI_INT, A->Importer()->ExportPIDs()[n], 0, MPI_COMM_WORLD, &recv_request[n]);
//        }
//
//        MPI_Status status_arr[A->Importer()->NumSend()];
//        MPI_Waitall(A->Importer()->NumSend(), send_request, status_arr);
//        MPI_Waitall(A->Importer()->NumSend(), recv_request, status_arr);
//
//        std::cout << "(" << myRank << ") Received: ";
//        for(int n = 0; n < A->Importer()->NumRecv(); ++n) {
//            std::cout << rcv[n] << " ";
//        }
//        std::cout << "\n";
//        std::cout << "(" << myRank << ") Pids: ";
//        for(int n = 0; n < A->Importer()->NumRecv(); ++n) {
//            std::cout << A->Importer()->ExportPIDs()[n] << " ";
//        }
//        std::cout << "\n";
    }
    if (A->Exporter() != nullptr)
        std::cout << "(" << myRank << ") A nums (exp): " << A->Exporter()->NumSend() << " " << A->Exporter()->NumRecv() << "\n";

    /* How to get a map of ghost elements */
    const Epetra_Import *imp = A->Importer();
//    const Epetra_Export *imp = A->Exporter();
//    if (imp != nullptr) {
//        std::cout << "(" << myRank << ") NumRemoteIDs: " << imp->NumRemoteIDs() << "\n";
//        int numexids;
//        int *procs, *ids;
//        numexids = imp->NumExportIDs();
//        ids = imp->ExportLIDs();                // get ids which should be exported
//        procs = imp->ExportPIDs();              // get process ids to which those ids should be exported
//
//        std::vector<double> snd(numexids);
//        std::vector<int> pid(numexids);
//        values = A->ExpertExtractValues();
//        indices = &A->ExpertExtractIndices();
//        offsets = &A->ExpertExtractIndexOffset();
//        std::cout << "(" << myRank << ") : ";
//        for(int i = 0; i < numexids; ++i) {
//            int loc_row = ids[i];
//            for(int j = offsets->operator ()(loc_row); j < offsets->operator ()(loc_row+1); ++j) {
//                if (loc_row == indices->operator ()(j)) {
//                    snd[i] = 20+values[j];
//                    pid[i] = procs[i];
//                }
//            }
//
//            std::cout << snd[i] << " ";
//        }
//        std::cout << "\n";
//
//        /*
//         * Use two maps: one to store initially size of individual messages that should be sent,
//         * another one to store messages themselves. Keys are pids which should receive messages.
//         */
//        /* Count how many data points should be sent to each process */
//        std::unordered_map<int, int> count;
//        std::unordered_map<int, int>::iterator it_c, it_c_end;
//        for(int i = 0; i < A->Importer()->NumExportIDs(); i++)
//            count[A->Importer()->ExportPIDs()[i]]++;
//
////        it_c = count.begin();
////        it_c_end = count.end();
////        for(; it_c != it_c_end; ++it_c)
////            std::cout << "(" << myRank << ") Map : " << it_c->first << ": " << it_c->second << "\n";
////        std::cout << "\n";
//
//        double *data = snd.data();
//        int data_size = snd.size();
//        int *lids = A->Importer()->ExportLIDs();
//        std::unordered_map<int, std::vector<double*> > snd_buf;
//        std::unordered_map<int, std::vector<int> > ids_buf;
//
//        // Allocate memory for the sending map
//        it_c = count.begin();
//        for(int n = 0; n < count.size(); ++n) {
//            snd_buf[it_c->first].resize(it_c->second);
//            ids_buf[it_c->first].resize(it_c->second);
//            ++it_c;
//        }
//
//        // Assign data to the sending map
//        int offset = 0;
//        std::unordered_map<int, std::vector<double*> >::iterator it_d, it_d_end;
//        std::unordered_map<int, std::vector<int> >::iterator it_i, it_i_end;
//        it_d = snd_buf.begin();
//        it_d_end = snd_buf.end();
//
//        // original data consists of chunks which we need to extract and send separately to each neighbor
//        it_c = count.begin();
//        it_c_end = count.end();
//        for(int n = 0; n < data_size; ++n) {
//            int pid_loc = A->Importer()->ExportPIDs()[n];                                           // check out the pid
//            it_d = snd_buf.find(pid_loc);
//            if (it_d != it_d_end) {                                                             // if pid found we are safe to go
//                int chunk_size = it_d->second.size();                                           // get number of elements in a chunk
////                memcpy(it_d->second.data(), data + offset, sizeof(double) * chunk_size);           // copy particular chunk from the original array
//                for(int n = 0; n < chunk_size; ++n)
//                    it_d->second[n] = data + offset + n;
//                memcpy(ids_buf[pid_loc].data(), lids + offset, sizeof(int) * chunk_size);           // copy particular chunk from the original array
//                n += chunk_size;
//                offset += chunk_size;
//            }
//            else
//                std::cout << "Error!" << "\n";
//        }
//
//        it_d = snd_buf.begin();
//        it_d_end = snd_buf.end();
//        for(; it_d != it_d_end; ++it_d) {
//            it_i = ids_buf.find(it_d->first);
//            int size = it_i->second.size();
//            // do stupid bubble sort of data and index vectors...
//            for(int n = 0; n < size; ++n) {
//                int max_ind = n;
//                int my_value = it_i->second[max_ind];
//                for(int m = n + 1; m < size; ++m) {
//                    if (it_i->second[m] < my_value) {
//                        std::swap(it_i->second[m], it_i->second[max_ind]);
////                        std::swap(it_d->second[m], it_d->second[max_ind]);
//                        double *tmp = it_d->second[m];
//                        it_d->second[m] = it_d->second[max_ind];
//                        it_d->second[max_ind] = tmp;
//                        max_ind = m;
//                        my_value = it_i->second[max_ind];
//                    }
//                }
//            }
//            std::cout << "(" << myRank << ") Snd buf : " << it_d->first << ": ";
//            for(int n = 0; n < it_d->second.size(); ++n)
//                std::cout << *it_d->second[n] << "(" << it_i->second[n] << ")" << " ";
//            std::cout << "\n";
//        }
//        // after this point we have two sorted vectors: vector of pointers on diagonal elements and
//        // vector of corresponding row indices
//        // TODO: wrap it in a single class (store map of pids and elements, map of pids and indices)
//    }
    /* ********************************* */

//    MatrixDiagonal md;

//    md.RegisterDiagonal(A, &comm);
//    std::vector<std::vector<double> > rcv_data;
//    std::vector<std::vector<int> > rcv_ind;
//    md.Send(comm.Comm(), rcv_data, rcv_ind);

//    MatrixDiagonal2 md;
//    bool sendig_proc = false;
//    int NumIDs = 2;
//    std::vector<int> list_gid;
//
//    /* 1) Get a GID */
//    if (myRank == 0) {
//        list_gid.resize(NumIDs);
//        list_gid[0] = 8;
//        list_gid[1] = 9;
//        sendig_proc = true;
//    }
//    else if (myRank == 1) {
//        list_gid.resize(NumIDs);
//        list_gid[0] = 10;
//        list_gid[1] = 11;
//        sendig_proc = true;
//    }
//    md.RegisterDiagonal(A, &comm, list_gid);

    MatrixDiagonal3 md;
    bool sendig_proc = false;
    int NumIDs = 2;
    std::vector<int> list_gid;

    /* 1) Get a GID */
    if (myRank == 0) {
        list_gid.resize(NumIDs);
        list_gid[0] = 8;
        list_gid[1] = 9;
        sendig_proc = true;
    }
    else if (myRank == 1) {
        list_gid.resize(NumIDs);
        list_gid[0] = 10;
        list_gid[1] = 11;
        sendig_proc = true;
    }
    md.RegisterDiagonal(A, &comm, list_gid);

    std::unordered_map<int, std::vector<double> > rcv_data;
    std::unordered_map<int, std::vector<double> >::iterator it;
    md.ExchangeDiagonalCoefficients(&comm, rcv_data);
    md.ExchangeDiagonalCoefficients(&comm, rcv_data);

    std::cout << "(" << myRank << ") Rcv buf : ";
    for(it = rcv_data.begin(); it != rcv_data.end(); ++it) {
        for(uint32_t n = 0; n < it->second.size(); ++n) {
            std::cout << it->second[n] << "(" << it->first << ") ";
        }
        std::cout << "\n              ";
    }
    std::cout << std::endl;

//    /* Complex communication :) */
//    bool sendig_proc = false;
//    int NumIDs = 0;
////    std::vector<int> GIDList, PIDList, LIDList;
////    GIDList.resize(NumIDs);
////    PIDList.resize(NumIDs);
////    LIDList.resize(NumIDs);
//    int *GIDList = nullptr;
//    int *PIDList = nullptr;
//    int *LIDList = nullptr;
//
//    /* 1) Get a GID */
//    if (myRank == 0) {
//        NumIDs = 2;
////        GIDList.resize(NumIDs);
////        PIDList.resize(NumIDs);
////        LIDList.resize(NumIDs);
//        GIDList = new int [2];
//        PIDList = new int [2];
//        LIDList = new int [2];
//        GIDList[0] = 8;
//        GIDList[1] = 9;
//        sendig_proc = true;
//    }
//    else if (myRank == 1) {
//        NumIDs = 2;
////        GIDList.resize(NumIDs);
////        PIDList.resize(NumIDs);
////        LIDList.resize(NumIDs);
//        GIDList = new int [2];
//        PIDList = new int [2];
//        LIDList = new int [2];
//        GIDList[0] = 10;
//        GIDList[1] = 11;
//        sendig_proc = true;
//    }
//    else {
////        GIDList = new int [2];
////        PIDList = new int [2];
////        LIDList = new int [2];
////        GIDList[0] = 0;
////        GIDList[1] = 0;
//    }
//
//    /* 2) Get PIDs and LIDs corresponding to GIDs */
////    A->Map().RemoteIDList(NumIDs, GIDList.data(), PIDList.data(), LIDList.data());
//    int sizzzz = 0;
//    A->Map().RemoteIDList(NumIDs, GIDList, PIDList, LIDList, &sizzzz);
//
////    std::cout << "sizzzz " << sizzzz << " : ";
//    for(int n = 0; n < NumIDs; ++n)
//        std::cout << "(" << myRank << ") : " << GIDList[n] << ", " << PIDList[n] << ", " << LIDList[n] << "\n";
//
////    if (myRank == 0 || myRank == 1)
//    {
//        delete [] GIDList;
//        delete [] PIDList;
//        delete [] LIDList;
//    }

//
//    /* 3) Send an array of LIDs to other process and inform it that we are waiting for some data from it */
//    /*   (only for processes who has to collect remote data) */
//    if (sendig_proc)
//    {
//        MPI_Send(LIDList.data(), LIDList.size(), MPI_INT, PIDList[0], 0, MPI_COMM_WORLD);           // <--- change PIDList[0]!!!
//    }
//    MPI_Barrier(MPI_COMM_WORLD);            // !!! Absolutely necessary !!!
//
//    /* 4) Check by all processes if they should receive any message */
//    int flag;
//    MPI_Status status;
//    MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
//    if (flag == true) {
//        /* If so - receive the message */
//        std::cout << "PID " << myRank << " has received a message from PID " << status.MPI_SOURCE << "\n";
//
//        std::vector<int> LIDList;
//        int msg_size = 0;
//        MPI_Get_count(&status, MPI_INT, &msg_size);
//        LIDList.resize(msg_size);
//        MPI_Recv(LIDList.data(), msg_size, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &status);
//
//        for(int n = 0; n < msg_size; ++n)
//            std::cout << LIDList[n] << " ";
//        std::cout << std::endl;
//
//        /* 5) Assemble and send back the new message consists of values from the diagonal */
//        double *values = A->ExpertExtractValues();                              // Get array of values
//        Epetra_IntSerialDenseVector *indices = &A->ExpertExtractIndices();     // Get array of column indices
//        Epetra_IntSerialDenseVector *offsets = &A->ExpertExtractIndexOffset(); // Get array of offsets
//
//        /* Assemble diagonal values which should be exported */
//        std::vector<double> diag_values;
//        diag_values.resize(LIDList.size());
//        for(uint32_t i = 0; i < LIDList.size(); ++i) {
//            int loc_row = LIDList[i];
//            for(int j = offsets->operator ()(loc_row); j < offsets->operator ()(loc_row+1); ++j) {
//                if (loc_row == indices->operator ()(j)) {
//                    diag_values[i] = 20 + values[j];
//                }
//            }
//        }
//        MPI_Send(diag_values.data(), diag_values.size(), MPI_DOUBLE, status.MPI_SOURCE, 1, MPI_COMM_WORLD);           // <--- change PIDList[0]!!!
//    }
//    else
//        std::cout << "PID " << myRank << " has not received any message\n";
//
//    /* 6) Receive the data as a response on sent message */
//    if (sendig_proc)
//    {
//        MPI_Probe(PIDList[0], 1, MPI_COMM_WORLD, &status);
//        int msg_size = 0;
//        MPI_Get_count(&status, MPI_DOUBLE, &msg_size);
//        std::vector<double> rcv_msg(msg_size);
//        MPI_Recv(rcv_msg.data(), msg_size, MPI_DOUBLE, PIDList[0], 1, MPI_COMM_WORLD, &status);
//
//        std::cout << "Received: ";
//        for(int n = 0; n < msg_size; ++n)
//            std::cout << rcv_msg[n] << " ";
//        std::cout << std::endl;
//    }





//    MPI_Wait(&recv_request, &status);
//    MPI_Wait(&send_request, &status);


//    std::cout << "(" << myRank << ") Rcv buf : ";
//    for(uint32_t n = 0; n < rcv_data.size(); ++n) {
//        for(uint32_t m = 0; m < rcv_data[n].size(); ++m) {
//            std::cout << rcv_data[n][m] << "(" << rcv_ind[n][m] << ") ";
//        }
//        std::cout << "\n              ";
//    }
//    std::cout << std::endl;

//    std::cout << Map << "\n";
//    std::cout << "error Import: " << B.Import(*A, Import, Insert) << "\n";
//    B.FillComplete();
//    std::cout << "error FillComplete: " << B.FillComplete() << "\n";
//    std::cout << B << "\n";
////    rows = B.NumMyRows();
//
//    if (B.Filled())
//        std::cout << "************** Matrix B is filled\n";
//    else
//        std::cout << "-------------- Matrix B is not filled\n";

//    const Epetra_Import *Importer = A->Importer();

//    B.Import(*A, *Importer, Insert);
//    std::cout << B << "\n";
//    Importer->Print(std::cout);

//    Epetra_Vector diag(B.Map());
//    B.ExtractDiagonalCopy(diag);
//
//    std::cout << "diag: " << "\n";
//    for(int n = 0; n < diag.MyLength(); ++n)
//        std::cout << diag[n] << "\n";

//    double *val_loc = nullptr;
//    int *ind_loc = nullptr;
//    int *off_loc = nullptr;
//    int length = B.NumMyNonzeros();
//    B.ExtractCrsDataPointers(off_loc, ind_loc, val_loc);
//
//    if (off_loc == nullptr)
//        std::cout << "off_loc" << "\n";
//    if (ind_loc == nullptr)
//        std::cout << "ind_loc" << "\n";
//    if (val_loc == nullptr)
//        std::cout << "val_loc" << "\n";
//    std::cout << "rows: " << rows << "\n";
//    std::cout << "indices  values\n";
////    for(int n = 0; n < length; ++n) {
////        std::cout << ind_loc[n] << " " << val_loc[n] << "\n";
////    }
//    std::cout << "offsets\n";
//    for(int n = 0; n < B.NumMyRows(); ++n) {
//        std::cout << off_loc[n] << "\n";
//    }

//    for(int i = 0; i < rows; ++i) {
//        for(int j = offsets->operator ()(i); j < offsets->operator ()(i+1); ++j) {
//            if (i == indices->operator ()(j)) {
//                std::cout << "diag : " << values[j] << "\n";
//            }
//        }
//    }


//    y.Import(x, Import, Insert);
//    std::cout << "NumExportIDs: " << A->Importer()->NumExportIDs() << "\n";
//    std::cout << "NumSameIDs: " << A->Importer()->NumSameIDs() << "\n";
//    std::cout << "NumRemoteIDs: " << A->Importer()->NumRemoteIDs() << "\n";

//    double *t;
//
//    t = y.Values();
//
//    for(int n = 0; n < y.MyLength(); ++n)
//        std::cout << n << ": " << t[n] << "\n";
//
////    std::cout << x << "\n";
////
//    std::cout << y << "\n";

//    delete [] values;
//    delete [] indices;

      delete myMap;
    delete A;
//    delete MLPrec;

#ifdef HAVE_MPI
    // Since you called MPI_Init, you are responsible for calling
    // MPI_Finalize after you are done using MPI.
    (void)MPI_Finalize();
#endif // HAVE_MPI

    return 0;
}

inline double getRealTime() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return (double)tv.tv_sec + 1.0e-6 * (double)tv.tv_usec;
}

int StripeDecomposition(Epetra_Map *&Map, int _nx, int _ny, int _nz, int _size,
    const Epetra_MpiComm &comm) {

    int myRank = comm.MyPID();
    int numProc = comm.NumProc();
    int chunk_size;
    int chunk_start;
    int chunk_end;
    int nynz;

    /*
     * First check if this kind of decomposition is possible at all
     */
    if (numProc > _nx) {
        if (myRank == 0)
            std::cout << "ERROR: Map for stripe decomposition can't be performed, since number "
                "of cores is greater than number of nodes along x direction.\n"
                "Standard Epetra Map will be create instead...\t" << std::endl;
        Map = new Epetra_Map(_size, 0, comm);
        return 1;
    }

    /*
     * Treat 2d case
     */
    if (_nz == 0) _nz = 1;

    nynz = _ny * _nz;

    chunk_size = _nx / numProc; // because c always round to the lower boundary
    chunk_start = myRank * chunk_size;
    chunk_end = chunk_start + chunk_size;

    /*
     * Assign last process with the end of domain, so it will contain last stripe
     */
    if (myRank == (numProc - 1)) chunk_end = _nx;

    chunk_size = (chunk_end - chunk_start) * nynz;

    Map = new Epetra_Map(_size, chunk_size, 0, comm);

    return 0;
}


int Decomposition4(Epetra_Map *&Map, int _size, const Epetra_MpiComm &comm) {

    int my_pid = comm.MyPID();
    int MyElements = 0;
    int *MyGlobalElements;
    switch(my_pid) {
        case 0:
            MyElements = 4;
            MyGlobalElements = new int[MyElements];
            MyGlobalElements[0] = 0;
            MyGlobalElements[1] = 1;
            MyGlobalElements[2] = 4;
            MyGlobalElements[3] = 5;
            break;

        case 1:
            MyElements = 4;
            MyGlobalElements = new int[MyElements];
            MyGlobalElements[0] = 2;
            MyGlobalElements[1] = 3;
            MyGlobalElements[2] = 6;
            MyGlobalElements[3] = 7;
            break;

        case 2:
            MyElements = 4;
            MyGlobalElements = new int[MyElements];
            MyGlobalElements[0] = 8;
            MyGlobalElements[1] = 9;
            MyGlobalElements[2] = 12;
            MyGlobalElements[3] = 13;
            break;

        case 3:
            MyElements = 4;
            MyGlobalElements = new int[MyElements];
            MyGlobalElements[0] = 10;
            MyGlobalElements[1] = 11;
            MyGlobalElements[2] = 14;
            MyGlobalElements[3] = 15;
            break;
    }

    Map = new Epetra_Map(-1, MyElements, MyGlobalElements, 0, comm);

    return 0;
}

int Decomposition3(Epetra_Map *&Map, int _size, const Epetra_MpiComm &comm) {

    int my_pid = comm.MyPID();
    int MyElements = 0;
    int *MyGlobalElements;
    switch(my_pid) {
        case 0:
            MyElements = 4;
            MyGlobalElements = new int[MyElements];
            MyGlobalElements[0] = 0;
            MyGlobalElements[1] = 1;
            MyGlobalElements[2] = 4;
            MyGlobalElements[3] = 5;
            break;

        case 1:
            MyElements = 4;
            MyGlobalElements = new int[MyElements];
            MyGlobalElements[0] = 2;
            MyGlobalElements[1] = 3;
            MyGlobalElements[2] = 6;
            MyGlobalElements[3] = 7;
            break;

        case 2:
            MyElements = 8;
            MyGlobalElements = new int[MyElements];
            MyGlobalElements[0] = 8;
            MyGlobalElements[1] = 9;
            MyGlobalElements[2] = 10;
            MyGlobalElements[3] = 11;
            MyGlobalElements[4] = 12;
            MyGlobalElements[5] = 13;
            MyGlobalElements[6] = 14;
            MyGlobalElements[7] = 15;
            break;
    }

    Map = new Epetra_Map(-1, MyElements, MyGlobalElements, 0, comm);

    return 0;
}




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
int Decomposition2(Epetra_Map *&Map, int _size, const Epetra_MpiComm &comm);
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

    int NumMyElements = A->Map().NumMyElements();           // Number of local elements
    int *MyGlobalElements = A->Map().MyGlobalElements();    // Global index of local elements
    int NumGlobalElements = A->Map().NumGlobalElements();   // Number of global elements

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

/*!
 * \class MatrixDiagonal
 * \brief Responsible for exchange of off-process diagonal elements
 * Class contains method which help to receive off-process diagonal elements. To use it one needs first
 * to call for RegisterDiagonal(). This method will setup all internal maps necessary for communication.
 * Once maps are setup one can use ExchangeDiagonalCoefficients() method.
 */
class MatrixDiagonal {
public:
    MatrixDiagonal() { }
    ~MatrixDiagonal() { }

    /*
     * \brief Registers pointer on diagonal elements that should be sent in internal maps
     * Method fills internal maps and makes pointers to the diagonal elements and their row ids. Later
     * this data can be exported to the neighbor processes
     *
     * \note Method has no effect if has already been called of for serial code (or with 1 process)
     *
     * \note Method will look for GIDs in the \e list_gid only in the neighborhood of <this> process.
     * The neighborhood is defined by the internal map of Epetra_CrsMatrix object.
     *
     * @param Matrix Pointer to the local matrix object
     * @param comm MPI communicator
     * @param list_gid List of global IDs which <this> process is looking for
     */
    void RegisterDiagonal(Epetra_CrsMatrix *Matrix, const MPI_Comm comm,
            const std::vector<int> &list_gid, std::unordered_map<int, std::vector<int> > &found_lids);

    /*
     * \brief Receives diagonal elements
     * Receives diagonal elements from processes determined during RegisterDiagonal() call.
     * @param comm MPI communicator
     * @param rcv_data Map of received data (key - PID, value - vector of received diagonal elements)
     */
    void ReceiveDiagonalCoefficients(const MPI_Comm comm,
            std::unordered_map<int, std::vector<double> > &rcv_data);

    /*
     * \brief Updates internal map of diagonal coefficients
     * @param comm MPI communicator
     * @param rcv_data Map of received data (key - PID, value - vector of received diagonal elements)
     */
    void UpdateDiagonalCoefficients(const MPI_Comm comm);

    /*!
     * \brief Returns pointer to the vector of diagonal values that is sent to proc \e pid
     * @param pid Receiving process
     */
    inline std::vector<double*> *GetDiagPointers(const int pid) {
        std::unordered_map<int, std::vector<double*> >::iterator it;
        it = local_ptrs.find(pid);
        if (it == local_ptrs.end()) {
            std::cout << "Error! Can't find PID " << pid
                    << " in the map of diagonal values." << std::endl;
            return nullptr;
        }
        return &it->second;
    }

    /*!
     * \brief Returns pointer to the vector of local ids that is sent to proc \e pid
     * @param pid Receiving process
     */
    inline std::vector<int> *GetLocalIDs(const int pid) {
        std::unordered_map<int, std::vector<int> >::iterator it;
        it = local_ids.find(pid);
        if (it == local_ids.end()) {
            std::cout << "Error! Can't find PID " << pid
                    << " in the map of diagonal values." << std::endl;
            return nullptr;
        }
        return &it->second;
    }

    /*!
     * \brief Returns pointer to the vector of local data that is received from proc \e pid
     * @param pid Receiving process
     */
    inline std::vector<double> *GetLocalData(const int pid) {
        std::unordered_map<int, std::vector<double> >::iterator it;
        it = local_data.find(pid);
        if (it == local_data.end()) {
            std::cout << "Error! Can't find PID " << pid
                    << " in the map of diagonal values." << std::endl;
            return nullptr;
        }
        return &it->second;
    }

    /*!
     * \brief Returns number of elements in a map of local data
     */
    inline uint32_t GetLocalDataPids() {
        return local_data.size();
    }

    /*!
     * \brief Returns pointer to the vector of local data that is received from proc \e pid
     * @param pid Receiving process
     */
    inline std::unordered_map<int, std::vector<double> > *GetLocalDataMap() {
        return &local_data;
    }

private:
    /*!
     * \brief Returns number of process' neighbors counted using provided communicator
     * \note If communicator doesn't describe any topology the method returns total number of
     * processes. Since this method is used to determine loops ranges it may affect the performance.
     * Therefore, it is highly recommended to use communicator with topology.
     * @param comm MPI communicator
     */
    int CountMyNeighbors(const MPI_Comm comm);

    /*!
     * \brief Makes pointers to local diagonal elements determined after RegisterDiagonal() call
     * Results will be written in the internal map \e local_ptrs
     * @param Matrix Pointer to the local matrix object
     * @param comm MPI communicator
     */
    void MakePointersOnDiagonal(Epetra_CrsMatrix *Matrix, const MPI_Comm comm);

    /*!
     * \brief Sends data from one map to another
     * \note Internal maps \e local_data and \e local_ids should already been allocated and filled. The method
     * is using them to pre-allocate \e received_map.
     * @param sent_map The map to be sent
     * @param received_map The map to be received
     */
    void SendMapToNeighbor(const MPI_Comm comm, const std::unordered_map<int, std::vector<int> > &sent_map,
            std::unordered_map<int, std::vector<int> > *received_map);

private:
    /*
     * The idea: use local_ids to assemble a vector of double from local_ptrs and send the vector to
     * the neighbor. Store received message in local_data.
     *
     * In all maps below the key stores process id to which data should be send or from which it
     * should be received. Note, not always there is only one process to communicate with.
     */
    // why unordered? Honestly, don't know...
    std::unordered_map<int, std::vector<int> > local_ids;           //!< Local IDs which should be sent to the neighbor(s)
    std::unordered_map<int, std::vector<double> > local_data;       //!< Local data that is received from the nighbor(s)
    std::unordered_map<int, std::vector<double*> > local_ptrs;      //!< Local pointers on diagonal elements of the matrix
};

int MatrixDiagonal::CountMyNeighbors(const MPI_Comm comm) {

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

void MatrixDiagonal::RegisterDiagonal(Epetra_CrsMatrix *Matrix, const MPI_Comm comm,
        const std::vector<int> &list_gid, std::unordered_map<int, std::vector<int> > &found_lids) {

    int my_rank = 0;
    const Epetra_Import *importer = Matrix->Importer();
    bool print = false;

    MPI_Comm_rank(comm, &my_rank);

    if (importer != nullptr) {
        int number_exp_rows = importer->NumExportIDs();
        std::vector<int> exp_procs_v(number_exp_rows);
        int *exp_procs = importer->ExportPIDs();
        int my_ngb = CountMyNeighbors(comm);
        std::unordered_map<int, std::vector<int> > local_list_gid;          // this map will store gids received and detected on a process

        for(int n = 0; n < number_exp_rows; ++n) {
            exp_procs_v[n] = exp_procs[n];
            std::cout << exp_procs[n] << " - ";
        }
        std::cout << "\n";
        sort( exp_procs_v.begin(), exp_procs_v.end() );
        exp_procs_v.erase( std::unique( exp_procs_v.begin(), exp_procs_v.end() ), exp_procs_v.end() );
        for(uint32_t n = 0; n < exp_procs_v.size(); ++n) {
            std::cout << exp_procs_v[n] << " . ";
        }
        std::cout << "\n";

        /* Send locally stored GIDs to export PIDs */
        int tag = 0;
        for(size_t n = 0; n < exp_procs_v.size(); ++n) {
            if (!list_gid.empty())
                MPI_Send(list_gid.data(), list_gid.size(), MPI_INT, exp_procs_v[n], tag, comm);
        }
        MPI_Barrier(comm);

        /* Check if process should receive any message */
        for(int n = 0; n < my_ngb; ++n) {
//        for(size_t n = 0; n < exp_procs_v.size(); ++n) {
            int flag;
            MPI_Status status;
            MPI_Iprobe(MPI_ANY_SOURCE, tag, comm, &flag, &status);
            if (flag == true) {
                /* If so - receive the message and put it into the map */
                int sender_id = status.MPI_SOURCE;

                if (print)
                    std::cout << "1. PID " << my_rank << " has received a message from PID " << sender_id << "\n";

                int msg_size = 0;
                MPI_Get_count(&status, MPI_INT, &msg_size);
                local_list_gid[sender_id].resize(msg_size);
                MPI_Recv(local_list_gid[sender_id].data(), msg_size, MPI_INT, sender_id, tag, comm, &status);

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
            MPI_Send(&num_loc_points, 1, MPI_INT, it_i->first, tag, comm);           // <--- change PIDList[0]!!!
        }
        MPI_Barrier(comm);

        for(int n = 0; n < my_ngb; ++n) {
            int flag;
            MPI_Status status;
            MPI_Iprobe(MPI_ANY_SOURCE, tag, comm, &flag, &status);

            if (flag == true) {
                /* If so - receive the message and put it into the map */
                int sender_id = status.MPI_SOURCE;

                if (print)
                    std::cout << "2. PID " << my_rank << " has received a message from PID " << sender_id << "\n";

                int num_remote_points = 0;
                MPI_Recv(&num_remote_points, 1, MPI_INT, sender_id, tag, comm, &status);

                local_data[sender_id].resize(num_remote_points);
            }
            else {
                if (print)
                    std::cout << "2. PID " << my_rank << " has not received any message\n";
            }
        }

        MPI_Barrier(comm);

        /* Once finished - fill local data with zeros. It will be rewritten during the real communication */
        std::unordered_map<int, std::vector<double> >::iterator it_d, it_d_end;
        it_d = local_data.begin();
        it_d_end = local_data.end();
        for(; it_d != it_d_end; ++it_d)
            memset(it_d->second.data(), 0x00, sizeof(double) * it_d->second.size());

        MPI_Barrier(comm);

        SendMapToNeighbor(comm, local_list_gid, &found_lids);
    }

    MakePointersOnDiagonal(Matrix, comm);
}

void MatrixDiagonal::SendMapToNeighbor(const MPI_Comm comm, const std::unordered_map<int, std::vector<int> > &sent_map,
            std::unordered_map<int, std::vector<int> > *received_map) {

    std::unordered_map<int, std::vector<int> >::const_iterator it_beg_s, it_end_s;
    std::unordered_map<int, std::vector<int> >::iterator it_beg_r, it_end_r;
    std::unordered_map<int, std::vector<double> >::iterator it_beg_d, it_end_d;

    MPI_Request send_request[sent_map.size()];
    MPI_Request recv_request[received_map->size()];
    MPI_Status send_status[sent_map.size()];
    MPI_Status recv_status[sent_map.size()];

    /* Preallocate memory for received map (note, it is of the same size as "local_data") */
    it_beg_d = local_data.begin();
    it_end_d = local_data.end();
    for(; it_beg_d != it_end_d; ++it_beg_d) {
        const int loc_pid = it_beg_d->first;
        const int size = it_beg_d->second.size();
        received_map->operator[](loc_pid).resize(size);
    }

    int counter = 0;
    int tag = 0;

    it_beg_s = sent_map.begin();
    it_end_s = sent_map.end();
    for(; it_beg_s != it_end_s; ++it_beg_s) {
        const int loc_pid = it_beg_s->first;
        MPI_Isend(it_beg_s->second.data(), it_beg_s->second.size(), MPI_INT,
                loc_pid, tag, comm, &send_request[counter]);
        ++counter;
    }

    it_beg_r = received_map->begin();
    it_end_r = received_map->end();
    counter = 0;
    for(; it_beg_r != it_end_r; ++it_beg_r) {
        const int loc_pid = it_beg_r->first;
        MPI_Irecv(it_beg_r->second.data(), it_beg_r->second.size(), MPI_INT,
                loc_pid, tag, comm, &recv_request[counter]);
        ++counter;
    }

    int my_rank;
    MPI_Comm_rank(comm, &my_rank);

    MPI_Waitall(sent_map.size(), send_request, send_status);
    MPI_Waitall(received_map->size(), recv_request, recv_status);

    it_beg_r = received_map->begin();
    it_end_r = received_map->end();
    for(; it_beg_r != it_end_r; ++it_beg_r) {
        std::cout << "(" << my_rank << ") " << it_beg_r->first << " : ";
        for(uint32_t n = 0; n < it_beg_r->second.size(); ++n) {
            std::cout << it_beg_r->second[n] << " ";
        }
        std::cout << "\n";
    }
}

void MatrixDiagonal::MakePointersOnDiagonal(Epetra_CrsMatrix *Matrix, const MPI_Comm comm) {

    if (local_ids.empty())
        return;

    int my_rank = 0;

    MPI_Comm_rank(comm, &my_rank);

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

void MatrixDiagonal::ReceiveDiagonalCoefficients(const MPI_Comm comm,
        std::unordered_map<int, std::vector<double> > &rcv_data) {

    int my_rank = 0;
    MPI_Comm_rank(comm, &my_rank);
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
        MPI_Isend(data.data(), size, MPI_DOUBLE, pid, tag, comm, &request_snd[counter]);
        ++counter;
    }

    std::vector<double> rcv(2);
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
        std::cout << "pid: " << my_rank << "<-" << pid << " " << it_d->second.size() << "\n";
        // Allocate memory only if incoming map is empty
        if (!allocated) {
            rcv_data[pid].resize(local_data[pid].size());
        }
        prt_out = &rcv_data[pid];
        MPI_Irecv(prt_out->data(), prt_out->size(), MPI_DOUBLE, pid, tag, comm, &request_rcv[counter]);
        ++counter;
    }


//    int counter = 0;
////    std::vector<double> data(2);
////    std::vector<double> rcv(2);
//    double data, rcv;
//    data = 2;
//    rcv = 0;
//    MPI_Request r;
//    MPI_Status st;
////    const void *buf, int count, MPI_Datatype datatype, int dest,
////                                 int tag, MPI_Comm comm, MPI_Request *request
//
////    const void *buf, int count, MPI_Datatype datatype, int dest,
////                                int tag, MPI_Comm comm
//    if (my_rank != 0)
//        MPI_Irecv(&rcv,  1, MPI_DOUBLE, 0, 1, comm, &r);
//    else
//        MPI_Isend(&data, 1, MPI_DOUBLE, 1, 1, comm, &r);
//    counter++;
//
//    MPI_Barrier(comm);
////    MPI_Wait(&r, &st);


    MPI_Waitall(local_ptrs.size(), request_snd, status_snd);
    MPI_Waitall(local_data.size(), request_rcv, status_rcv);
//    if (my_rank == 1)
//        std::cout << my_rank << " : " << rcv << "\n";
//    else
//        std::cout << my_rank << " : " << data << "\n";
}

void MatrixDiagonal::UpdateDiagonalCoefficients(const MPI_Comm comm) {

    ReceiveDiagonalCoefficients(comm, local_data);
//    int my_rank = 0;
//    MPI_Comm_rank(comm, &my_rank);
//    std::unordered_map<int, std::vector<double*> >::iterator it_dp, it_dp_end;
//
//    it_dp = local_ptrs.begin();
//    it_dp_end = local_ptrs.end();
//
//    int neighbors = local_ptrs.size();
//    MPI_Request request_snd[neighbors];
//    MPI_Status status_snd[neighbors];
//    int tag = 0;
//    int counter = 0;
//    for(; it_dp != it_dp_end; ++it_dp) {
//        int pid = it_dp->first;
//        uint32_t size = it_dp->second.size();
//        std::vector<double> data(size);
//        std::cout << my_rank << "(" << pid << ") Send : " << size << " : ";
//        for(uint32_t n = 0; n < size; ++n) {
//            data[n] = *it_dp->second[n];
//            std::cout << data[n] << " ";
//        }
//        std::cout << std::endl;
//
//        MPI_Isend(data.data(), size, MPI_DOUBLE, pid, tag, comm, &request_snd[counter]);
//        ++counter;
//    }
//
//    neighbors = local_data.size();
//    MPI_Request request_rcv[neighbors];
//    MPI_Status status_rcv[neighbors];
//    std::unordered_map<int, std::vector<double> >::iterator it_d, it_d_end;
//    it_d = local_data.begin();
//    it_d_end = local_data.end();
//    counter = 0;
//    for(; it_d != it_d_end; ++it_d) {
//        int pid = it_d->first;
//        std::vector<double> *prt_out;
////        prt_out = &local_data[pid];
//        prt_out = &it_d->second;
//        MPI_Irecv(prt_out->data(), prt_out->size(), MPI_DOUBLE, pid, tag, comm, &request_rcv[counter]);
//        ++counter;
//    }
//
//    MPI_Waitall(local_ptrs.size(), request_snd, status_snd);
//    MPI_Waitall(local_data.size(), request_rcv, status_rcv);
//
//    if (my_rank == 1)
//        std::cout << "prt_out->size(): " << local_data[0][0] << " " << local_data[0][1] << "\n";
}

/* ********************************************************* */
/* ********************************************************* */
/* ********************************************************* */
/* ********************************************************* */
/* ********************************************************* */
/* ********************************************************* */


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
        _nx = 21;
        _ny = 21;
        _nz = 81;
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

  Epetra_Map *myMap = nullptr;  // if create do not forget do delete!
//  StripeDecomposition(myMap, _nx, _ny, _nz, NumGlobalElements, comm);
//  Decomposition2(myMap, NumGlobalElements, comm);

  //  cout << *myMap << endl;

//  Epetra_CrsMatrix A(Copy, *myMap, dimension+1, false);

    /*
     * Lowlevel matrix assembly (row-by-row from left to right)
     */
    Epetra_CrsMatrix *A;
    A = new Epetra_CrsMatrix(Copy, Map, dimension+1, false);

//    std::cout << *myMap << "\n";
    time1 = time.WallTime();
    AssembleMatrixGlob(dimension, &Map, _nx, _ny, _nz, A);
    time2 = time.WallTime();
    MPI_Reduce(&time1, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, comm.Comm());
    MPI_Reduce(&time2, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm.Comm());
    if (myRank == 0) {
        full = max_time - min_time;
//        std::cout << "Assembly time: " << full << std::endl;
    }

    int grid_cell_id_p = 1;
    double *values;
    int *indices;
    int num_elts;
    double coeff_id_p = 1.;

    // Find the diagonal element
    A->ExtractMyRowView(grid_cell_id_p, num_elts, values, indices);
    for(int n = 0; n < num_elts; ++n) {
        int ind = *(indices + n);
        if (ind == grid_cell_id_p) {
            coeff_id_p = *(values + n);
            std::cout << "ind " << ind << "\n";
            break;
        }
    }

    std::cout << "(" << myRank << ") : " << coeff_id_p << "\n";

//    std::cout << A->Map() << "\n";
//    std::cout << *A << std::endl;

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

    /* **************************** */
    /* **************************** */
    /* **************************** */

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
//
//    // Create Linear Problem
//    Epetra_LinearProblem problem;
//
//    problem.SetOperator(A);
//    problem.SetLHS(&x);
//    problem.SetRHS(&b);
//
//    // Create AztecOO instance
////#define AZ_r0               0 /* ||r||_2 / ||r^{(0)}||_2                      */
////#define AZ_rhs              1 /* ||r||_2 / ||b||_2                            */
////#define AZ_Anorm            2 /* ||r||_2 / ||A||_infty                        */
////#define AZ_sol              3 /* ||r||_infty/(||A||_infty ||x||_1+||b||_infty)*/
////#define AZ_weighted         4 /* ||r||_WRMS                                   */
////#define AZ_expected_values  5 /* ||r||_WRMS with weights taken as |A||x0|     */
////#define AZ_noscaled         6 /* ||r||_2                                      */
////#define AZTECOO_conv_test   7 /* Convergence test will be done via AztecOO    */
////#define AZ_inf_noscaled     8 /* ||r||_infty
//
//    AztecOO solver(problem);
//    solver.SetAztecOption(AZ_conv, AZ_noscaled);
//    solver.SetAztecOption(AZ_solver, AZ_bicgstab);
//    solver.SetAztecOption(AZ_precond, AZ_none);
//    solver.SetAztecOption(AZ_output, 1);
//    solver.Iterate(100, 1.0E-8);
//
////    // Create AztecOO instance
////    AztecOO solver(problem);
////
//////    ML_Epetra::MultiLevelPreconditioner MLPrec2(*A, MLList);
////
////    double time5 = time.WallTime();
////    solver.SetPrecOperator(MLPrec);
////    double time6 = time.WallTime();
////    MPI_Reduce(&time5, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, comm.Comm());
////    MPI_Reduce(&time6, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm.Comm());
////    if (myRank == 0) {
////        full = max_time - min_time;
////        std::cout << "SetPrecOperator() time: " << "\t" << full << std::endl;
////    }
////    solver.SetAztecOption(AZ_conv, AZ_noscaled);
////    solver.SetAztecOption(AZ_solver, AZ_bicgstab);
////    solver.SetAztecOption(AZ_output, 1);
//////    solver.SetAztecOption(AZ_precond, AZ_dom_decomp);//AZ_ilu);//AZ_dom_decomp);
//////    solver.SetAztecOption(AZ_subdomain_solve, AZ_icc);
////    solver.SetAztecOption(AZ_precond, AZ_Jacobi);
////    solver.SetAztecOption(AZ_omega, 0.72);
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
//
//    if (myRank == 0) {
//        full = max_time - min_time;
//        std::cout << "Total time: (" << numProcs << ") " << full << std::endl;
//        std::cout << "Iterations: " << solver.NumIters() << "\n";
//        std::cout << "Residual: " << solver.TrueResidual() << "\n";
//        std::cout << "Residual: " << solver.ScaledResidual() << "\n";
//    }

    /* **************************** */
    /* **************************** */
    /* **************************** */

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

    /*MatrixDiagonal md;
    bool sendig_proc = false;
    int NumIDs = 2;
    std::vector<int> list_gid;

     1) Get a GID
    if (myRank == 0) {
        list_gid.resize(NumIDs);
        list_gid[0] = 8;
        list_gid[1] = 9;
        sendig_proc = true;
    }
    else if (myRank == 1) {
        list_gid.resize(NumIDs);
        list_gid[0] = 6;
        list_gid[1] = 7;
        sendig_proc = true;
    }
    std::unordered_map<int, std::vector<int> > found_lids;
    md.RegisterDiagonal(A, comm.Comm(), list_gid, found_lids);

    std::unordered_map<int, std::vector<double> > rcv_data;
    std::unordered_map<int, std::vector<double> >::iterator it;

    md.UpdateDiagonalCoefficients(comm.Comm());


    std::unordered_map<int, std::vector<double> > *ptr = md.GetLocalDataMap();
    it = ptr->begin();
    std::cout << "(" << myRank << ") Rcv buf : ";
    for(; it != ptr->end(); ++it) {
        for(uint32_t n = 0; n < it->second.size(); ++n) {
            std::cout << it->second[n] << "(" << it->first << ") ";
        }
        std::cout << "\n              ";
    }
    std::cout << std::endl;*/

//    md.ReceiveDiagonalCoefficients(&comm, rcv_data);
//    md.ReceiveDiagonalCoefficients(&comm, rcv_data);
//
//    std::cout << "(" << myRank << ") Rcv buf : ";
//    for(it = rcv_data.begin(); it != rcv_data.end(); ++it) {
//        for(uint32_t n = 0; n < it->second.size(); ++n) {
//            std::cout << it->second[n] << "(" << it->first << ") ";
//        }
//        std::cout << "\n              ";
//    }
//    std::cout << std::endl;


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

    delete [] MyGlobalElements;
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

    delete [] MyGlobalElements;
    return 0;
}


int Decomposition2(Epetra_Map *&Map, int _size, const Epetra_MpiComm &comm) {

    int my_pid = comm.MyPID();
    int MyElements = 0;
    int *MyGlobalElements;
    switch(my_pid) {
        case 0:
            MyElements = 8;
            MyGlobalElements = new int[MyElements];
            MyGlobalElements[0] = 0;
            MyGlobalElements[1] = 1;

            MyGlobalElements[2] = 5;
            MyGlobalElements[3] = 6;

            MyGlobalElements[4] = 10;
            MyGlobalElements[5] = 11;

            MyGlobalElements[6] = 15;
            MyGlobalElements[7] = 16;
            break;

        case 1:
            MyElements = 12;
            MyGlobalElements = new int[MyElements];
            MyGlobalElements[0] = 2;
            MyGlobalElements[1] = 3;
            MyGlobalElements[2] = 4;

            MyGlobalElements[3] = 7;
            MyGlobalElements[4] = 8;
            MyGlobalElements[5] = 9;

            MyGlobalElements[6] = 12;
            MyGlobalElements[7] = 13;
            MyGlobalElements[8] = 14;

            MyGlobalElements[9] = 17;
            MyGlobalElements[10] = 18;
            MyGlobalElements[11] = 19;
            break;
    }

    Map = new Epetra_Map(-1, MyElements, MyGlobalElements, 0, comm);

    delete [] MyGlobalElements;
    return 0;
}


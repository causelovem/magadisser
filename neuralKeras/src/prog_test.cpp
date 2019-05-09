#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>
#include <string>

#include <mpi.h>

using namespace std;

string make_proc_file_name (int rank, string str)
{
    //string tr_fold = "/gpfs/data/edu-cmc-sqi16y3-004/kursach/real_prog/trace_folder";

    //string file_name = tr_fold + "/proc_";
    string file_name = str + "/proc_";
    char *str_rank = NULL;

    int i = 10, cnt = 1;
    int tmp_rank = rank;

    while ((tmp_rank % i) != tmp_rank)
    {
        cnt++;
        i *= 10;
    }

    str_rank = new char [cnt + 1];
    str_rank[cnt] = '\0';

    if (rank == 0)
        str_rank[0] = '0';

    while (tmp_rank > 0)
    {
        int tmp = tmp_rank % 10;

        str_rank[cnt - 1] = tmp + '0';
        cnt--;

        tmp_rank /= 10;
    }

    string tmp_str(str_rank);

    file_name += tmp_str;

    delete [] str_rank;

    return file_name;
}

int main(int argc, char *argv[])
{
    /*<trace_folder> <rand_send1> <rand_send2> <rand_send3>*/
    if (argc != 5)
    {
        cerr << ">Unexpected quantity of arguments, check your comand string." << endl;
        return -1;
    }

    string tr_fold(argv[1]);

    double time = clock();
    MPI_Init(&argc, &argv);

    int nProc = 0, myRank = 0;
    MPI_Status status;
    MPI_Request req;
    MPI_Comm_size(MPI_COMM_WORLD, &nProc);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);


    string file_name = make_proc_file_name(myRank, tr_fold);

    ofstream proc_file(file_name.c_str());
    if (proc_file.is_open() == false)
    {
        cerr << "> Can not open proc_file with such name." << endl;
        MPI_Finalize();
        return -1;
    }

    proc_file << myRank << endl;

    double *mas1 = NULL, *mas2 = NULL, *mas3 = NULL, *mas4 = NULL;    
    
    int size = 900000;
    mas1 = new double [size];
    mas2 = new double [size];
    mas3 = new double [size];
    mas4 = new double [size];

    int *rand_rank_send = new int [nProc];
    int *rand_rank_recv = new int [nProc];

    for (int i = 0; i < size; i++)
    {
        mas1[i] = i;
        mas2[i] = i;
    }

    for (int j = 0; j < 3; j++)
    {
        ifstream rand_send(argv[j + 2]);
        if (proc_file.is_open() == false)
        {
            cerr << "> Can not open proc_file with such name." << endl;
            MPI_Finalize();
            return -1;
        }
        for (int i = 0; i < nProc; i++)
        {
            rand_send >> rand_rank_send[i];
            rand_send >> rand_rank_recv[i];
        }
        rand_send.close();

        if (myRank < nProc / 2)
        {   
            proc_file << nProc - 1 - myRank << " " << size << endl;
            MPI_Send(mas1, size, MPI_DOUBLE, nProc - 1 - myRank, 0, MPI_COMM_WORLD);
        }
        else
        {
            MPI_Recv(mas3, size, MPI_DOUBLE, abs(nProc - 1 - myRank), 0, MPI_COMM_WORLD, &status);
        }

        proc_file << rand_rank_send[myRank] << " " << size << endl;
        MPI_Isend(mas2, size, MPI_DOUBLE, rand_rank_send[myRank], 0, MPI_COMM_WORLD, &req);
        MPI_Irecv(mas4, size, MPI_DOUBLE, rand_rank_recv[myRank], 0, MPI_COMM_WORLD, &req);

        for (int i = 0; i < size; i++)
        {
            mas2[i] += mas3[i];
            mas4[i] += mas1[i];
        }
    }

    delete [] mas1;
    delete [] mas2;
    delete [] mas3;
    delete [] mas4;

    delete [] rand_rank_send;
    delete [] rand_rank_recv;

    double fin_time = clock() - time;

    MPI_Reduce(&fin_time, &time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (myRank == 0)
        cout << ">Time of computation = " << time / CLOCKS_PER_SEC << endl;

    proc_file.close();
    MPI_Finalize();
    return 0;
}

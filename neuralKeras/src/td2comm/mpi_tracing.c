#include <mpi.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

static double TIME_START;
static FILE *OUTF = NULL;

double
my_get_time(void)
{
    struct timeval tv;
    if (gettimeofday(&tv, NULL)) {
        fputs("error in gettimeofday()\n", stderr);
        return -1.0;
    }
    return (double) tv.tv_sec + tv.tv_usec * 1e-6;
}

int
MPI_Init(int *argc, char ***argv)
{
    enum { maxlen = 100 };
    int retval, rank;
    char filename[maxlen];
    double local_time_start;
    retval = PMPI_Init(argc, argv);
    PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
    snprintf(filename, maxlen, "%d.txt", rank);
    OUTF = fopen(filename, "w");
    if (!OUTF) {
        fprintf(stderr,
                "mpi_tracing: proc %d: failed to open output file\n",
                rank);
    }
    local_time_start = my_get_time();
    PMPI_Allreduce(&local_time_start, &TIME_START, 1, MPI_DOUBLE,
                   MPI_MIN, MPI_COMM_WORLD);
    /* TIME_START = MPI_Wtime(); */
    return retval;
}

void
log_send(MPI_Comm comm, int count, MPI_Datatype datatype, int dest)
{
    double time;
    int world_dest, msg_size;
    if (!OUTF) {
        return;
    }
    if (comm != MPI_COMM_WORLD) {
        MPI_Group gw, gt;
        PMPI_Comm_group(MPI_COMM_WORLD, &gw);
        PMPI_Comm_group(comm, &gt);
        PMPI_Group_translate_ranks(gt, 1, &dest, gw, &world_dest);
        PMPI_Group_free(&gt);
        PMPI_Group_free(&gw);
    } else {
        world_dest = dest;
    }
    MPI_Type_size(datatype, &msg_size);
    msg_size *= count;
    time = my_get_time() - TIME_START;
    fprintf(OUTF, "%f,%d,%d\n", time, world_dest, msg_size);
}

int
MPI_Send(void *buf,
         int count,
         MPI_Datatype datatype,
         int dest,
         int tag,
         MPI_Comm comm)
{
    log_send(comm, count, datatype, dest);
    return PMPI_Send(buf, count, datatype, dest, tag, comm);
}


int
MPI_Isend(void *buf,
          int count,
          MPI_Datatype datatype,
          int dest,
          int tag,
          MPI_Comm comm,
          MPI_Request *request)
{
    log_send(comm, count, datatype, dest);
    return PMPI_Isend(buf, count, datatype, dest, tag, comm, request);
}

int
MPI_Sendrecv(void *sendbuf,
             int sendcount,
             MPI_Datatype sendtype,
             int dest,
             int sendtag,
             void *recvbuf,
             int recvcount,
             MPI_Datatype recvtype,
             int source,
             int recvtag,
             MPI_Comm comm,
             MPI_Status *status)
{
    log_send(comm, sendcount, sendtype, dest);
    return PMPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag,
                         recvbuf, recvcount, recvtype, source, recvtag,
                         comm, status);
}

int
MPI_Finalize(void)
{
    double time_finish = my_get_time();
    if (OUTF) {
        fprintf(OUTF, "%f\n", time_finish - TIME_START);
        fclose(OUTF);
    }
    return PMPI_Finalize();
}

#include <stdio.h>
#include <mpi.h>
#include <sys/time.h>
#include <unistd.h>

static double TIME_START;
static FILE *OUTF = NULL;

extern void pmpi_init_(int *);
extern void pmpi_send_(void *, int *, int *, int *, int *, int *, int *);
extern void pmpi_isend_(void *, int *, int *, int *, int *, int *,
                       int *, int *);
extern void pmpi_sendrecv_(void *, int *, int *, int *, int *,
                           void *, int *, int *, int *, int *,
                           int *, int *, int *);
extern void pmpi_finalize_(int *);

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

void mpi_init(int *err)
{
    enum { maxlen = 100 };
    int rank;
    char filename[maxlen];
    double local_time_start;
    pmpi_init_(err);
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

void mpi_send(void *buf,
              int *count,
              int *datatype,
              int *dest,
              int *tag,
              int *comm,
              int *ierr)
{
    MPI_Comm c_comm = PMPI_Comm_f2c(*(MPI_Fint *) comm);
    MPI_Datatype dt = PMPI_Type_f2c(*(MPI_Fint *) datatype);
    log_send(c_comm, *(MPI_Fint *)count, dt, *(MPI_Fint *)dest);
    pmpi_send_(buf, count, datatype, dest, tag, comm, ierr);
}

void mpi_isend(void *buf,
               int *count,
               int *datatype,
               int *dest,
               int *tag,
               int *comm,
               int *request,
               int *ierr)
{
    MPI_Comm c_comm = PMPI_Comm_f2c(*(MPI_Fint *) comm);
    MPI_Datatype dt = PMPI_Type_f2c(*(MPI_Fint *) datatype);
    log_send(c_comm, *(MPI_Fint *)count, dt, *(MPI_Fint *)dest);
    pmpi_isend_(buf, count, datatype, dest, tag, comm, request, ierr);
}

void mpi_sendrecv(void *sendbuf,
                  int *sendcount,
                  int *sendtype,
                  int *dest,
                  int *sendtag,
                  void *recvbuf,
                  int *recvcount,
                  int *recvtype,
                  int *source,
                  int *recvtag,
                  int *comm,
                  int *status,
                  int *ierr)
{
    MPI_Comm c_comm = PMPI_Comm_f2c(*(MPI_Fint *) comm);
    MPI_Datatype dt = PMPI_Type_f2c(*(MPI_Fint *) sendtype);
    log_send(c_comm, *(MPI_Fint *)sendcount, dt, *(MPI_Fint *)dest);
    pmpi_sendrecv_(sendbuf, sendcount, sendtype, dest, sendtag,
                   recvbuf, recvcount, recvtype, source, recvtag,
                   comm, status, ierr);
}

void mpi_finalize(int *ierr)
{
    double time_finish = my_get_time();
    if (OUTF) {
        fprintf(OUTF, "%f\n", time_finish - TIME_START);
        fclose(OUTF);
    }
    pmpi_finalize_(ierr);
}

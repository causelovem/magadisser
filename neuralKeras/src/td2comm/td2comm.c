#include <stdio.h>
#include <stdlib.h>

static int
read_args(int argc,
          char **argv,
          unsigned *np,
          char **dirname,
          char **outf_name)
{
    enum { arg0, arg_dir, arg_np, arg_outf, nargs };
    const char usage_str[] = "usage %s: <dir> <np> <outfile>\n";
    if (argc != nargs) {
        fprintf(stderr, usage_str, argv[arg0]);
        return 1;
    }
    *dirname = argv[arg_dir];
    if (sscanf(argv[arg_np], "%u", np) != 1) {
        fprintf(stderr, "bad value 'np'\n");
        fprintf(stderr, usage_str, argv[arg0]);
        return 1;
    }
    *outf_name = argv[arg_outf];
    return 0;
}

static int
trace_to_matrix(const char *dirname, unsigned np, unsigned **matrix)
{
    enum { maxlen = 4096 };
    char filename[maxlen];
    FILE *f;
    unsigned rank;
    *matrix = calloc(np * np, sizeof((*matrix)[0]));
    if (!*matrix) {
        fprintf(stderr, "trace_to_matrix: memory allocation error\n");
        return 1;
    }
    for (rank = 0; rank < np; ++rank) {
        double time;
        unsigned dest = 0, msg_size = 0;
        int r1, r2, r3, r4, r5;
        snprintf(filename, maxlen, "%s/%d.txt", dirname, rank);
        f = fopen(filename, "r");
        if (!f) {
            fprintf(stderr, "failed to open file '%s'\n", filename);
            free(matrix);
            return 1;
        }
        do {
            if (dest < np) {
                (*matrix)[rank * np + dest] += msg_size;
            }
            r1 = fscanf(f, "%lf", &time);
            r2 = fgetc(f);
            r3 = fscanf(f, "%u", &dest);
            r4 = fgetc(f);
            r5 = fscanf(f, "%u", &msg_size);
        } while (r1 == 1 && r2 == ',' && r3 == 1 && r4 == ',' && r5 == 1);
        fclose(f);
    }
    return 0;
}

static void
write_matrix_to_stream(const unsigned *matrix,
                       unsigned msize,
                       FILE *f)
{
    int i, j;
    for (i = 0; i < msize; ++i) {
        if (msize) {
            fprintf(f, "%u", matrix[i * msize]);
        }
        for (j = 1; j < msize; ++j) {
            fprintf(f, " %u", matrix[i * msize + j]);
        }
        fputc('\n', f);
    }
}

static int
write_matrix(const unsigned *matrix, unsigned msize, const char *filename)
{
    FILE *f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "failed to open file '%s'\n", filename);
        return 1;
    }
    write_matrix_to_stream(matrix, msize, f);
    if (fclose(f)) {
        fprintf(stderr, "failed to close file '%s'\n", filename);
        return 1;
    }
    return 0;
}

int
main(int argc, char **argv)
{
    int retval = 1;
    unsigned np = 0, *matrix = NULL;
    char *dirname = NULL, *outf_name = NULL;

    if (read_args(argc, argv, &np, &dirname, &outf_name)) {
        goto finish;
    }

    if (trace_to_matrix(dirname, np, &matrix)) {
        goto finish;
    }

    if (write_matrix(matrix, np, outf_name)) {
        goto finish;
    }
    
    retval = 0;
 finish:
    if (matrix) {
        free(matrix);
    }
    return retval;
}

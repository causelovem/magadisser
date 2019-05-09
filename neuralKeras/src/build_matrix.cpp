#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>

#include <ctime>

using namespace std;

int main(int argc, char *argv[])
{

    /*<matrix_file> <num_of_proc>*/
    /*if (argc != 2)
    {
        cerr << "> Unexpected quantity of arguments, check your comand string:\n./<prog> <matrix_file> <num_of_proc> [mapping_file]" << endl;
        return -1;
    }*/

    ofstream matrix_file(argv[1]);
    if (matrix_file.is_open() == false)
    {
        cerr << "> Can not open matrix with such name." << endl;
        return -1;
    }

    int num_of_proc = atoi(argv[2]);

    std::vector< std::vector<int> > vec(num_of_proc);

    for (int i = 0; i < num_of_proc; i++)
    {
        vec[i].resize(num_of_proc);
        for (int j = 0; j < num_of_proc; j++)
            vec[i][j] = 0;
    }

    for (int i = 3; i < argc; i++)
    {
        ifstream proc_file(argv[i]);

        int proc = 0;
        proc_file >> proc;

        int dest = 0, size = 0;
        while (!proc_file.eof())
        {
            proc_file >> dest;
            if (proc_file.eof())
                break;
            proc_file >> size;

            vec[proc][dest] +=size;
        }

        proc_file.close();
    }

    for (int i = 0; i < num_of_proc; i++)
    {
        for (int j = 0; j < num_of_proc; j++)
            matrix_file << vec[i][j] << " ";

        matrix_file << endl;
    }

    matrix_file.close();
    return 0;
}

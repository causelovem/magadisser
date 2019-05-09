#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>

using namespace std;

int main(int argc, char const *argv[])
{
    if (argc != 3)
    {
        cerr << ">Unexpected quantity of arguments, check your comand string." << endl;
        return -1;
    }

    ofstream matrix_file(argv[1]);
    if (matrix_file.is_open() == false)
    {
        cerr << ">Can not open matrix with such name." << endl;
        return -1;
    }

    srand(time(0));

    int num = atoi(argv[2]);

    int **matix = new int* [num];
    for (int i = 0; i < num; i++)
        matix[i] = new int [num];

    for (int i = 0; i < num; i++)
        for (int j = 0; j < num; j++)
            matix[i][j] = 0;


    /*for (int i = 0; i < num; i++)
        for (int j = i; j < num; j++)
            matix[i][j] = (int)(rand());*/

    /*for (int j = 0; j < num; j++)
        for (int i = j; i < num; i++)
            matix[i][j] = (int)(rand());*/

    /*for (int i = 0; i < num; i++)
    {
        int rnd = (int)(rand());
        for (int j = 0; j < num; j++)
            matix[i][j] = (int)(rand()) % rnd;
    }*/

    // for (int i = 0; i < num; i++)
    //     for (int j = 0; j < num; j++)
    //         matix[i][j] = (int)(rand());

    for (int i = 0; i < num - 1; i++)
    {
        // matix[i][i + 1] = (int)(rand());
        // matix[i + 1][i] = (int)(rand());

        matix[i][num - 2 - i] = (int)(rand());
        matix[num - 1 - i][i + 1] = (int)(rand());
        matix[i][num - 1 - i] = (int)(rand());
    }
    matix[num - 1][0] = (int)(rand());


    //int rnd_num = (int)(rand()) % num;
    // int rnd_num = num;
    // for (int i = 0; i < rnd_num; i++)
    // {
    //     for (int j = 0; j < rnd_num; j++)
    //     {
    //         int rnd_i = (int)(rand()) % num, rnd_j = (int)(rand()) % num;
    //         matix[rnd_i][rnd_j] = 0;
    //     }
    // }


    for (int i = 0; i < num; i++)
        matix[i][i] = 0;

    for (int i = 0; i < num; i++)
    {
        for (int j = 0; j < num; j++)
        {
            matrix_file << matix[i][j] << " ";
        }
        matrix_file << endl;
    }

    for (int i = 0; i < num; i++)
        delete [] matix[i];
    delete [] matix;

    return 0;
}

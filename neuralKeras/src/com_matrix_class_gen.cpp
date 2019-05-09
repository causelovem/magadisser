#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>

using namespace std;

int main(int argc, char const *argv[])
{
    if (argc != 4)
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

    /*for (int i = 0; i < num; i++)
    {
        int rnd = (int)(rand());
        for (int j = 0; j < num; j++)
            matix[i][j] = (int)(rand()) % rnd;
    }*/

    // for (int i = 0; i < num; i++)
    //     for (int j = 0; j < num; j++)
    //         matix[i][j] = (int)(rand());

    int cl = (int)(rand() % 7);

    ofstream cl_file(argv[3]);
    cl_file << cl << endl;
    cl_file.close();

    if (cl == 0)
    {
        /*TRIPLE INVERSE DIAGONAL*/
        for (int i = 0; i < num - 1; i++)
        {
            matix[i][num - 2 - i] = (int)(rand());
            matix[num - 1 - i][i + 1] = (int)(rand());
            matix[i][num - 1 - i] = (int)(rand());
        }
        matix[num - 1][0] = (int)(rand());
    }
    else
    if (cl == 1)
    {
        /*DOUBLE DIRECT DIAGONAL*/
        for (int i = 0; i < num - 1; i++)
        {
            matix[i][i + 1] = (int)(rand());
            matix[i + 1][i] = (int)(rand());
        }
    }
    else
    if (cl == 2)
    {
        /*DOUBLE DIRECT + DOUBLE INVERSE DIAGONALS*/
        for (int i = 0; i < num - 1; i++)
        {
            matix[i][i + 1] = (int)(rand());
            matix[i + 1][i] = (int)(rand());

            matix[i][num - 2 - i] = (int)(rand());
            matix[num - 1 - i][i + 1] = (int)(rand());
        }
    }
    else
    if (cl == 3)
    {
        /*UPPER TIANGLE*/
        for (int i = 0; i < num; i++)
            for (int j = i + 1; j < num; j++)
                matix[i][j] = (int)(rand());
    }
    else
    if (cl == 4)
    {
        /*DOUBLE INVERSE DIAGONALS*/
        for (int i = 0; i < num - 1; i++)
        {
            matix[i][num - 2 - i] = (int)(rand());
            matix[num - 1 - i][i + 1] = (int)(rand());
        }
    }
    else
    if (cl == 5)
    {
        /*LOWWER TIANGLE*/
        for (int i = 1; i < num; i++)
            for (int j = 0; j < i; j++)
                matix[i][j] = (int)(rand());
    }
    else
    if (cl == 6)
    {
        /*RANDOM*/
        for (int i = 0; i < num; i++)
            for (int j = 0; j < num; j++)
                matix[i][j] = (int)(rand());

        int rnd_num = (int)(num / 2) + (int)(rand()) % (int)(num);
        // int rnd_num = num;
        for (int i = 0; i < rnd_num; i++)
        {
            for (int j = 0; j < rnd_num; j++)
            {
                int rnd_i = (int)(rand()) % num, rnd_j = (int)(rand()) % num;
                matix[rnd_i][rnd_j] = 0;
            }
        }
    }

    /*NOISE*/
    int noise_num = (int)(rand()) % (int)(num + 1);
    for (int i = 0; i < noise_num; i++)
    {
        int rnd_i = (int)(rand()) % num, rnd_j = (int)(rand()) % num;
        matix[rnd_i][rnd_j] = (int)(rand());
    }


    // int rnd_num = (int)(rand()) % num;
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

    matrix_file.close();
    for (int i = 0; i < num; i++)
        delete [] matix[i];
    delete [] matix;

    return 0;
}

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
    srand(time(0));

    if (argc != 3)
    {
        cerr << "> Unexpected quantity of arguments, check your comand string:\n./<prog> <matrix_file> <num_of_proc> [mapping_file]" << endl;
        return -1;
    }

    ofstream send_mas(argv[1]);
    if (send_mas.is_open() == false)
    {
        cerr << "> Can not open matrix with such name." << endl;
        return -1;
    }

    int flag = 1;

    int num_of_proc = atoi(argv[2]);

    std::vector< std::vector<int> > vec(num_of_proc);

    for (int i = 0; i < num_of_proc; i++)
    {
        vec[i].resize(2);
    }
    
start:
    for (int i = 0; i < num_of_proc; i++)
    {        
        vec[i][0] = -1;
        vec[i][1] = -1;
    }

    for (int i = 0; i < num_of_proc; i++)
    {
        flag = 1;
        int tmp = (int)rand() % num_of_proc;

        int cnt = num_of_proc;
        while ((flag == 1) && (cnt > 0))
        {
            cnt--;
            if (cnt < 1)
            {
                cerr << "GG" << endl;
                goto start;

            }
            flag = 0;

            for (int j = 0; j < i; j++)
                if (vec[j][0] == tmp)
                {
                    flag = 1;
                    break;
                }

            if (i == tmp)
                flag = 1;

            if (flag == 1)
                tmp = (int)rand() % num_of_proc;
        }

        vec[i][0] = tmp;
        //vec[tmp][1] = i;
    }

    for (int i = 0; i < num_of_proc; i++)
        for (int j = 0; j < num_of_proc; j++)
            if (vec[j][0] == i)
                vec[i][1] = j;

    for (int i = 0; i < num_of_proc; i++)
        send_mas << vec[i][0] << " " << vec[i][1] << endl;

    for (int i = 0; i < num_of_proc; i++)
    {
        if ((vec[i][0] == -1) || (vec[i][1] == -1)
            || (vec[i][0] == i) || ((vec[i][1] == i)))
        {
            cerr << "> Something wrong in \"buld_rand_send\"" << endl;
            send_mas.close();
            return -1;
        }
    }

    send_mas.close();
    return 0;
}

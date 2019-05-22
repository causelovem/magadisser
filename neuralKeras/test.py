from mpi4py import MPI as mpi
from time import sleep

comm = mpi.COMM_WORLD
rank = comm.Get_rank()

# if rank == 1:
#     sleep(2)

for i in range(0, 10):
    if rank == 0:
        data = {'a': i, 'b': 3.14}
        print(data)
        # sleep(1)
        req = comm.isend(data, dest=(rank + 1), tag=0)
        sleep(2)
        # req.wait()
        # print(rank, req.wait())
    elif rank == 1:
        req = comm.recv(source=(rank - 1), tag=0)
        print(rank, req)
        # data = req.wait()
        # while 1:
        #     r = req.test()
        #     if r[0]:
        #         print(r[1])
        #         break
        # print(rank, req.test()[1])

print(rank, 'OK')
comm.Barrier()

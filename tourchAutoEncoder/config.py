import platform as plt


sys = plt.system()
print(sys)

fileDir = '/mnt/ssd1/prog/pdbFiles'
if sys == 'Windows':
    fileDir = 'F:/prog/magadisser/tourchAutoEncoder/data/pdbFilesGdata'

threshold = 7

# pdbFile = True
pdbFile = False

validatePart = 0.3

batchSize = 24

epochsNum = 10

numWorkers = 12
if sys == 'Windows':
    numWorkers = 0

device = 'cpu'
if sys == 'Windows':
    device = 'cuda'

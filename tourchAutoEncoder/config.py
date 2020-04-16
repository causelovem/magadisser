import platform as plt


sys = plt.system()
print(sys)

rawFileDir = '/mnt/ssd2/pdbFiles'
fileDir = '/mnt/hdd1/pdbFilesGdata'
vectorDir = '-'
modelsDir = '/mnt/ssd1/prog/magadisser/tourchAutoEncoder/models'
statDir = '/mnt/ssd1/prog/magadisser/tourchAutoEncoder/stat'
if sys == 'Windows':
    rawFileDir = '-'
    fileDir = 'F:/prog/magadisser/tourchAutoEncoder/data/pdbFilesGdata'
    vectorDir = 'F:/prog/magadisser/tourchAutoEncoder/data/pdbFilesVector'
    modelsDir = 'F:/prog/magadisser/tourchAutoEncoder/models'
    statDir = 'F:/prog/magadisser/tourchAutoEncoder/stat'

threshold = 7

# pdbFile = True
pdbFile = False

validatePart = 0.2

batchSize = 20

epochsNum = 10

numWorkers = 12
if sys == 'Windows':
    numWorkers = 0

device = 'cpu'
if sys == 'Windows':
    device = 'cuda'

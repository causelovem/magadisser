import platform as plt


sys = plt.system()
print(sys)

rawFileDir = '/mnt/ssd2/pdbFiles'
fileDir = '/mnt/hdd1/pdbFilesGdata'
modelsDir = '/mnt/ssd1/prog/magadisser/tourchAutoEncoder/models'
statDir = '/mnt/ssd1/prog/magadisser/tourchAutoEncoder/stat'
if sys == 'Windows':
    fileDir = 'F:/prog/magadisser/tourchAutoEncoder/data/pdbFilesGdata'
    vectorDir = 'F:/prog/magadisser/tourchAutoEncoder/data/pdbFilesVector'
    rawVectorDir = 'F:/prog/magadisser/tourchAutoEncoder/data/pdbFilesVectorRaw'
    attentVectorDir = 'F:/prog/magadisser/tourchAutoEncoder/data/pdbFilesVectorAttent'
    modelsDir = 'F:/prog/magadisser/tourchAutoEncoder/models'
    statDir = 'F:/prog/magadisser/tourchAutoEncoder/stat'

threshold = 6

# pdbFile = True
pdbFile = False

validatePart = 0.3

batchSize = 20

epochsNum = 10

numWorkers = 12
if sys == 'Windows':
    numWorkers = 0

device = 'cpu'
if sys == 'Windows':
    device = 'cuda'

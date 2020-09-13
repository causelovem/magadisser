# конфигурационный файл
# если используется одна операционная система для тестов,
# то можно убать библиотеку platform и ифы

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
    rawAttentVectorDir = 'F:/prog/magadisser/tourchAutoEncoder/data/pdbFilesVectorRawAttent'
    modelsDir = 'F:/prog/magadisser/tourchAutoEncoder/models'
    statDir = 'F:/prog/magadisser/tourchAutoEncoder/stat'

# "расстояние" между атомами
# если меньше или равно, то считается, что между ними есть связь
threshold = 6

# pdb файл или уже готовый граф
# pdbFile = True
pdbFile = False

# процент валидационной выборки
validatePart = 0.3

# размер батча
batchSize = 20

# количество эпох
epochsNum = 10

# количество воркеров для считывания данных
# (не работает для винды)
numWorkers = 12
if sys == 'Windows':
    numWorkers = 0

# на чем обучать
device = 'cpu'
if sys == 'Windows':
    device = 'cuda'

import numpy as np

def dataLoader_cancer(path):
    #path = './dataset/breast-cancer-wisconsin.txt'
    benignData = np.empty(shape = [0, 11])
    malignantData = np.empty(shape = [0, 11])
    benignError = []
    malignantError = []

    with open(path) as dataset:
        for line in dataset.readlines():

            rawStr = line.strip()
            strAfter = rawStr.split('\t')
            rawList = list(strAfter)
            
            if '?' in rawList:
                if rawList[10] == '0':
                    benignError.append(rawList)
                else:
                    malignantError.append(rawList)
            else:
                numberList = list(map(int, rawList))
                if numberList[10] == 0:
                    benignData = np.append(benignData, np.array([numberList]), axis = 0)
                else:
                    malignantData = np.append(malignantData, np.array([numberList]), axis = 0)

    # here we have read the file totally

    benignAver = benignData.mean(axis = 0)
    malignantAver = malignantData.mean(axis = 0)

    # fix the missing data ? with the average value.
     
    for lists in benignError:
        for i in range(0, 11):
            if lists[i] == '?':
                lists[i] = str(benignAver[i])
            lists[i] = float(lists[i])
        benignData = np.append(benignData, np.array([lists]), axis = 0)

    for lists in malignantError:
        for i in range(0, 11):
            if lists[i] == '?':
                lists[i] = str(malignantAver[i])
            lists[i] = float(lists[i])
        malignantData = np.append(malignantData, np.array([lists]), axis = 0)

    all = np.append(benignData, malignantData, axis=0)

    X = all[:, 1 : 10]
    y = all[:, 10]

    return X, y


def dataLoader_diabetes(path):
    #path = './dataset/diabetes.csv'

    data = np.empty(shape = [0, 9])

    with open(path) as dataset:
        num = 0
        for i in dataset:
            if num == 0:
                num = 1
            else:
                strAfter = i.split(',')
                number = list(map(float, strAfter))
                data = np.append(data, np.array([number]), axis = 0)

    X = data[:, 0 : 8]
    y = data[:, 8]

    return X, y

# 60 + 1
def dataLoader_sonar(path):
    #path = './dataset/sonar_csv.csv'

    data = np.empty(shape = [0, 61])
    with open(path) as dataset:
        num = 0
        for i in dataset:
            if num == 0:
                num = 1
            else:
                strAfter = i.split(',')
                if strAfter[60] == 'Rock\n':
                    strAfter[60] = '1'
                else:
                    strAfter[60] = '0'
                number = list(map(float, strAfter))
                data = np.append(data, np.array([number]), axis = 0)
    X = data[:, 0 : 60]
    y = data[:, 60]

    return X, y


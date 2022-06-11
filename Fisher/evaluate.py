import argparse
import numpy as np
import cv2
import os

class FisherFace(object):
    def __init__(self, k, dsize):
        self.k = k  # 指定投影w的个数
        self.dsize = dsize  # 统一尺寸大小

    def loadImg(self, fileName, dsize):
        img = cv2.imread(fileName)
        retImg = cv2.resize(img, dsize)
        retImg = cv2.cvtColor(retImg, cv2.COLOR_RGB2GRAY)
        retImg = cv2.equalizeHist(retImg)
        return retImg

    def createImgMat(self, dirName):
        dataMat = np.zeros((10, 1))
        label = []
        dataList = []
        for parent, dirnames, filenames in os.walk(dirName):
            for dirname in dirnames:
                for subParent, subDirName, subFilenames in os.walk(parent + '/' + dirname):
                    for index, filename in enumerate(subFilenames):
                        img = self.loadImg(subParent + '/' + filename, self.dsize)
                        tempImg = np.reshape(img, (-1, 1))
                        if index == 0:
                            dataMat = tempImg
                        else:
                            dataMat = np.column_stack((dataMat, tempImg))
                        label.append(subParent)
                dataList.append(dataMat)

        return dataList, label

    def LDA(self, dataList, k):
        n = dataList[0].shape[0]
        W = np.zeros((n, self.k))
        Sw = np.zeros((n, n))
        Sb = np.zeros((n, n))
        u = np.zeros((n, 1))
        N = 0
        meanList = []
        sampleNum = []

        for dataMat in dataList:
            meanMat = np.mat(np.mean(dataMat, 1)).T
            meanList.append(meanMat)
            sampleNum.append(dataMat.shape[1])

            dataMat = dataMat - meanMat
            sw = dataMat * dataMat.T
            Sw += sw

        for index, meanMat in enumerate(meanList):
            m = sampleNum[index]
            u += m * meanMat
            N += m
        u = u / N

        for index, meanMat in enumerate(meanList):
            m = sampleNum[index]
            sb = m * (meanMat - u) * (meanMat - u).T
            Sb += sb


        eigVals, eigVects = np.linalg.eig(np.mat(np.linalg.inv(Sw) * Sb))
        eigValInd = np.argsort(eigVals)
        eigValInd = eigValInd[::-1]
        eigValInd = eigValInd[:k]  # 取出指定个数的前k大的特征值

        eigVects = eigVects / np.linalg.norm(eigVects, axis=0)  # 归一化特征向量
        redEigVects = eigVects[:, eigValInd]


        transMatList = []
        for dataMat in dataList:
            transMatList.append(redEigVects.T * dataMat)
        return transMatList, redEigVects

    def train(self, dirName):
        dataMat, label = self.createImgMat(dirName)
        transMatList, redVects = fisherface.LDA(dataMat, self.k)
        return transMatList, redVects, label

    def compare(self, redVects, transMatList, testImg, label):
        testImg = cv2.resize(testImg, self.dsize)
        testImg = cv2.cvtColor(testImg, cv2.COLOR_RGB2GRAY)
        testImg = np.reshape(testImg, (-1, 1))
        testImg = np.mat(redVects).T * testImg

        disList = []
        testVec = np.reshape(testImg, (1, -1))
        for transMat in transMatList:
            for sample in transMat.T:
                disList.append(np.linalg.norm(testVec - sample))

        sortIndex = np.argsort(disList)
        return label[sortIndex[0]]

    def predict(self, transMatList, redVects, testFileName, label):
        testImg = cv2.imread(testFileName)

        ans = self.compare(redVects, transMatList, testImg, label)
        return ans


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fisher Face train.')
    parser.add_argument('--param_save_path', type=str, default="./param",
                        help='The train parameter save path.')
    parser.add_argument('--valid_data_path', type=str, default="./faces_test",
                        help='The path of the valid data')
    opt = parser.parse_args()
    print(opt)

    transMatList_save_path = os.path.join(opt.param_save_path, "transMatList")
    assert os.path.exists(transMatList_save_path)
    transMatPathList = os.listdir(transMatList_save_path)
    transMatList = []
    for i in range(len(transMatPathList)):
        t = np.load(os.path.join(transMatList_save_path, "{}.npy".format(i)))
        transMatList.append(t)
    redVects = np.load(os.path.join(opt.param_save_path, "redVects.npy"))
    label = np.load(os.path.join(opt.param_save_path, "label.npy"))
    label = label.tolist()
    with open(os.path.join(opt.param_save_path, "k.txt"), "r") as f:
        k = int(f.readline())
        dsize = int(f.readline())
    print("k=", k, "  dsize=", dsize)

    fisherface = FisherFace(k, (dsize, dsize))


    test_path = opt.valid_data_path
    path_ = os.listdir(test_path)
    truth = 0
    sample = 0
    for p in path_:
        sample += 1
        tt = os.path.join(test_path, p)
        t = os.listdir(tt)[0]
        pp = os.path.join(tt, t)
        ans = fisherface.predict(transMatList, redVects, pp, label)

        if p == ans.split("/")[-1]:
            truth += 1

    acc = truth / sample
    print("acc = ", acc)



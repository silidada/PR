# encoding=utf-8
import numpy as np
import cv2
import os


class FisherFace(object):
    def __init__(self, threshold, k, dsize):
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

    def compare(self, dataList, testImg, label):
        testImg = cv2.resize(testImg, self.dsize)
        testImg = cv2.cvtColor(testImg, cv2.COLOR_RGB2GRAY)
        testImg = np.reshape(testImg, (-1, 1))

        disList = []
        testVec = np.reshape(testImg, (-1,1))
        for d in dataList:
            dist = d - testVec
            dist = np.sum(dist*dist, axis=0)
            for i in range(dist.shape[0]):
                disList.append(dist[i])

        sortIndex = np.argsort(disList)
        return label[sortIndex[0]]

    def predict(self, dirName, testFileName):
        testImg = cv2.imread(testFileName)
        dataMat, label = self.createImgMat(dirName)
        ans = self.compare(dataMat, testImg, label)
        return ans


if __name__ == "__main__":
    fisherface = FisherFace(10, 20, (20, 20))
    test_path = "./faces_test"
    path_ = os.listdir(test_path)
    truth = 0
    sample = 0
    for p in path_:
        sample += 1
        tt = os.path.join(test_path, p)
        t = os.listdir(tt)[0]
        pp = os.path.join(tt ,t)
        ans = fisherface.predict('./faces', pp)

        print(p, ans, "   ", end="")
        if p == ans.split("/")[-1]:
            truth += 1
            print("yes")
        else:
            print("no")


    acc = truth / sample
    print("acc = ", acc)


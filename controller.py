import cPickle, os, random
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, ndimage
from sklearn import linear_model
from sklearn.model_selection import StratifiedShuffleSplit

def loadMaskWise(imageNameList,path):
	imageAndMask = []
	for e in imageNameList:
		if "-mask" not in e:
			imageAndMask.append([ndimage.imread(path+"/"+e, mode="RGB"),ndimage.imread(path+"/"+e.replace('.','-mask.'), mode="RGB")])
	return imageAndMask

def returnImageList():
	return [filename for root, dirnames, filenames in os.walk("SigTuple_data/Train_Data") for filename in filenames]

def dumpAsPickle(filename, toDump):
	with open("dumps/"+filename+".pkl", 'wb') as fid:
		cPickle.dump(toDump, fid)

def loadFromPickle(filename):
	with open("dumps/"+filename+".pkl", 'r') as fid:
		return cPickle.load(fid)

def findIndices(elementTofind,array):
	return [[r,c] for r,j in enumerate(array) for c,k in enumerate(j) if np.all(k == elementTofind)]

def differenceOfHistograms(image_array,wbc_points,non_wbc_points):
	kickedOut_colors = []
	differ = []
	for each_wbc_point in wbc_points:
		for each_non_wbc_point in non_wbc_points:
			if euclideanDistance(each_wbc_point,each_non_wbc_point):
				kickedOut_colors.append(each_wbc_point)
				break
	kickedOut_colors = np.array(kickedOut_colors)
	print len(wbc_points),len(kickedOut_colors)
	# differ.append(euclideanDistance(each_wbc_point,each_non_wbc_point))
	# differ = -np.sort(-np.array(differ))
	# print differ[0],differ[20000]
	# plt.plot(range(0,len(differ)),differ)
	# plt.show()
	exit()
	for x in wbc_points:
		if x not in kickedOut_colors:
			print x

def euclideanDistance(p1,p2):
	if np.linalg.norm(p1-p2) > 20:
		return True

def pixelClassification():
	wbc = loadFromPickle("wbcpixeldata")
	nonwbc = loadFromPickle("nonwbcdata")
	dependents = np.zeros(len(wbc)+2*len(wbc))
	dependents[0:len(wbc)] = 1
	independent = np.array(list(wbc)+list(nonwbc)[0:2*len(wbc)])
	sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
	for train_index, test_index in sss.split(independent, dependents):
		X_train, X_test = independent[train_index], independent[test_index]
		y_train, y_test = dependents[train_index], dependents[test_index]
		logreg = linear_model.LogisticRegression(C=1e5)
		logreg.fit(X_train,y_train)
		res = logreg.predict(X_test)
	dumpAsPickle("logisticClassifier",logreg)
		# (n,d)=(0,0)
		# for i,r in enumerate(res):
		# 	if r == 1 and r == y_test[i]:
		# 		n+=1
		# 	if r == 1:
		# 		d+=1
		# print n/float(d)


if __name__ == '__main__':
	# imageArrays = loadMaskWise(returnImageList(),"SigTuple_data/Train_Data")
	# dumpAsPickle("traindata",imageArrays)
	traindata = loadFromPickle("traindata")
	# wbc_points = []
	# non_wbc_points = []
	# for eachImageData in traindata:
	# 	indices = findIndices([1,1,1],eachImageData[1])
	# 	wbc_points = [eachImageData[0][index[0],index[1]] for index in indices]
	# 	non_wbc_points = [eachImageData[0][r,c] for r in range(0,np.shape(eachImageData[0])[0]) for c in range(0,np.shape(eachImageData[0])[1]) if [r,c] not in indices]
	# 	differenceOfHistograms(eachImageData[0],wbc_points,non_wbc_points)
	# 	exit()
	# pixelClassification()
	logreg = loadFromPickle("logisticClassifier")
	testImage = ndimage.imread("SigTuple_data/Test_Data/F1BFEA74B33D.jpg", mode="RGB")
	dummy_image = np.zeros(np.shape(testImage))
	for r in range(np.shape(testImage)[0]):
		for c in range(np.shape(testImage)[1]):
			if logreg.predict_proba(testImage[r,c].reshape(1,-1))[0][1] > 0.85:
				print "Here",r,c
				dummy_image[r,c] = [1,1,1]
			else:
				dummy_image[r,c] = [0,0,0]
	plt.imshow(dummy_image)
	plt.show()

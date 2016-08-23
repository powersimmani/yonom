#-*- coding: utf-8 -*-

import datetime
import time
import random
from pprint import pprint 
from sklearn import tree
from sklearn import metrics
from sklearn.externals import joblib
from sklearn import svm
from sklearn import linear_model, decomposition, datasets
import os
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_curve

def change_data():
	mapping = {"서울특별시" : 1, "부산광역시" : 2, "대구광역시" : 3, "인천광역시" : 4, "광주광역시" : 5, "대전광역시" : 6, "울산광역시" : 7, "세종특별자치시" : 8, "경기도" : 9, "경기남부" : 10, "경기북부" : 11, "강원도" : 12, "충청북도" : 13, "충청남도" : 14, "전라북도" : 15, "전라남도" : 16, "경상북도" : 17, "경상남도" : 18, "제주특별자치도" : 19, "":20}

	output =  open("./data/utf_8.data.csv","w")
	for line in open("./data/utf_8.newfile.csv").readlines():
		parse = line.split(",")
		if parse[1] == "c_0":
			continue

		#시/도 변환
		parse[15] = mapping[parse[15]]
		
		#시간 -> 시간대로 변경
		x = time.strptime(parse[2][11:].split(',')[0],'%H:%M:%S')
		t = datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()
		parse[2] = int(t/3600)

		new_line = parse[1:4] + parse[6:16]
		for item in new_line:
			output.write(str(item) +",")
		output.write("\n")

def divide_users():
	file_io = {}
	for line in open("./data/utf_8.data.csv","r"):
		parse = line.split(",")
		if (parse[0] not in file_io):
			file_io[parse[0]] = open("./data/file_per_car/"+str(id_mapping[parse[0]])+".csv","w")
		file_io[parse[0]].write(line)


def cal_freq_feature(freq_list):	
	import math
	import pywt
	freq_list = [float(i) for i in freq_list]
	
	FFT = np.fft.fft(freq_list)
	
	FFT_r =np.real(FFT)
	FFT_i =np.imag(FFT)
	FFT_a =np.angle(FFT)
	
	DWT_at, DWT_dt = pywt.dwt(freq_list, 'db1')
	mul = 1
	if (len(FFT) % 2 ==0):
		mul = len(DWT_at)*2
	else:
		mul = len(DWT_at)*2-1
	DWT_a = [0.0]*mul
	DWT_d = [0.0]*mul
	for i in range(0,mul):
		DWT_a[i] = DWT_at[i/2]
		DWT_d[i] = DWT_dt[i/2]
	
	ave = {1:[],5:[],9:[],13:[],17:[]}
	delta = {1:[],5:[],9:[],13:[],17:[]}

	for i in [1,5,9,13,17]:
		for ite in range(0,len(freq_list)):
			start_add = ite-i
			if start_add < 0:
				start_add = 0
			delta[i].append(freq_list[ite] - freq_list[start_add])
			ave[i].append(sum(freq_list[start_add:ite]) / max(ite-start_add,1))
	ret = []
	add = 0	

	for i in range(0,len(freq_list)):
		atom = [0.0]*5
		
		atom[0] = FFT_a[i]
		atom[1] = FFT_r[i]
		atom[2] = FFT_i[i]
		atom[3] = DWT_a[i]
		atom[4] = DWT_d[i]
		atom[5] = ave[1][i]
		atom[6] = ave[5][i]
		atom[7] = ave[9][i]
		atom[8] = ave[13][i]
		atom[9] = ave[17][i]
		atom[10] = delta[1][i]
		atom[11] = delta[5][i]
		atom[12] = delta[9][i]
		atom[13] = delta[13][i]
		atom[14] = delta[17][i]
		"""
		atom[0] = ave[1][i]
		atom[1] = ave[5][i]
		atom[2] = ave[9][i]
		atom[3] = ave[13][i]
		atom[4] = ave[17][i]
		atom[5] = delta[1][i]
		atom[6] = delta[5][i]
		atom[7] = delta[9][i]
		atom[8] = delta[13][i]
		atom[9] = delta[17][i]
		"""
		ret.append(atom)
	return ret


def add_features(car_no):
	
	#np.fft.fftn(a, s=None, axes=None, norm=None)[source]
	lines = open("./data/file_per_car/"+str(car_no)+".csv").readlines()
	
	col_data = [3,4,5,7,9,10]
	
	data_sheet = []
	for line in lines:
		parse = line.split(",")
		parse[-1] = parse[-1].strip("\n")
		data_sheet.append(parse[:-1])

	for add in col_data:
		freq_list = [i[add] for i in data_sheet] 
		freq_feature = cal_freq_feature(freq_list)			
		for ite in range(0,len(data_sheet)):
			data_sheet[ite] = data_sheet[ite] + freq_feature[ite]

	f_out = open("./data/add_features/"+str(car_no)+".csv","w")

	for line in data_sheet:
		f_out.write(str(line[0]))
		for item in line[1:]:
			f_out.write("," + str(item))
		f_out.write("\n")


def make_data_set():

	past_time = time.time()

	path_in  =	"./data/add_features/"
	path_out  =	"./data/train_test_set/"

	for i in range(1,2):
		true_val = []
		false_val = []
		out_train = open(path_out+ str(i)+"_train.csv","w")
		out_test = open(path_out+ str(i)+"_test.csv","w")
		#scale 된 데이터 
		#out_train = open(path_out+ str(i)+"_train_scale.csv","w")
		#out_test = open(path_out+ str(i)+"_test_scale.csv","w")


		for line in open(path_in + str(i)+".csv"):
			parse =  line[:-1] + ",1\n"
			true_val.append(parse[33:])
			#out.write(parse)

		for j in range(1,31):
			if j == i :
				continue
			for line in open(path_in + str(j)+".csv"):
				parse =  line[:-1] + ",-1\n"
				false_val.append(parse[33:])
				#out.write(parse)

		len_true = len(true_val)
		len_false = len(false_val)

		for line in true_val[len_true*9/10:]:
			out_test.write(line)
		for line in false_val[len_false*9/10:]:
			out_test.write(line)

		for line in true_val[:len_true*9/10]:
			out_train.write(line)
		for line in false_val[:len_false*9/10]:
			out_train.write(line)

		print "train/test data make iteration " + str(i) +" done, takes " + str(time.time() - past_time)
		past_time = time.time()


def new_intergrated_performance(dataset_number):
	data_type = "all"
	print data_type
	in_path = "./data/train_test_set/" + str(dataset_number)
	out_path = "./model/"+ str(dataset_number)
	X_train = []
	y_train = []

	X_test = []
	y_test = []
	#학습 및 진단데이터 입력
	
	print "1"
	for line in open(in_path + "_train_"+data_type+".csv"):
		X_train.append(map(float, line.split(",")[0:-1]))
		y_train.append(line.split(",")[-1].strip("\n"))

	y_train = map(int,y_train)
	print line

	for line in open(in_path + "_test_"+data_type+".csv"):
		X_test.append(map(float, line.split(",")[0:-1]))
		y_test.append(line.split(",")[-1].strip("\n"))

	y_test = map(int,y_test)

	#X_train = preprocessing.scale(X_train)



	names = [   #"Support vector machine",
				#"RandomForestClassifier",
				"AdaBoostClassifier",
				"LinearDiscriminantAnalysis",
				"QuadraticDiscriminantAnalysis"]
	classifiers = [
		#SVC(max_iter = 1000),
		#RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
		AdaBoostClassifier(),
		LinearDiscriminantAnalysis(),
		QuadraticDiscriminantAnalysis()]

	clf = ""

	print "2"

	#parameters = [{}]
	print 3
	past_time = time.time()

	for name, clf in zip(names, classifiers):
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)
		print str(name) +" done, takes " + str(time.time() - past_time)

		print y_pred
		print "-"*40
		print "all data "+ data_type
		print "accuracy : " + str(metrics.accuracy_score(y_test,y_pred))
		print "f1_score : " + str(metrics.f1_score(y_test,y_pred))
		print "roc_auc  : " + str(metrics.roc_auc_score(y_test,y_pred))
		print "precision: " + str(metrics.precision_score(y_test,y_pred))
		print "recall   : " + str(metrics.recall_score(y_test,y_pred))
		print "-"*40

		y_pred_rf = clf.predict_proba(X_test)[:, 1]
		fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)

		plt.figure(name)
		plt.plot([0, 1], [0, 1], 'k--')
		plt.plot(fpr_rf, tpr_rf, label='RF')
		plt.xlabel('False positive rate')
		plt.ylabel('True positive rate')
		plt.title(name + ' ROC curve')
		plt.legend(loc='best')
		plt.show()

		past_time = time.time()
		input()


def feature_selection(dataset_number):
	normal = [0,1,2,3,4,5,6,7,8,9,10,11]

	added_all = ["FFT_a","FFT_r","FFT_i","DWT_a","DWT_d","ave1","ave5","ave9","ave13","ave17","delta1","delta5","delta9","delta13","delta17"]

	added_time = ["FFT_a","FFT_r","FFT_i","DWT_a","DWT_d"]
	added_ave = ["ave1","ave5","ave9","ave13","ave17"]
	added_delta  = ["delta1","delta5","delta9","delta13","delta17"]


	data = {"FFT_a": [12,27,42,57,72,87],
	"FFT_r": [13,28,43,58,73,88],
	"FFT_i": [14,29,44,59,74,89],
	"DWT_a": [15,30,45,60,75,90],
	"DWT_d": [16,31,46,61,76,91],
	"ave1": [17,32,47,62,77,92],
	"ave5": [18,33,48,63,78,93],
	"ave9": [19,34,49,64,79,94],
	"ave13": [20,35,50,65,80,95],
	"ave17": [21,36,51,66,81,96],
	"delta1": [22,37,52,67,82,97],
	"delta5": [23,38,53,68,83,98],
	"delta9": [24,39,54,69,84,99],
	"delta13": [25,40,55,70,85,100],
	"delta17": [26,41,56,71,86,101]}

	pick = normal
	name = "AdaBoostClassifier all data normal feature"


	in_path = "./data/train_test_set/" + str(dataset_number)
	X_train = []
	y_train = []

	X_test = []
	y_test = []
	#학습 및 진단데이터 입력
	
	print "1"
	for line in open(in_path + "_train.csv"):
		parse = line.split(",")

		parse_picked = []
		for item in pick:
			parse_picked.append(parse[item])

		parse_picked.append(parse[-1])

		X_train.append(map(float, parse_picked[0:-1]))
		y_train.append(parse_picked[-1].strip("\n"))

	y_train = map(int,y_train)

	for line in open(in_path + "_test.csv"):
		parse_picked = []
		for item in pick:
			parse_picked.append(parse[item])

		parse_picked.append(parse[-1])

		X_test.append(map(float, parse_picked[0:-1]))
		y_test.append(parse_picked[-1].strip("\n"))

	y_test = map(int,y_test)

	print 2
	clf = AdaBoostClassifier()
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)

	name += "all"
	f_out = open(name + ".txt", "w")
	for i in y_pred:
		f_out.write(str(i) + ",")
	f_out.write("\n")

	for i in y_test:
		f_out.write(str(i) + ",")
	f_out.write("\n")
	

	print y_pred 
	print y_test
	print "-"*40
	print name
	print "accuracy : " + str(metrics.accuracy_score(y_test,y_pred))
	print "f1_score : " + str(metrics.f1_score(y_test,y_pred))
	print "roc_auc  : " + str(metrics.roc_auc_score(y_test,y_pred))
	print "precision: " + str(metrics.precision_score(y_test,y_pred))
	print "recall   : " + str(metrics.recall_score(y_test,y_pred))
	print "-"*40





def ocsvm():
	print(__doc__)

	import numpy as np
	import matplotlib.pyplot as plt
	import matplotlib.font_manager
	from sklearn import svm

	xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
	# Generate train data
	X = 0.3 * np.random.randn(100, 2)
	X_train = np.r_[X + 2, X - 2]
	# Generate some regular novel observations
	X = 0.3 * np.random.randn(20, 2)
	X_test = np.r_[X + 2, X - 2]
	# Generate some abnormal novel observations
	X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
	#print X_test

	# fit the model
	clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
	clf.fit(X_train)

	y_pred_train = clf.predict(X_train)
	y_pred_test = clf.predict(X_test)
	y_pred_outliers = clf.predict(X_outliers)

	print y_pred_train
	n_error_train = y_pred_train[y_pred_train == -1].size
	n_error_test = y_pred_test[y_pred_test == -1].size
	n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

	# plot the line, the points, and the nearest vectors to the plane
	Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)

	plt.title("Novelty Detection")
	plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
	a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
	plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange')

	b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white')
	b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green')
	c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red')
	plt.axis('tight')
	plt.xlim((-5, 5))
	plt.ylim((-5, 5))
	plt.legend([a.collections[0], b1, b2, c],
			   ["learned frontier", "training observations",
				"new regular observations", "new abnormal observations"],
			   loc="upper left",
			   prop=matplotlib.font_manager.FontProperties(size=11))
	plt.xlabel(
		"error train: %d/200 ; errors novel regular: %d/40 ; "
		"errors novel abnormal: %d/40"
		% (n_error_train, n_error_test, n_error_outliers))
	plt.show()	



def ocsvm_test():
	train_path = "./data/add_features/1.csv"
	test_path = "./data/add_features/2.csv"

	X_train = []
	X_test = []
	print 1
	for line in open(train_path).readlines():
		X_train.append(map(float, line.split(",")[1:2] + line.split(",")[3:-1]))
		#y_train.append(line.split(",")[-1].strip("\n"))

	for line in open(test_path).readlines():
		X_test.append(map(float, line.split(",")[1:2] + line.split(",")[3:-1]))
		#y_train.append(line.split(",")[-1].strip("\n"))

	print 2
	len_train = len(X_train)
	X_test_1 = X_train[len_train*9/10:-1]
	X_train = X_train[0:len_train*9/10]

	clf = svm.OneClassSVM(nu=1, kernel="rbf", gamma=0.1,max_iter = 800)
	clf = clf.fit(X_train)

	print 3
	ans = clf.predict(X_test_1)
	print len(ans)
	print list(ans).count(-1)
	print list(ans).count(1)

	print "-"*40
	ans = clf.predict(X_test)
	print len(ans)
	print list(ans).count(-1)
	print list(ans).count(1)


def pca_view():
	
	import matplotlib.pyplot as plt

	from sklearn import datasets
	from sklearn.decomposition import PCA
	from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

	iris = datasets.load_iris()

	X = iris.data
	y = iris.target
	target_names = iris.target_names

	pca = PCA(n_components=2)
	X_r = pca.fit(X).transform(X)

	print X_r
	print y
	lda = LinearDiscriminantAnalysis(n_components=2)
	X_r2 = lda.fit(X, y).transform(X)

	# Percentage of variance explained for each components
	print('explained variance ratio (first two components): %s'
		  % str(pca.explained_variance_ratio_))

	plt.figure()
	for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
		plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
	plt.legend()
	plt.title('PCA of IRIS dataset')

	plt.figure()
	for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
		plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], c=c, label=target_name)
	plt.legend()
	plt.title('LDA of IRIS dataset')

	plt.show()
	
	#X를 로드애서 PCA돌리고 이를 plot 한다. 
	c =  ['','red', 'green', 'blue', 'brown', 'yellow',"black","purple"]	
	for i in range(15,21):
		#path = "./data/file_per_car/"+str(i)+".csv"
		path = "./data/add_features/"+str(i)+".csv"
		X = []
		y = []

		for line in open(path).readlines():
			X.append(map(float, line.split(",")[1:2] + line.split(",")[3:-1]))
			y.append(i)

		from sklearn.decomposition import PCA
		X_pca = random.sample(PCA(n_components=2).fit_transform(X),100)
		
		X_lda = lda.fit(X, y).transform(X)



		for ite in range(0,100):
			plt.scatter(X[ite][0], X[ite][1], c=c[i-14])

	for i in range(1,3):
		#path = "./data/file_per_car/"+str(i)+".csv"
		path = "./data/add_features/"+str(i)+".csv"
		X = []
		y = []
		lines = random.sample(open(path).readlines(), 100)
		
		for line in lines:
			X.append(map(float, line.split(",")[1:2] + line.split(",")[3:-1]))
			y.append(i)
			print i
	lda = LinearDiscriminantAnalysis(n_components=2)
	X_lda = lda.fit(X, y).transform(X)
	print X_lda

	for ite in range(0,100):
		plt.scatter(X[ite][0], X[ite][1], c=c[i-14])
	plt.title('PCA of car dataset')

	plt.show()	

	 #여기서부터 다시 시작
def roc_view():

	y_pred_rf = clf.predict_proba(X_test)[:, 1]
	fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)

	plt.figure(name)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr_rf, tpr_rf, label='RF')
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title(name + ' ROC curve')
	plt.legend(loc='best')
	plt.show()

global id_mapping

id_mapping = {"02dae3bcbc7dfc8201e9115063846a0a" : 1,
"050ae5c4c9743edf4f9b982dbab3b5c8" : 2,
"05712c7b058474e8f0e652e7256d781d" : 3,
"08c4d3f9024564a85bd9a1f69bb0efae" : 4,
"0bf1ef1d1eafbc934301326325b6903d" : 5,
"1beceabea00c36fb01f5c9cd82bddf71" : 6,
"1fbe8f642c1b6c23100534c9ef006089" : 7,
"2e52e9435e9f77357214e64255a30533" : 8,
"390b1fcf3ad83be250299e899132a74d" : 9,
"3f5c578e6d409faed623eebbc7512236" : 10,
"44be80722dcf560e39644b00113eeeec" : 11,
"5243333b557dab4a0396fb7e15e6061b" : 12,
"65e7d99f12bab89f21dd2373f5df58e7" : 13,
"70478328aba4ce04b8c3a949077f120e" : 14,
"755cb52a49b102cddae4e13f02111ca0" : 15,
"83f65de7874743181678a7e29c062ee4" : 16,
"96f754da7f2ca52afeda426fc3cc7fb6" : 17,
"981774ae6e3d1daec1b43a490a29447e" : 18,
"9e12b39d1fdc533e338b39301e17745c" : 19,
"ad1c743bb27dceb0510a3bdf847e5dd3" : 20,
"b1f933e43991c00e25f144e9f560a5ba" : 21,
"c034d1d68fa5f311acb766aed4226130" : 22,
"cb78382a5dfdee9ac09c1a9cb9898bf5" : 23,
"cc97f891cad5a1cf0e2a8e51ef0365c5" : 24,
"ce6798c0e9004f7c03f8da8c30f3b2a0" : 25,
"d0f5cdd6a406f518e1daf28f6bd54482" : 26,
"d1611c35ae49db1006efb627d8301260" : 27,
"dc4cde0a763a1438fbd1246b4ea26b3a" : 28,
"dc7a92564a2ac9a2a34bc8b53a5831cf" : 29,
"f5bd957eb51485110bb8cce1925a72e7" : 30}

#change_data()
#divide_users()

#add_features(1)
"""
past = time.time()
for i in range(1,31):
	add_features(i)
	print str(i) + " done " + str(time.time()- past)
	past = time.time()


make_data_set()
"""

#algorithm_model_making()
#performance_test()
#new_intergrated_performance(1)
#ocsvm()
#ocsvm_test()
#pca_view()
#feature_selection(1)




#Name        : P3Task1andTask2.py
#Author      : Anmol Nayak
#Version     : October 2017
#Description : Forecasting time series
import csv
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

with open('NAYAK ANMOL.csv') as file:
	next(file)
	data = csv.reader(file)
	x=[]
	xtrain=[]
	xtest=[]
	counter=0
	for row in data:
		x.append(float(row[0]))
		if(counter<1500):
			xtrain.append(float(row[0]))
		else:
			xtest.append(float(row[0]))
		counter=counter+1
	
	#Simple moving average model for m=1 on training dataset
	m = 1
	cumsum, moving_aves = [0], []
	exp_pred = []
	
	for i, xval in enumerate(xtrain, 1):
		cumsum.append(cumsum[i-1] + xval)
		if i>m:
			moving_ave = (cumsum[i-1] - cumsum[i-m-1])/m
			moving_aves.append(moving_ave)
		else:
			moving_aves.append(xval)
	
	print('Mean squared error:',mean_squared_error(xtrain, moving_aves))
	rms = sqrt(mean_squared_error(xtrain, moving_aves))
	print('RMSE:',rms)
	
	#Exponential smoothing model for a=0.9 on training dataset
	a=0.9
	for j, xval in enumerate(xtrain, 1):
		if (j==1):
			exp_pred.append(float(0))	
		else:
			exp_calc= (xtrain[j-2])*a + (1-a)*(exp_pred[j-2])
			exp_pred.append(exp_calc)
	
	rms = sqrt(mean_squared_error(xtrain, exp_pred))
	print('Mean squared error:',mean_squared_error(xtrain, exp_pred))
	print('RMSE:',rms)
	
	#Plot for SMA model
	xrange=list(range(1,1501))
	plt.scatter(xrange,xtrain,s=5)
	plt.scatter(xrange,moving_aves,color='red',s=5)
	axes = plt.gca()
	axes.set_xlim([0,1500])
	axes.set_ylim([min(min(xtrain),min(moving_aves)),max(max(xtrain),max(moving_aves))])
	plt.title("Original values(blue) vs Predicted values(red)", size=28)
	plt.xlabel("Point#(1-1500)", size=24)
	plt.ylabel("Value", size=24)
	plt.show()
	
	#Plot for Exponential smoothing model
	xrange=list(range(1,1501))
	plt.scatter(xrange,xtrain,s=5)
	plt.scatter(xrange,exp_pred,color='red',s=5)
	axes = plt.gca()
	axes.set_xlim([0,1500])
	axes.set_ylim([min(min(xtrain),min(exp_pred)),max(max(xtrain),max(exp_pred))])
	plt.title("Original value(blue) vs Predicted values(red)", size=28)
	plt.xlabel("Point#(1-1500)", size=24)
	plt.ylabel("Value", size=24)
	plt.show()
	
	#SMA for test dataset
	testcumsum, testmoving_aves = [0], []
	testexp_pred = []
	for i, xval in enumerate(xtest, 1):
		testcumsum.append(testcumsum[i-1] + xval)
		if i>m:
			testmoving_ave = (testcumsum[i-1] - testcumsum[i-m-1])/m
			testmoving_aves.append(testmoving_ave)
		else:
			testmoving_aves.append(xval)
	
	
	print('Mean squared error:',mean_squared_error(xtest, testmoving_aves))
	rms = sqrt(mean_squared_error(xtest, testmoving_aves))
	print('RMSE:',rms)
	
	#Exponential smoothing model for test dataset
	a=0.9
	for j, xval in enumerate(xtest, 1):
		if (j==1):
			testexp_pred.append(float(0))	
		else:
			testexp_calc= (xtest[j-2])*a + (1-a)*(testexp_pred[j-2])
			testexp_pred.append(testexp_calc)
	
	rms = sqrt(mean_squared_error(xtest, testexp_pred))
	
	print('Mean squared error:',mean_squared_error(xtest, testexp_pred))
	print('RMSE:',rms)
		

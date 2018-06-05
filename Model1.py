#Name        : Model1.py
#Author      : Anmol Nayak
#Version     : March 2018
#Description : Face detection using PCA and Gaussian Distributions
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image,ImageOps
import glob
from math import pow,sqrt,exp,pi,log
from sklearn.decomposition import PCA

pca = PCA(n_components=100)

n_components=100
image_list=[]
num_train_imgs=1000
rows=60
cols=60
fnf=2
posprob_facemdl=[]
posprob_nonfacemdl=[]
negprob_facemdl=[]
negprob_nonfacemdl=[]
num_pos_testimg=100
num_neg_testimg=100

def meanfunction( x ):
        return sum(x)/float(num_train_imgs)


def pdf_face_nonface(filepath,flag):
    if flag==1:
        imgac =Image.open(filepath)
        imgrays = ImageOps.grayscale(imgac)
    if flag==0:
        imgfullac = Image.open(filepath)
        crop_rectangle = (0, 0, 100, 100)
        cropped_imac = imgfullac.crop(crop_rectangle)
        imgrays = ImageOps.grayscale(cropped_imac)

    new_imgs = imgrays.resize((rows,cols))
    new_m = np.array(new_imgs.getdata()).reshape(1,rows*cols)
    trans = np.dot(new_m,np.transpose(eig))

    s,logdet = np.linalg.slogdet(newcov)
    tran = trans.reshape(n_components,1)

    num = np.dot(np.dot(np.transpose(tran-mean_tranfmd),np.linalg.inv(newcov)),(tran-mean_tranfmd))
    prob = exp(-0.5*num[0]-log(pi)-0.5*logdet)
    if flag==1 and fnfcount==0:
        posprob_facemdl.append(prob)
    elif flag==0 and fnfcount==0:
        negprob_facemdl.append(prob)
    elif flag==1 and fnfcount==1:
        posprob_nonfacemdl.append(prob)
    else:
        negprob_nonfacemdl.append(prob)

for fnfcount in range(0,fnf):
    if fnfcount==0:
        path= '/home/anmol/Downloads/CroppedYale/*/*.pgm' 
    else:
        path = '/home/anmol/Downloads/non_face_patch/*/*.pgm'

    count=0
    
    for filename in glob.glob(path):
        if fnfcount==0:
            img =Image.open(filename)
            imgray = ImageOps.grayscale(img)
            new_img = imgray.resize((rows,cols))

        if fnfcount==1:
            imgfull = Image.open(filename)
            crop_rectangle = (0, 0, 100, 100)
            cropped_im = imgfull.crop(crop_rectangle)
            imgray = ImageOps.grayscale(cropped_im)
            new_img = imgray.resize((rows,cols))

        if new_img not in image_list:
            if count==0:
                prev=np.array(new_img.getdata())
            else:
                prev=prev+np.array(new_img.getdata())
            image_list.append(np.array(new_img.getdata()).reshape(-1,1))
            count=count+1
        if count==num_train_imgs-1:
            break
    
    count=0

    nmean=prev/float(num_train_imgs)
    mean = np.reshape(nmean,(rows*cols,1))


    for item in image_list:
        if count==0:
                face_train = image_list[0]
                count=count+1
        else:
                face_train = np.hstack((face_train,item))

    image_listnew = np.transpose(face_train)
    inputmat = pca.fit_transform(image_listnew)
    pca.fit(image_listnew)
    newcov= pca.get_covariance()
    eig = pca.components_

    pca.fit(inputmat)

    newcov =  pca.get_covariance()
    d, u = np.linalg.eigh(newcov)

    meanj = np.apply_along_axis( meanfunction, axis=1, arr=np.transpose(inputmat))
    mean_tranfmd= meanj.reshape(n_components,1)

    if fnfcount==0:
        misc.imsave('Model1-face-meanimg.jpg', np.reshape(mean_tranfmd,(sqrt(n_components),sqrt(n_components))))
        misc.imsave('Model1-face-covarianceimg.jpg', np.reshape(newcov,(n_components,n_components)))
    if fnfcount==1:
        misc.imsave('Model1-nonface-meanimg.jpg', np.reshape(mean_tranfmd,(sqrt(n_components),sqrt(n_components))))
        misc.imsave('Model1-nonface-covarianceimg.jpg', np.reshape(newcov,(n_components,n_components)))

    posimgcnt=0
    for filename in glob.glob('/home/anmol/Downloads/face-test-imgs/*.pgm'):
        if(posimgcnt<=num_pos_testimg-1):
            pdf_face_nonface(filename,1)
            posimgcnt=posimgcnt+1
        else:
            break
    negimgcnt=0
    for filename in glob.glob('/home/anmol/Downloads/non-face-test-imgs/*.pgm'):
        if(negimgcnt<=num_neg_testimg-1):
            pdf_face_nonface(filename,0)
            negimgcnt=negimgcnt+1
        else:
            break

    fnfcount=fnfcount+1

truepos=0
falsepos=0
trueneg=0
falseneg=0

postotal=num_pos_testimg
negtotal=num_neg_testimg

for i in range(0,postotal):
    if posprob_facemdl[i]>posprob_nonfacemdl[i]:
        truepos+=1
    if posprob_facemdl[i]<=posprob_nonfacemdl[i]:
        falseneg+=1

for i in range(0,negtotal):
    if negprob_facemdl[i]>negprob_nonfacemdl[i]:
        falsepos+=1
    if negprob_facemdl[i]<=negprob_nonfacemdl[i]:
        trueneg+=1

print "True positives: ",truepos
print "False positives: ", falsepos
print "True negatives: " ,trueneg
print "False negatives: " ,falseneg

fpr=falsepos/float(negtotal)
fnr=falseneg/float(postotal)
mcr=(falsepos+falseneg)/float(postotal+negtotal)

print "False positive rate: ",fpr
print "False negative rate: " ,fnr
print "Misclassification rate: ", mcr


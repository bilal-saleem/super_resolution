import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy import ndimage

def loadImages(img_path):
	''' 
	Loads all grayscale images in img_path, returns in np array
	Shape of returned list is (# of images, width, height)
	If there is a non image file, will exit with a runtime error.
	'''

	img_list = []
	file_names = os.listdir(img_path)
	for filename in file_names:
		img = cv2.imread(img_path + "/" + filename, 0)
		img_list.append(img)
	return np.array(img_list)

def getTranslationModel(images):
	'''
	Calculates the motion vector for each image
	with respect to the first image
	'''
  
	F = [np.array([[1, 0, 0], [0, 1, 0]])]
	# Create a sift detector/computer
	sift = cv2.xfeatures2d.SIFT_create()	
	# Create a matcher to find matching points
	matcher = cv2.DescriptorMatcher_create("BruteForce")

	# For each images in the image set, register it to the first image.
	kp1, features1 = sift.detectAndCompute(images[0, :, :], None)
	kp1 = np.float32([kp.pt for kp in kp1])

	for i in range(1, images.shape[0]):
		# Get the feature points
		kp2, features2 = sift.detectAndCompute(images[i, :, :], None)
		# Convert the points to indexable points
		kp2 = np.float32([kp.pt for kp in kp2])
		# Get the matches
		rawMatches = matcher.knnMatch(features1, features2, k=2)
		matches = []
		
		r = 0.8
		for m in rawMatches:
		# ensure the distance is within a certain ratio of each
		# other (i.e. Lowe's ratio test)   	
			temp_dist = np.array([m[0].distance, m[1].distance])
			ratio = np.min(temp_dist)/np.max(temp_dist)
			if len(m) == 2 and  ratio < r: 
				matches.append((m[0].trainIdx, m[0].queryIdx))				

		# Convert the matches to a format used by cv2.estimateAffinePartial2D
		ptsA = np.float32([kp1[i] for (_,i) in matches])
		ptsB = np.float32([kp2[i] for (i,_) in matches])
		# Calculate the translational model.
		M, w = cv2.estimateAffinePartial2D(ptsA, ptsB)
		M[0:2, 0:2] = np.array([[1, 0], [0, 1]])
		F.append(M)
	return np.array(F)

def invertTranslationModels(F):
	# Inverts each translation matrix
	Finv = np.copy(F)
	Finv[:, :, 2] = -2*Finv[:, :, 2]
	#Finv[i] = cv2.invertAffineTransform(Finv[i])
	return Finv


def getTranslatedImages(images, F):
	# Translates each image according to its corresponding
	# translation matrix in F
	translatedImages = []
	ncol, nrow = images[0].shape[1], images[0].shape[0]
	for i in range(0, images.shape[0]):
		translatedImages.append(cv2.warpAffine(images[i, :, :], F[i], (ncol, nrow)) )
	return np.array(translatedImages)



def data_fusion_gradient(images, X, F, H, r):
	'''
	Returns: gradient of data fusion term
	images: original low-res images
	X: super-resolution image we're trying to find
	F: motion vectors for low-res images
	H: the point spread function for the camera/atmosphere.
	 Really, just a gaussian blur kernel.
	r: The upscaling factor (2x, 3x, etc.)
	'''

	#Filter the current guess.
	X_filt = ndimage.correlate(X, H, mode='constant')

	# Allocate space for the shifted, low-res versions of current guess, X.
	X_new_lr = np.zeros(images.shape)
	down_dim = (X.shape[1] // r, X.shape[0] // r)
	up_dim = (X.shape[1], X.shape[0])

	# For each shift matrix in F, shift the downsampled HR guess
	for i in range(0, images.shape[0]):
		X_new_lr[i, :, :] = cv2.resize(cv2.warpAffine(X_filt, F[i], up_dim), down_dim, interpolation = cv2.INTER_AREA)

	X_new_lr = np.sign(X_new_lr - images)

	X_new_hr = np.zeros((images.shape[0], r*images.shape[1], r*images.shape[2]))

	Finv = invertTranslationModels(F)

	for i in range(0, images.shape[0]):
		X_new_hr[i, :, :] = cv2.resize(X_new_lr[i, :, :], up_dim)
		X_new_hr[i, :, :] = ndimage.correlate(X_new_hr[i, :, :], H.T, mode='constant')
		X_new_hr[i, :, :] = cv2.warpAffine(X_new_hr[i, :, :], Finv[i], up_dim)

	df_grad = np.sum(X_new_hr, axis=0)

	return df_grad


def regularization_gradient(images, X, F, r):
	'''
	Returns: gradient of regularization term
	Calculate the gradient of the bilateral TV regularization term
	images: original low-res images
	X: super-resolution image we're trying to find
	F: motion vectors for low-res images
	r: The upscaling factor (2x, 3x, etc.)	
	'''
	P = 2
	alpha = 0.7
	i = 0
	dim = (X.shape[1], X.shape[0])

	X_new = np.zeros(X.shape)

	for l in range(-P, P+1):
		for m in range(-P, P+1):						
			shift_mat = np.array([[1, 0, l], [0, 1, m]], dtype='float32')

			X_shift = cv2.warpAffine(X, shift_mat, dim)

			X_shift_sign = np.sign(X - X_shift)

			shift_mat = np.array([[1, 0, -1*l], [0, 1, -1*m]], dtype='float32')

			X_back_shift = cv2.warpAffine(X_shift_sign, shift_mat, dim)

			X_new = X_new + ((X_shift_sign - X_back_shift)*(alpha**(np.abs(l)+np.abs(m))))

	return X_new


def gradient_descent(images, F, H, r=2, max_iter=20):
	'''
	Perform gradient descent to find the high resolution
	super resolution image X from LR images
	F: The motion vectors predicted from getTranslationalModel
	H: The gaussian blurring kernel
	r: Upscaling faction
	max_iter: Maximum number of gradient descent steps
	'''
	up_dim = (r*images[0].shape[1], r*images[0].shape[0])
	X = cv2.resize(images[0],  up_dim)
	i = 0
	beta = 1
	lam = 0.05
	
	while i < max_iter:
		df_grad = data_fusion_gradient(images, X, F, H, r)
		reg_grad = regularization_gradient(images, X, F, r)
		X = X - beta * (df_grad + lam*reg_grad)
		#X = cv2.normalize(X - beta * (Xgrad + lam*Zgrad), X, 0, 255, cv2.NORM_MINMAX)
		i = i + 1

	return X

def gauss(siz, sigma):
	'''
	Generate a gaussian kernel with size siz*siz,
	and std sigma
	'''
	tmp = int(siz / 2)
	if np.mod(siz,2)==0:
		x = np.linspace(-(tmp-0.5), tmp-0.5, siz)
	else:
		x = np.linspace(-(tmp), tmp, siz) # for odd size

	sig_sq = sigma*sigma

	gauss = (1/np.sqrt(2*np.pi*sig_sq))*np.exp(-1*(x**2 / sig_sq)) # make 1D gaussian filter
	gauss = gauss.reshape(1, siz)
	# Compose 2D Gaussian filter from 1D
	gauss2 =  np.repeat(gauss, siz, axis=0) * np.repeat(np.transpose(gauss), siz, axis=1) 

	return gauss2
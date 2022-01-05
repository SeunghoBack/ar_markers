import sys
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# ar_markers 라이브러리에서 detect_markers를 사용
from ar_markers import detect_markers

#fPath = '/home/nearthlab/test_cpp/OksangSagin/'
fPath = '/home/nearthlab/Pictures/'

def detect_blur_fft(image, size=60):
    # grab the dimensions of the image and use the dimensions to
    # derive the center (x, y)-coordinates
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    # compute the FFT to find the frequency transform, then shift
    # the zero frequency component (i.e., DC component located at
    # the top-left corner) to the center where it will be more
    # easy to analyze
    fft = np.fft.fft2(image)
    spec1 = 20 * np.log(np.abs(fft))
    fftShift = np.fft.fftshift(fft)
    spec2 = 20 * np.log(np.abs(fftShift))
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    spec3 = 20 * np.log(np.abs(fftShift))
    fftShift = np.fft.ifftshift(fftShift)
    spec4 = 20 * np.log(np.abs(fftShift))
    recon = np.fft.ifft2(fftShift)
    magnitude = 20 * np.log(np.abs(recon))
    plt.figure()
    plt.subplot(161), plt.imshow(image, cmap='gray')
    plt.subplot(162), plt.imshow(spec1, cmap='gray')
    plt.subplot(163), plt.imshow(spec2, cmap='gray')
    plt.subplot(164), plt.imshow(spec3, cmap='gray')
    plt.subplot(165), plt.imshow(spec4, cmap='gray')
    plt.subplot(166), plt.imshow(magnitude, cmap='gray')
    plt.show()
    mean = np.mean(magnitude)
    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value
    return mean


def main(argv, imgName):
	# 마커 인식할 이미지 파일 이름 표시
	print("the image file name is " + argv)
	# 이미지 파일에서 이미지 추출
	frame = cv2.imread(argv, cv2.IMREAD_UNCHANGED)
	# 이미지에서 marker 위치 추출(찾기)
	markers = detect_markers(frame)
	# 찾은 마커 개수 표시
	print("{} markers found.".format( len(markers )))
	if len(markers) == 0:
		cv2.imwrite(fPath+'result/' + "undetected_" + imgName[:-4] + ".jpg", frame)
	# 마커 정보 표시
	prevMarker = None
	for marker in markers:
		# id = marker.id, 마커의 중심 위치 : marker.center 표시
		print("ID {}'s position is {}".format(marker.id, marker.center))
		# 해당 마커의 코너를 이쁘게(?) 표시. 사각형이니 4포인트(x,y)
		minX = 10000
		maxX = 0
		minY = 10000
		maxY = 0
		#if prevMarker == marker.id or marker.id != 1646:
		if marker.id != 1646:
			break;		
		prevMarker = marker.id
		for pos in marker.contours:
			if pos[0][0] < minX:
				minX = pos[0][0]
			if pos[0][0] > maxX:
				maxX = pos[0][0]
			if pos[0][1] < minY:
				minY = pos[0][1]
			if pos[0][1] > maxY:
				maxY = pos[0][1]
		img = frame[minY-20:maxY+20, minX-20:maxX+20]
		img = cv2.resize(img, dsize=(500,500), interpolation=cv2.INTER_AREA)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		mean = detect_blur_fft(gray, size=60)
		print("blur-fft-mean",mean)
		blurtype = 'None'
		if mean < 15:
			print("image is bluury")
			blurtype = 'blur'
		else:
			print("image is normal")
			blurtype = 'normal'
		
		#cv2.imwrite(fPath+'result/'+blurtype + "_" + imgName[:-4] +"_" + str(mean) + ".jpg", img)
		cv2.imshow('img', img)
		cv2.waitKey(0)

def main2(argv, imgName):
	# 마커 인식할 이미지 파일 이름 표시
	print("the image file name is " + argv)
	# 이미지 파일에서 이미지 추출
	frame = cv2.imread(argv, cv2.IMREAD_UNCHANGED)
	# 이미지에서 marker 위치 추출(찾기)
	img = frame[2250:2700, 3920:4370]
	#img = cv2.resize(img, dsize=(500,500), interpolation=cv2.INTER_AREA)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	mean = detect_blur_fft(gray, size=60)
	print("blur-fft-mean",mean)
	blurtype = 'None'
	if mean < 15:
		print("image is bluury")
		blurtype = 'blur'
	else:
		print("image is normal")
		blurtype = 'normal'
	
	#cv2.imwrite(fPath+'result/'+"undetected_" + blurtype + "_" + imgName[:-4] +"_" + str(mean) + ".jpg", img)
	#cv2.imshow('img', img)
	#cv2.waitKey(0)

if __name__ == '__main__':
	imgName = 'Screenshot from 2021-11-25 15-35-34.png'
	#main2(fPath + imgName, imgName)
	main(fPath+imgName, imgName)



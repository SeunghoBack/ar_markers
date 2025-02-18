import sys
import cv2
import numpy as np
import os
import csv
# ar_markers 라이브러리에서 detect_markers를 사용
from ar_markers import detect_markers

#fPath = '/home/nearthlab/test_cpp/OksangSagin/'
fPath = '/home/nearthlab/Downloads/143DKS00/'
f = open('write.csv','w', newline='')
wr = csv.writer(f)

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
    fftShift = np.fft.fftshift(fft)
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value
    return mean

def increase_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def isNear(markerPos, markerCenter):
    for markerP in markerPos:
        if abs(markerP[0] - markerCenter[0]) < 3 and abs(markerP[1] - markerCenter[1]) < 3:
        #    print("isNear", markerP, markerCenter)
            return True
        #else:
            #print("not Near", markerP, markerCenter)
    return False


def main(argv, imgName):
	# 마커 인식할 이미지 파일 이름 표시
	print("the image file name is " + argv)
	# 이미지 파일에서 이미지 추출
	frame = cv2.imread(argv, cv2.IMREAD_UNCHANGED)
	#frame2 = frame
	#frame = increase_brightness(frame, 90)
	# 이미지에서 marker 위치 추출(찾기)
	markers = detect_markers(frame)
	# 찾은 마커 개수 표시
	print("{} markers found.".format( len(markers )))
	#if len(markers) == 0:
		#cv2.imwrite(fPath+'result/' + "undetected_" + imgName[:-4] + ".jpg", frame)
	# 마커 정보 표시
	prevMarker = None
	markerPos = []
	csvText = []
	meanList = []
	csvText.append(imgName)

	for marker in markers:
		# id = marker.id, 마커의 중심 위치 : marker.center 표시
		if isNear(markerPos, marker.center):
			continue
		print("ID {}'s position is {}".format(marker.id, marker.center))
		markerPos.append(marker.center)

		#if mar
		# 해당 마커의 코너를 이쁘게(?) 표시. 사각형이니 4포인트(x,y)
		minX = 10000
		maxX = 0
		minY = 10000
		maxY = 0
		if marker.id != 1646:
			print("But it's not")
			break;		
		#prevMarker = marker.id
		for pos in marker.contours:
			if pos[0][0] < minX:
				minX = pos[0][0]
			if pos[0][0] > maxX:
				maxX = pos[0][0]
			if pos[0][1] < minY:
				minY = pos[0][1]
			if pos[0][1] > maxY:
				maxY = pos[0][1]
		if minY < 20 or minX < 20 or maxY > frame.shape[0]-20 or maxX > frame.shape[1]-20:
			continue
		img = frame[minY-20:maxY+20, minX-20:maxX+20]
		img = cv2.resize(img, dsize=(500,500), interpolation=cv2.INTER_AREA)
		#img2 = frame2[minY-20:maxY+20, minX-20:maxX+20]
		#img2 = cv2.resize(img2, dsize=(500,500), interpolation=cv2.INTER_AREA)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		mean = detect_blur_fft(gray, size=60)
		meanList.append(mean)
		print("blur-fft-mean",mean)
	if len(meanList) != 0:
		csvText.append(min(meanList))
		csvText.append(max(meanList))
		csvText.append(meanList)
	wr.writerow(csvText)
	print(csvText)		
'''
		blurtype = 'None'		
		if mean < 15:
			print("image is bluury")
			blurtype = 'blur'
		else:
			print("image is normal")
			blurtype = 'normal'
		
		cv2.imwrite(fPath+'result/'+blurtype + "_" + str(mean) + "_" + imgName[:-4] + ".jpg", img)
		print(" ")
		#cv2.imshow('img', img)
		#cv2.waitKey(0)
'''

if __name__ == '__main__':
	imgNames = os.listdir(fPath)
	imgNames.sort()
	i = 0
	for imgName in imgNames:        
		main(fPath + imgName, imgName)
		print(" ")
		print(" ")
        #main('/home/nearthlab/Pictures/Screenshot from 2021-11-25 15-35-34.png')
	f.close()


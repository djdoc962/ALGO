import cv2
# import pytesseract
import numpy as np
# import preprocessing as process
import module.table_detection as table_detection
from skimage.transform import hough_line, hough_line_peaks
from skimage.color import rgb2gray
from skimage.feature import canny
import os, datetime

# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
# tessdata_dir_config = r'--tessdata-dir "/Users/jojolin/Desktop/work/開發工具/sources/systalk-ocr/tessdata"'


def skew_angle_hough_transform(image):
    # convert to edges
    edges = canny(image)
    # Classic straight-line Hough transform between 0.1 - 180 degrees.
    tested_angles = np.deg2rad(np.arange(0.1, 180.0))
    h, theta, d = hough_line(edges, theta=tested_angles)
    
    # find line peaks and angles
    accum, angles, dists = hough_line_peaks(h, theta, d)
    print('peaks------------------')
    print('angles: '+str(np.around(angles, decimals=2)))
    
    # round the angles to 2 decimal places and find the most common angle.
    most_common_angle = mode(np.around(angles, decimals=2))[0]
    print('most_common_angle: '+str(most_common_angle[0]))
    # convert the angle to degree for rotation.
    skew_angle = np.rad2deg(most_common_angle - np.pi/2)
    return skew_angle

def draw_hough_lines(image):
    tested_angles = np.deg2rad(np.arange(0.1, 180.0))
    h, theta, d = hough_line(image, theta=tested_angles)

    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    # print('origin: '+str(origin))
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        print('angle: '+str(angle))
        print('(y0,y1): ('+str(y0)+', '+str(y1)+')')

        # ax[1].plot(origin, (y0, y1), '-r')
    


print('[START] Text extraction from table')
name = 'import_template'
filename = './image/'+name+'.png'

src = cv2.imread(filename)
print('image size: '+ str(src.shape))

# table lines removal
kernel_size = 5
kernel = np.ones((kernel_size,kernel_size), np.uint8)

# morphology operations for remove table/border
# TODO: 先灰階且二元化
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
thresh, img_bw = cv2.threshold(gray,0,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imwrite("../Images/BINARY.jpg", img_bw)

img_erosion = cv2.erode(img_bw, kernel, iterations=1)
img_dilate = cv2.dilate(img_erosion, kernel, iterations=1)
cv2.imwrite("../Images/erosion.jpg", img_erosion)
cv2.imwrite("../Images/dilation.jpg", img_dilate)

tic = datetime.datetime.now()
# get line for table
horizontal, vertical = table_detection.detect_lines(src,title=name,threshold = 20, minLinLength = 20, maxLineGap = 6, display=False, write = False)
print("preprocess: The processing of Hough transform elapsed time : ", datetime.datetime.now() - tic)
print('horizontal: '+ str(len(horizontal) ))
for i, line in enumerate(horizontal):
    print(line)

print('vertical: '+ str(len(vertical) ))
for i, line in enumerate(vertical):
    print(line)


s = np.shape(src)
print("rgb size: "+str(s)+ ' len: '+str(len(s)))
rgb = np.asarray(src)
if len(s) > 2:
    rgb = rgb[...,:3]
    image = rgb2gray(rgb)
edges = canny(image)
tested_angles = np.deg2rad(np.arange(0.1, 180.0))
h, theta, d = hough_line(edges, theta=tested_angles)
# print('tested_angles => '+ str(theta))

tic = datetime.datetime.now()
table_detection.line_detector(src,title=name)
print("preprocess: The processing of line detector elapsed time : ", datetime.datetime.now() - tic)



# filename = '../images/'+name+'_VH.png'
# src = cv2.imread(filename)
# print('image size: '+ str(src.shape))
# # get line for table
# horizontal, vertical = table_detection.detect_lines(src,title=name,threshold = 0, minLinLength = 150, maxLineGap = 0, display=False, write = False)
# print('horizontal: '+ str(len(horizontal) ))
# for i, line in enumerate(horizontal):
#     print(line)

# print('vertical: '+ str(len(vertical) ))
# for i, line in enumerate(vertical):
#     print(line)



# finally text recognition via tesseract 
# config = '-l eng --psm 3 --oem 3 ' + tessdata_dir_config
# text = pytesseract.image_to_string(src,config=config)
# print(text)


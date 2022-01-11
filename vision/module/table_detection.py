
import sys
import math
import cv2
import numpy as np
import pandas as pd
# import pytesseract

def is_vertical(line):
    return line[0]==line[2]

def is_horizontal(line):
    return line[1]==line[3]
    
def overlapping_filter(lines, sorting_index):
    separation = 5
    filtered_lines = []
    
    lines = sorted(lines, key=lambda lines: lines[sorting_index])
    
    for i in range(len(lines)):
            l_curr = lines[i]
            if(i>0):
                l_prev = lines[i-1]
                if ( (l_curr[sorting_index] - l_prev[sorting_index]) > separation):
                    filtered_lines.append(l_curr)
            else:
                filtered_lines.append(l_curr)
                
    return filtered_lines
               
def detect_lines(image, title='hough_line', rho = 1, theta = np.pi/180, threshold = 50, minLinLength = 290, maxLineGap = 6, display = False, write = False):
    print('[START] detecting lines via Hough transform ...')
    # Check if image is loaded fine
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # (thresh, gray) = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    if gray is None:
        print ('Error opening image!')
        return -1
    
    dst = cv2.Canny(gray, 50, 150, None, 3)
    
    # Copy edges to the images that will display the results in BGR
    cImage = np.copy(image)
    
    #linesP = cv.HoughLinesP(dst, 1 , np.pi / 180, 50, None, 290, 6)
    linesP = cv2.HoughLinesP(dst, rho , theta, threshold, None, minLinLength, maxLineGap)
    
    horizontal_lines = []
    vertical_lines = []
    
    if linesP is not None:
        #for i in range(40, nb_lines):
        print('seperating lines into horizontal and vertical lines ...')
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            # print(l)
            if (is_vertical(l)):
                vertical_lines.append(l)
                
            elif (is_horizontal(l)):
                horizontal_lines.append(l)
        # remove overlapping lines(distance in 5 pixels) with the same angle  
        horizontal_lines = overlapping_filter(horizontal_lines, 1)
        vertical_lines = overlapping_filter(vertical_lines, 0)

    display = True        
    if (display):
        for i, line in enumerate(horizontal_lines):
            cv2.line(cImage, (line[0], line[1]), (line[2], line[3]), (0,255,0), 2, cv2.LINE_AA)
            
            cv2.putText(cImage, str(i) + "h", (line[0] + 5, line[1]), cv2.FONT_HERSHEY_SIMPLEX,  
                       0.5, (0, 0, 0), 1, cv2.LINE_AA) 
            
        for i, line in enumerate(vertical_lines):
            cv2.line(cImage, (line[0], line[1]), (line[2], line[3]), (0,0,255), 2, cv2.LINE_AA)
            cv2.putText(cImage, str(i) + "v", (line[0], line[1] + 5), cv2.FONT_HERSHEY_SIMPLEX,  
                       0.5, (0, 0, 0), 1, cv2.LINE_AA) 
            
        # cv2.imshow("Source", cImage)
        # #cv.imshow("Canny", cdstP)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
    if (write):
        print('saving detected lines image ...')
        cv2.imwrite("../Images/" + title + ".png", cImage)

    cv2.imwrite("../Images/" + title + "_canny.png", dst)
    cv2.imwrite("../Images/" + title + "_HoughLines.png", cImage)
    print('[FINISH] detecting lines via Hough transform ...')  
    return (horizontal_lines, vertical_lines)


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)



def line_detector(image,title='line_detector'):
    print('line detetcor for straigh line ')
    copy_img = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
    if gray is None:
        print ('Error opening image!')
        return -1

    #thresholding the image to a binary image
    thresh,img_bin = cv2.threshold(gray,128,255,cv2.THRESH_BINARY |cv2.THRESH_OTSU)
    #inverting the image 
    img_bin = 255-img_bin
    cv2.imwrite("../Images/" + title + "_binary.png", img_bin) 
    #Plotting the image to see the output
    # plotting = plt.imshow(img_bin,cmap='gray') 
    # plt.show()
    # cv2.imshow("Binary", img_bin)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Length(width) of kernel as 100th of total width
    kernel_len = np.array(image).shape[1]//100
    # Defining a vertical kernel to detect all vertical lines of image 
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    # Defining a horizontal kernel to detect all horizontal lines of image
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    # A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    #Use vertical kernel to detect and save the vertical lines in a jpg
    image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
    cv2.imwrite("../Images/" + title + "_vertical_lines.png", vertical_lines)

    #Use horizontal kernel to detect and save the horizontal lines in a jpg
    image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
    cv2.imwrite("../Images/" + title + "_horizontal_lines.png",horizontal_lines)

    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)

    #Eroding and thesholding the image
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    thresh, img_vh = cv2.threshold(img_vh,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite("../Images/" + title + "_VH.png", img_vh)
    bitxor = cv2.bitwise_xor(gray,img_vh)
    cv2.imwrite("../Images/" + title + "_bitxor.png", bitxor)
    bitnot = cv2.bitwise_not(bitxor)
    cv2.imwrite("../Images/" + title + "_bitnot.png", bitnot)
    # cv2.imshow("Bitnot", bitnot)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Detect contours for following box detection
    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort all the contours by top to bottom.
    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")

    #Creating a list of heights for all detected boxes
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
    #Get mean of heights
    mean = np.mean(heights)

    #Create list box to store all boxes in  
    box = []
    point_color = (0,0,255)
    count = 0
    shape = copy_img.shape
    print('heigh: '+str(shape[1]))
    # 有些box太大的不可能是欄位，所以可以設定w/h來過濾 Get position (x,y), width and height for every contour and show the contour on image
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w<1000 and h<500):
            copy_img = cv2.rectangle(copy_img,(x,y),(x+w,y+h),(0,255,0),1)
            if y < int(shape[1]*0.2) or y > int(shape[1]*(3/4)):
                print('y value: '+ str(y))
                point_color = (255,0,0)
            else:
                point_color = (0,0,255)
            copy_img = cv2.circle(copy_img,(x,y),3,point_color,-1)
            box.append([x,y,w,h])
            count+=1
        else:
            print('[Warning] The countour with width >= 1000 or height >= 500, skip it to next one !')

    cv2.imwrite("../Images/" + title + "_box.png", copy_img)
    # TODO: return boxes[(x,y w,h),...]
    #Creating two lists to define row and column in which cell is located
    row=[]
    column=[]
    j=0
    #Sorting the boxes to their respective row and column
    for i in range(len(box)):
        if(i==0):
            column.append(box[i])
            previous=box[i]
        else:
            if(box[i][1]<=previous[1]+mean/2):
                column.append(box[i])
                previous=box[i]
                if(i==len(box)-1):
                    row.append(column)
            else:
                row.append(column)
                column=[]
                previous = box[i]
                column.append(box[i])
    print(str(len(column)) + ' Columns ======================')
    print(str(column))
    print(str(len(row)) + ' Rows ==========================')
    print(str(row))
    #calculating maximum number of cells
    countcol = 0
    for i in range(len(row)):
        countcol = len(row[i])
        if countcol > countcol:
            countcol = countcol

    print('countcil: ' + str(countcol))
    #Retrieving the center of each column
    center = [int(row[i][j][0]+row[i][j][2]/2) for j in range(len(row[i])) if row[0]]
    center=np.array(center)
    center.sort()


    finalboxes = []
    for i in range(len(row)):
        lis=[]
        for k in range(countcol):
            lis.append([])
        for j in range(len(row[i])):
            diff = abs(center-(row[i][j][0]+row[i][j][2]/4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            lis[indexing].append(row[i][j])
        finalboxes.append(lis)

    print('finalboxes===============')
    print(np.shape(finalboxes))
    """"
    copy_img2 = image.copy()
    # 最花時間的步驟from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
    outer=[]
    for i in range(len(finalboxes)):
        for j in range(len(finalboxes[i])):
            inner=""
            if(len(finalboxes[i][j])==0):
                outer.append(' ')
            else:
                for k in range(len(finalboxes[i][j])):
                    y,x,w,h = finalboxes[i][j][k][0],finalboxes[i][j][k][1], finalboxes[i][j][k][2],finalboxes[i][j][k][3]
                    copy_img2 = cv2.rectangle(copy_img2,(x,y),(x+w,y+h),(0,255,0),1)
                    copy_img2 = cv2.circle(copy_img2,(x,y),3,(255,0,0),-1)
                    finalimg = bitnot[x:x+h, y:y+w]
                    if k == 4:
                        cv2.imwrite("../Images/" + title + "_finalimg.png", finalimg)
                #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                    border = cv2.copyMakeBorder(finalimg,2,2,2,2,   cv2.BORDER_CONSTANT,value=[255,255])
                #     resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                #     dilation = cv2.dilate(resizing, kernel,iterations=1)
                #     erosion = cv2.erode(dilation, kernel,iterations=1)

                    
                #     out = pytesseract.image_to_string(erosion)
                #     if(len(out)==0):
                #         out = pytesseract.image_to_string(erosion, config='--psm 3')
                #     inner = inner +" "+ out
                # outer.append(inner)
    cv2.imwrite("../Images/" + title + "_finalbox.png", copy_img2)
    
    print('cell/box the strings: '+str(len(outer)))
    #Creating a dataframe of the generated OCR list
    # arr = np.array(outer)
    # dataframe = pd.DataFrame(arr.reshape(len(row),countcol))
    # print(dataframe)
    # data = dataframe.style.set_properties(align="left")
    # data.to_excel("../Images/output.xlsx")
    """

"""
def get_cropped_image(image, x, y, w, h):
    cropped_image = image[ y:y+h , x:x+w ]
    return cropped_image
    
def get_ROI(image, horizontal, vertical, left_line_index, right_line_index, top_line_index, bottom_line_index, offset=4):
    x1 = vertical[left_line_index][2] + offset
    y1 = horizontal[top_line_index][3] + offset
    x2 = vertical[right_line_index][2] - offset
    y2 = horizontal[bottom_line_index][3] - offset
    
    w = x2 - x1
    h = y2 - y1
    
    cropped_image = get_cropped_image(image, x1, y1, w, h)
    
    return cropped_image, (x1, y1, w, h)
"""

def main(argv):
    
    default_file = '../Images/source6.png'
    filename = argv[0] if len(argv) > 0 else default_file
    
    src = cv2.imread(cv2.samples.findFile(filename))
    
    # Loads an image
    horizontal, vertical = detect_lines(src, display=True)
    
    return 0
    
if __name__ == "__main__":
    main(sys.argv[1:])
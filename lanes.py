import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny(image):
	gray  = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
	blur  = cv2.GaussianBlur(gray,(5,5),0)
	canny = cv2.Canny(blur,50,150)
	return canny

def region_of_interest(image):
	height = image.shape[0]
	polygons = np.array([
		[(200,height),(1100,height),(550,250)]
		])
	mask = np.zeros_like(image)
	cv2.fillPoly(mask,polygons,255)
	masked_image = cv2.bitwise_and(image,mask)
	return masked_image

def display_lines(image,lines):
	line_image = np.zeros_like(image)
	if lines is not None:
		for line in lines:
			x1,y1,x2,y2  = line.reshape(4)
			cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),13) #line_image is black image, the 13 is thickness of blue line
	return line_image


def average_slope_intercept(images,lines):
	left_fit = []
	right_fit = []
	for line in lines:
		x1,y1,x2,y2 = line.reshape(4)
		parameters = np.polyfit((x1,x2),(y1,y2),1) # polyfit fits a line through these points.the 1 is degree of polynomial to get a linear line.
		slope = parameters[0]
		intercept = parameters[1]
		# To check  if the line is left or right line.left line y value decreases as x inc. reverse true for right line.
		if slope<0: # if slope is neg. 
			left_fit.append((slope,intercept))
		else:
			right_fit.append((slope,intercept))

	left_fit_average = np.average(left_fit,axis=0) # axis=0 cause [slope,intercept] as multiple rows. so want to sum over the rows not columns
	right_fit_average = np.average(right_fit,axis=0)
	left_line = make_coordinates(images,left_fit_average)
	right_line = make_coordinates(images,right_fit_average)
	return np.array([left_line,right_line])

def make_coordinates(image,line_parameters):
	slope,intercept = line_parameters
	# print(image.shape) the shape of image is (704,1279,3). thus height(y) is 704
	y1 = image.shape[0]
	y2 = int(y1*(3/5)) # 704 * 3/5 = 422. thus starts from 704 at the bottom to upwards till 422.
	# on rearranging the variables set x1 as:
	x1 = int( (y1 - intercept)/slope)  # but the horizontal values(x) are dependent on the slope and intercept. hence doing this.
	x2 = int( (y2 - intercept) /slope)
	return np.array([x1,y1,x2,y2])

 

# working on image:
#image = cv2.imread('test_image.jpg')
#lane_image = np.copy(image)
#canny_image  = canny(lane_image)
#cropped_image = region_of_interest(canny_image)
#lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
#averaged_lines = average_slope_intercept(lane_image,lines)
#line_image = display_lines(lane_image,averaged_lines)
#combo_image = cv2.addWeighted(lane_image,0.8,line_image,1,1)

#plt.imshow(canny)
#plt.show()

#cv2.imshow('result',combo_image)
#cv2.waitKey(0)

# for video:   
cap = cv2.VideoCapture('test2.mp4')  # same code as image. ONLY change lane_image into frame.
while (cap.isOpened()):
	n,frame = cap.read()  # n is a boolean value(not useful now)
	canny_image  = canny(frame)
	cropped_image = region_of_interest(canny_image)
	lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
	averaged_lines = average_slope_intercept(frame,lines)
	line_image = display_lines(frame,averaged_lines)
	combo_image = cv2.addWeighted(frame,0.8,line_image,1,1)
	cv2.imshow('result',combo_image) # shows frame by frame
	if cv2.waitKey(1) & 0xFF ==ord('q'): # the video breaks on pressing 'q' char. the 0xFF is to convert the int into binary. for cross platoform compatibality.
		break
cap.release()
cv2.destroyAllWindows()
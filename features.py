import numpy as np
import cv2
import csv
import math

def detect_lines(font, images,testdata, threshold=20, min_line_length=20, max_line_gap=10):
    for image in images:
        
        edges = cv2.Canny(image, 20, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
        max_diff = 0
        # if lines is not None:
            # for line in lines:
            #     x1, y1, x2, y2 = line[0]
            #     max_diff = max(max_diff, np.abs(y1 - y2))
        degrees = []
        if lines is not None:
            y1_values = lines[:, 0, 1]
            y2_values = lines[:, 0, 3]
            absolute_diff = np.abs(y1_values - y2_values)
            max_diff = np.max(absolute_diff)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) < abs(y2 - y1):
                    slope = (y1 - y2) / (x2 - x1 + 0.00001)
                    angle_rad = math.atan(slope)  # Calculate angle in radians
                    angle_deg = math.degrees(angle_rad)  # Convert angle to degrees
                    if np.abs(angle_deg) > 75 and np.abs(y1 - y2) >max_diff-50:
                        degrees.append(np.abs(angle_deg))
                        # cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.imshow("Image", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            if degrees!=[]:
                testdata.append([np.mean(degrees),font])
                print([np.mean(degrees),font])

    # return testdata
        # Show the image with detected vertical lines
        # if degrees:
        #     print(np.mean(degrees))

        #     with open("output.csv", 'a', newline='') as csvfile:
        #         writer = csv.writer(csvfile)
        #         writer.writerow([np.mean(degrees)])
        # cv2.imshow('Vertical Lines', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
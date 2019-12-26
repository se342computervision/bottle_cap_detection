import  cv2
import numpy as np



# input must be a matrix of (H,W,BGR)
# output is a bool indicating if cap opens upward
def check_if_cap_open_upward(img):


    img = cv2.GaussianBlur(img,(3,3),1)
    edges = cv2.Canny(img, 100, 50, apertureSize = 3)
    result = img.copy()
    edges = cv2.GaussianBlur(edges,(3,3),0)

    minLineLength = 50
    maxLineGap = 50
    lines = cv2.HoughLinesP(edges,1,np.pi/180,20 ,minLineLength,maxLineGap)
    
    for i in range(lines.shape[0]):
        for x1,y1,x2,y2 in lines[i]:
            print(lines[i])
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)


    
    cv2.imshow('Canny', edges )
    cv2.imshow('img', img)
    cv2.waitKey(0)

    return True







if __name__ == "__main__":
    print('start')
    # img = cv2.imread(r'C:\Users\maze1\Desktop\side\DSC02617-7-1028-576.jpg')
    img = cv2.imread(r'C:\Users\maze1\Desktop\side\DSC02617-0-2724-2972.jpg')
    print(img)
    print(check_if_cap_open_upward(img))
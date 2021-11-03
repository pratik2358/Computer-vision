import numpy as np
import cv2 # You must not use any methods which has 'hough' in it!
from utils import  hough_peaks




def hough_lines_vote_acc(edge_img, rho_res=1, thetas= np.arange(0,180)):
    h,w=edge_img.shape[:2]
    d=int(np.round(np.sqrt(h**2+w**2)))
    y,x=np.nonzero(edge_img>0)
    A=np.zeros((2*d+1,len(thetas)))
    thetas1=np.deg2rad(thetas)
    rhos=np.arange(0,2*d)
    #rhos=[]
   
    for i in range(len(x)):
        for theta in thetas1:
            rho=np.int(np.ceil(x[i]*np.cos(theta)+y[i]*np.sin(theta)))
            A[rho,np.int(np.round(np.rad2deg(theta)))]+=1
            #rhos.append(rho)
           
    return A, thetas, rhos

    
def hough_circles_vote_acc(edge_img, radius):
    r=radius
    h,w=edge_img.shape[:2]
    edges=edge_img>0
    y_idxs,x_idxs=np.nonzero(edges)
    A=np.zeros((h,w))
    theta=np.linspace(-180,180,360)
    for i in range(len(x_idxs)):
        x=x_idxs[i]
        y=y_idxs[i]
        for t in theta:
            b=y-r*np.sin(t)
            a=x+r*np.cos(t)
            if a<h and a>0 and b<w and b>0:
                A[int(b),int(a)]+=1
            
    
    return A


def find_circles(edge_img, radius_range=[1,2], threshold=100, nhood_size=10):
    """
      A naive implementation of the algorithm for finding all the circles in a range.
      Feel free to write your own more efficient method [Extra Credit]. 
      For extra credit, you may need to add additional arguments. 


      Args
      - edge_img: numpy nd-array of dim (m, n). 
      - radius_range: range of radius. All cicles whose radius falls 
      in between should be selected.
      - nhood_size: size of the neighborhood from where only one candidate can be chosen. 
      
      Returns
      - centers, and radii i.e., (x, y) coordinates for each circle.

      HINTS:
      - I encourage you to use this naive version first. Just be aware that
       it may take a long time to run. You will get EXTRA CREDIT if you can write a faster
       implementaiton of this method, keeping the method signature (input and output parameters)
       unchanged. 
    """
    n = radius_range[1] - radius_range[0]
    H_size = (n,) + edge_img.shape
    H = np.zeros(H_size, dtype=np.uint)
    centers = ()
    radii = np.arange(radius_range[0], radius_range[1])
    valid_radii = np.array([], dtype=np.uint)
    num_circles = 0
    for i in range(len(radii)):
        H[i] = hough_circles_vote_acc(edge_img, radii[i])
        peaks = hough_peaks(H[i], numpeaks=10, threshold=threshold,
                            nhood_size=nhood_size)
        if peaks.shape[0]:
            valid_radii = np.append(valid_radii, radii[i])
            centers = centers + (peaks,)
            for peak in peaks:
                cv2.circle(edge_img, tuple(peak[::-1]), radii[i]+1, (0,0,0), -1)
        #  cv2.imshow('image', edge_img); cv2.waitKey(0); cv2.destroyAllWindows()
        num_circles += peaks.shape[0]
        print('Progress: %d%% - Circles: %d\033[F\r'%(100*i/len(radii), num_circles))
    print('Circles detected: %d          '%(num_circles))
    centers = np.array(centers)
    return centers, valid_radii.astype(np.uint)

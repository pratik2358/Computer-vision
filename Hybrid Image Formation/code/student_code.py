import numpy as np
#### DO NOT IMPORT cv2 

def my_imfilter(image, filter):
    
    """
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (k, k)
  Returns
  - filtered_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You may not use any libraries that do the work for you. Using numpy to work
   with matrices is fine and encouraged. Using opencv or similar to do the
   filtering for you is not allowed.
  - I encourage you to try implementing this naively first, just be aware that
   it may take an absurdly long time to run. You will need to get a function
   that takes a reasonable amount of time to run so that I can verify
   your code works.
  - Remember these are RGB images, accounting for the final image dimension.
  """

    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1
  

  ############################
  ### TODO: YOUR CODE HERE ###
    width=filter.shape[1]
    height=filter.shape[0]
    width=(width)//2
    height=height//2
    if (len(image.shape)==3):
        image_cp0=np.pad(image[:,:,0],((height,height),(width,width)),'reflect')
        image_cp1=np.pad(image[:,:,1],((height,height),(width,width)),'reflect')
        image_cp2=np.pad(image[:,:,2],((height,height),(width,width)),'reflect')
    
        imagest=np.dstack([image_cp0,image_cp1,image_cp2])
   
        filter=filter[::-1,::-1]
        h,w=imagest.shape[:2]
        n=filter.shape[0]//2
        m=filter.shape[1]//2
        filtered_image = np.zeros(imagest.shape)
    
        for i in range(n,h-n):
            for j in range(m,w-m):
                sum0=np.sum(filter*imagest[i-n:i+n+1,j-m:j+m+1,0])
                filtered_image[i,j,0]=sum0
                sum1=np.sum(filter*imagest[i-n:i+n+1,j-m:j+m+1,1])
                filtered_image[i,j,1]=sum1
                sum2=np.sum(filter*imagest[i-n:i+n+1,j-m:j+m+1,2])
                filtered_image[i,j,2]=sum2
    else:
        imagest=np.pad(image,((height,height),(width,width)),'reflect')
        
        filter=filter[::-1,::-1]
        h,w=imagest.shape[:2]
        n=filter.shape[0]//2
        m=filter.shape[1]//2
        filtered_image = np.zeros(imagest.shape)
        
        for i in range(n,h-n):
            for j in range(m,w-m):
                sum0=np.sum(filter*imagest[i-n:i+n+1,j-m:j+m+1])
                filtered_image[i,j]=sum0

  ### END OF STUDENT CODE ####
  ############################

    return filtered_image[n:h-n,m:w-m]

def create_hybrid_image(image1, image2, filter):
    """
    Takes two images and creates a hybrid image. Returns the low
    frequency content of image1, the high frequency content of
    image 2, and the hybrid image.

    Args
    - image1: numpy nd-array of dim (m, n, c)
    - image2: numpy nd-array of dim (m, n, c)
    Returns
    - low_frequencies: numpy nd-array of dim (m, n, c)
    - high_frequencies: numpy nd-array of dim (m, n, c)
    - hybrid_image: numpy nd-array of dim (m, n, c)

    HINTS:
    - You will use your my_imfilter function in this function.
    - You can get just the high frequency content of an image by removing its low
      frequency content. Think about how to do this in mathematical terms.
    - Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
      as 'clipping'.
    - If you want to use images with different dimensions, you should resize them
      in the notebook code.
    """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]

  ############################
  ### TODO: YOUR CODE HERE ###
    

    
    low_frequencies=my_imfilter(image1, filter)
    high_frequencies=image2-my_imfilter(image2, filter)
    hybrid_image=(low_frequencies+high_frequencies)/2

  ### END OF STUDENT CODE ####
  ############################

    
    return low_frequencies, high_frequencies, hybrid_image
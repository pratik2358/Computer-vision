import numpy as np


def calculate_projection_matrix(points_2d, points_3d):
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points:

                                                      [ M11      [ u1
                                                        M12        v1
                                                        M13        .
                                                        M14        .
    [ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1        M21        .
      0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1        M22        .
      .  .  .  . .  .  .  .    .     .      .       *   M23   =    .
      Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn        M24        .
      0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]      M31        .
                                                        M32        un
                                                        M33 ]      vn ]

    Then you can solve this using least squares with np.linalg.lstsq() or SVD.
    Notice you obtain 2 equations for each corresponding 2D and 3D point
    pair. To solve this, you need at least 6 point pairs.

    Args:
    -   points_2d: A numpy array of shape (N, 2)
    -   points_2d: A numpy array of shape (N, 3)

    Returns:
    -   M: A numpy array of shape (3, 4) representing the projection matrix
    """

    # Placeholder M matrix. It leads to a high residual. Your total residual
    # should be less than 1.
#     M = np.asarray([[0.1768, 0.7018, 0.7948, 0.4613],
#                     [0.6750, 0.3152, 0.1136, 0.0480],
#                     [0.1020, 0.1725, 0.7244, 0.9932]])

    ###########################################################################
    # TODO: YOUR PROJECTION MATRIX CALCULATION CODE HERE
    arr = np.column_stack((points_3d, [1]*points_3d.shape[0]))
    A1 = np.concatenate((arr, np.zeros_like(arr)), axis=1).reshape((-1, 4))
    A2 = np.concatenate((np.zeros_like(arr), arr), axis=1).reshape((-1, 4))
    A3 = -np.multiply(np.tile(points_2d.reshape((-1, 1)), 4), arr.repeat(2, axis=0))
    A = np.concatenate((A1, A2, A3), axis=1)
    U, s, V = np.linalg.svd(A)
    M = V[-1]
    M = M.reshape((3, 4))
    ###########################################################################

#     raise NotImplementedError('`calculate_projection_matrix` function in ' +
#         '`student_code.py` needs to be implemented')

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return M

def calculate_camera_center(M):
    """
    Returns the camera center matrix for a given projection matrix.

    The center of the camera C can be found by:

        C = -Q^(-1)m4

    where your project matrix M = (Q | m4).

    Args:
    -   M: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """

    # Placeholder camera center. In the visualization, you will see this camera
    # location is clearly incorrect, placing it in the center of the room where
    # it would not see all of the points.
    cc = np.asarray([1, 1, 1])

    ###########################################################################
    # TODO: YOUR CAMERA CENTER CALCULATION CODE HERE
    Q = np.split(M, [3], axis=1)
    cc= -np.squeeze(np.linalg.solve(Q[0],Q[1]))
    ###########################################################################

#     raise NotImplementedError('`calculate_camera_center` function in ' +
#         '`student_code.py` needs to be implemented')

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return cc

def estimate_fundamental_matrix(points_a, points_b):
    """
    Calculates the fundamental matrix. Try to implement this function as
    efficiently as possible. It will be called repeatedly in part 3.

    You must normalize your coordinates through linear transformations as
    described on the project webpage before you compute the fundamental
    matrix.

    Args:
    -   points_a: A numpy array of shape (N, 2) representing the 2D points in
                  image A
    -   points_b: A numpy array of shape (N, 2) representing the 2D points in
                  image B

    Returns:
    -   F: A numpy array of shape (3, 3) representing the fundamental matrix
    """

    # Placeholder fundamental matrix
    F = np.asarray([[0, 0, -0.0004],
                    [0, 0, 0.0032],
                    [0, -0.0044, 0.1034]])

    ###########################################################################
    # TODO: YOUR FUNDAMENTAL MATRIX ESTIMATION CODE HERE
    ###########################################################################

#     raise NotImplementedError('`estimate_fundamental_matrix` function in ' +
#         '`student_code.py` needs to be implemented')
    
    
    arr_a = np.column_stack((points_a, [1]*points_a.shape[0]))
    arr_b = np.column_stack((points_b, [1]*points_b.shape[0]))

    arr_a = np.tile(arr_a, 3)
    arr_b = arr_b.repeat(3, axis=1)
    A = np.multiply(arr_a, arr_b)

    
    U, s, V = np.linalg.svd(A)
    F_matrix = V[-1]
    F_matrix = np.reshape(F_matrix, (3, 3))

    
    '''Resolve det(F) = 0 constraint using SVD'''
    U, S, Vh = np.linalg.svd(F_matrix)
    S[-1] = 0
    F_matrix = U @ np.diagflat(S) @ Vh

    return F_matrix

    
    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    #return F

def estimate_fundamental_matrix_with_normalize(Points_a, Points_b):
    # Try to implement this function as efficiently as possible. It will be
    # called repeatly for part III of the project
    #

    #                                              [f11
    # [u1u1' v1u1' u1' u1v1' v1v1' v1' u1 v1 1      f12     [0
    #  u2u2' v2v2' u2' u2v2' v2v2' v2' u2 v2 1      f13      0
    #  ...                                      *   ...  =  ...
    #  ...                                          ...     ...
    #  unun' vnun' un' unvn' vnvn' vn' un vn 1]     f32      0]
    #                                               f33]
    assert Points_a.shape[0] == Points_b.shape[0]
    
    mean_a = Points_a.mean(axis=0)
    mean_b = Points_b.mean(axis=0)
    std_a = np.sqrt(np.mean(np.sum((Points_a-mean_a)**2, axis=1), axis=0))
    std_b = np.sqrt(np.mean(np.sum((Points_b-mean_b)**2, axis=1), axis=0))

    Ta1 = np.diagflat(np.array([np.sqrt(2)/std_a, np.sqrt(2)/std_a, 1]))
    Ta2 = np.column_stack((np.row_stack((np.eye(2), [[0, 0]])), [-mean_a[0], -mean_a[1], 1]))

    Tb1 = np.diagflat(np.array([np.sqrt(2)/std_b, np.sqrt(2)/std_b, 1]))
    Tb2 = np.column_stack((np.row_stack((np.eye(2), [[0, 0]])), [-mean_b[0], -mean_b[1], 1]))

    Ta = np.matmul(Ta1, Ta2)
    Tb = np.matmul(Tb1, Tb2)

    arr_a = np.column_stack((Points_a, [1]*Points_a.shape[0]))
    arr_b = np.column_stack((Points_b, [1]*Points_b.shape[0]))

    arr_a = np.matmul(Ta, arr_a.T)
    arr_b = np.matmul(Tb, arr_b.T)

    arr_a = arr_a.T
    arr_b = arr_b.T

    arr_a = np.tile(arr_a, 3)
    arr_b = arr_b.repeat(3, axis=1)
    A = np.multiply(arr_a, arr_b)

   
    U, s, V = np.linalg.svd(A)
    F_matrix = V[-1]
    F_matrix = np.reshape(F_matrix, (3, 3))
    F_matrix /= np.linalg.norm(F_matrix)


    '''Resolve det(F) = 0 constraint using SVD'''
    U, S, Vh = np.linalg.svd(F_matrix)
    S[-1] = 0
    F_matrix = U @ np.diagflat(S) @ Vh

    F_matrix = Tb.T @ F_matrix @ Ta
    return F_matrix    

def ransac_fundamental_matrix(matches_a, matches_b):
    """
    Find the best fundamental matrix using RANSAC on potentially matching
    points. Your RANSAC loop should contain a call to
    estimate_fundamental_matrix() which you wrote in part 2.

    If you are trying to produce an uncluttered visualization of epipolar
    lines, you may want to return no more than 100 points for either left or
    right images.

    Args:
    -   matches_a: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from image A
    -   matches_b: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from image B
    Each row is a correspondence (e.g. row 42 of matches_a is a point that
    corresponds to row 42 of matches_b)

    Returns:
    -   best_F: A numpy array of shape (3, 3) representing the best fundamental
                matrix estimation
    -   inliers_a: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from image A that are inliers with
                   respect to best_F
    -   inliers_b: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from image B that are inliers with
                   respect to best_F
    """

    # Placeholder values
    best_F = estimate_fundamental_matrix(matches_a[:10, :], matches_b[:10, :])
    inliers_a = matches_a[:100, :]
    inliers_b = matches_b[:100, :]

    ###########################################################################
    # TODO: YOUR RANSAC CODE HERE
    ###########################################################################

#     raise NotImplementedError('`ransac_fundamental_matrix` function in ' +
#         '`student_code.py` needs to be implemented')
    
    
    num_iterator = 10000
    threshold = 0.002
    best_F_matrix = np.zeros((3, 3))
    max_inlier = 0
    num_sample_rand = 8

    xa = np.column_stack((matches_a, [1]*matches_a.shape[0]))
    xb = np.column_stack((matches_b, [1]*matches_b.shape[0]))
    xa = np.tile(xa, 3)
    xb = xb.repeat(3, axis=1)
    A = np.multiply(xa, xb)

    for i in range(num_iterator):
        index_rand = np.random.randint(matches_a.shape[0], size=num_sample_rand)
        F_matrix = estimate_fundamental_matrix_with_normalize(matches_a[index_rand, :], matches_b[index_rand, :])
        err = np.abs(np.matmul(A, F_matrix.reshape((-1))))
        current_inlier = np.sum(err <= threshold)
        if current_inlier > max_inlier:
            best_F_matrix = F_matrix.copy()
            max_inlier = current_inlier

    err = np.abs(np.matmul(A, best_F_matrix.reshape((-1))))
    index = np.argsort(err)
    # print(best_F_matrix)
    # print(np.sum(err <= threshold), "/", err.shape[0])
    return best_F_matrix, matches_a[index[:29]], matches_b[index[:29]]
    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    #return best_F, inliers_a, inliers_b

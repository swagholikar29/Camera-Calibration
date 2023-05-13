import numpy as np
import cv2
import argparse
import glob
from scipy.spatial.transform import Rotation as scipyRot
from scipy.optimize import least_squares

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def convert_A_matrix_to_vector(A):
    return np.array([A[0,0],A[0,1],A[1,1],A[0,2],A[1,2]])

def convert_A_vector_to_matrix(a):
    alpha,gamma,beta,u0,v0 = a
    A1 = [alpha,gamma,u0]
    A2 = [  0  ,beta ,v0]
    A3 = [  0  ,  0  , 1]

    A  = np.vstack((A1,A2,A3)) # 3 x 3
    return A

def dissect_x_vector(x,n_imgs):
    A = convert_A_vector_to_matrix(x[0:5])
    k1 = x[5]
    k2 = x[6]
    transformations = x[7:].reshape(n_imgs,6)
    return A,k1,k2,transformations

def package_x_vector(A,k1,k2,transformations):
    a  = convert_A_matrix_to_vector(A) # 5,
    x0 = np.concatenate((a,k1,k2,transformations)) # 83,
    return x0

def draw_circles(img,base_path,name,m_new,m_old):
    for x,y,_ in m_new.T:
        cv2.circle(img,(int(x),int(y)),0,(255,0,0),30)
    for x,y in m_old:
        cv2.circle(img,(int(x),int(y)),0,(0,255,0),15)
    cv2.imwrite(f"{base_path}/rectified/{name}.png",img)

def get_images(base_path,input_extn):
    img_files = glob.glob(f"{base_path}/*{input_extn}",recursive=False)
    img_names = [img_file.replace(f"{base_path}/",'').replace(f"{input_extn}",'') for img_file in img_files]
    imgs = [cv2.imread(img_file) for img_file in img_files]
    return imgs,img_names

def get_chessboard_corners(img_color,pattern_size,name,args):
    if args.debug:
        cv2.imshow(name,img_color)

    img_gray = cv2.cvtColor(img_color,cv2.COLOR_RGB2GRAY)
    if args.debug:
        cv2.imshow(f"{name}_gray",img_gray)

    chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH     \
                        + cv2.CALIB_CB_NORMALIZE_IMAGE  \
                        + cv2.CALIB_CB_FAST_CHECK       
    ret, corners = cv2.findChessboardCorners(img_gray,pattern_size,flags=chessboard_flags)
    # TODO cv2.cornerSubPix
    if not ret:
        print(f"something went wrong while processing {name}")
        exit(1)

    if args.display:
        chessboard_img = cv2.drawChessboardCorners(img_color,pattern_size,corners,ret)
        cv2.imshow(f"{name}_chessboard",chessboard_img)

    corners = corners.reshape((corners.shape[0],-1))
    return corners

def get_world_corners(pattern_size,square_size):
    """
    description:
        returns world corners for a given pattern size and square size(mm)
    input:
        pattern_size - tuple (2)
        square_size - scalar (mm)
    output:
        world_corners - pattern_size[0]*pattern_size[1] x 2
    """
    x_lin = np.arange(0,pattern_size[0],1)
    y_lin = np.arange(0,pattern_size[1],1)
    x_grid, y_grid = np.meshgrid(x_lin,y_lin)
    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()
    world_corners = np.vstack((x_grid,y_grid)).T
    world_corners = world_corners*square_size
    return world_corners

def get_V_mat_element(H,i,j):
    """
    description:
        calculate element of v vector from homography
    input:
        H - homography matrix 3 x 3
        i - according the paper convention
        j - according the paper convention
    output:
        v - 6 x 1 according to the paper
    """
    # convering indices from paper convention to numpy convention
    i = i - 1
    j = j - 1

    # calculation vector v for a given homography
    v1 = H[0][i]*H[0][j]
    v2 = H[0][i]*H[1][j] + H[1][i]*H[0][j]
    v3 = H[1][i]*H[1][j]                  
    v4 = H[2][i]*H[0][j] + H[0][i]*H[2][j]
    v5 = H[2][i]*H[1][j] + H[1][i]*H[2][j]
    v6 = H[2][i]*H[2][j]
    v  = np.vstack((v1,v2,v3,v4,v5,v6))
    return v

def get_V_mat(H):
    """
    description:
        calculate V for a given homography
    input:
        H - homography matrix 3 x 3
    output:
        V - 2 x 6 V matrix according to the paper
    """
    V1  = get_V_mat_element(H,1,2) # 6 x 1
    V1  = V1.T # 1 x 6
    V20 = get_V_mat_element(H,1,1) # 6 x 1
    V20 = V20.T # 1 x 6
    V21 = get_V_mat_element(H,2,2) # 6 x 1
    V21 = V21.T # 1 x 6
    V2  = V20 - V21 # 1 x 6
    V   = np.vstack((V1,V2)) # 2 x 6

    return V
def get_L_mat(img_corner,world_corner):
    """
    description:
        calculate L for a given img_corner and world_corner
    input:
        image_corner  - 2,
        world_corners - 3,
    output:
        L - as per paper convention 2 x 9
    """
    L1 = np.hstack((world_corner, np.zeros((3)), -img_corner[0]*world_corner))
    L2 = np.hstack((np.zeros((3)), world_corner, -img_corner[1]*world_corner))
    L  = np.vstack((L1,L2))
    return L

def get_homography(img_corners,world_corners,name):
    world_corners = np.hstack((world_corners,np.ones((world_corners.shape[0],1))))

    L = tuple([get_L_mat(img_corner,world_corner) for img_corner,world_corner in zip(img_corners,world_corners)]) # 2 x 9
    L = np.vstack(L) # 2*N x 9

    eig_val,eig_vec = np.linalg.eig(L.T @ L)
    min_eig_vec_ind = np.argmin(eig_val) # 1 x 1
    min_eig_vec     = eig_vec[:,min_eig_vec_ind] # 6 x 1

    h1 = min_eig_vec[0:3]
    h2 = min_eig_vec[3:6]
    h3 = min_eig_vec[6:9]

    H = np.vstack((h1,h2,h3))
    H = H/H[2,2]
    # TODO optimize using LV MINPACK

    return H

def get_camera_intrinsic_from_b(b):
    """
    description:
        return camera intrinsics given b vector from paper
    input:
        b - vector as per convention from paper
    output:
        camera intrinsic matrix 3 x 3
            | alpha gamma u0 |
        A - |   0   beta  v0 |
            |   0     0    1 |
    """
    # TODO #1 could the divisions below fail?

    B11 = b[0] 
    B12 = b[1] 
    B22 = b[2] 
    B13 = b[3] 
    B23 = b[4] 
    B33 = b[5] 

    v0_num = B12*B13 - B11*B23
    v0_den = B11*B22 - B12*B12
    v0     = v0_num/v0_den #TODO #1

    lamda1_num  =  B13*B13 + v0*(B12*B13 - B11*B23)
    lamda = B33 - lamda1_num/B11 #TODO #1

    alpha = (lamda/B11)**(0.5) #TODO #1

    beta_num = lamda*B11
    beta_den = B11*B22 - B12*B12
    beta  = (beta_num/beta_den)**(0.5) #TODO #1

    gamma = (-B12*alpha*alpha*beta)/lamda

    u00 = (gamma*v0)/beta
    u01 = (B13*alpha*alpha)/lamda
    u0  = u00 - u01

    A  = convert_A_vector_to_matrix([alpha,gamma,beta,u0,v0])
    return A, lamda
    

def get_camera_intrinsics(homography_list):
    """
    description:
        calculate camera intrinsics based on the paper 3.1
    input:
        homography_list - list of size N homography matrices 3 x 3
    output:
        A - camera intrinsic matrix 3 x 3
    """
    V = tuple([get_V_mat(H) for H in homography_list]) # N, 2 x 6
    V = np.vstack(V) # (2*N) x 6
    M = V.T @ V # 6 x 6
    U,sigma,R = np.linalg.svd(V)
    eig_val,eig_vec = np.linalg.eig(V.T @ V) # 6 x 6
    min_eig_vec_ind = np.argmin(eig_val) # 1 x 1
    min_eig_vec     = eig_vec[:,min_eig_vec_ind] # 6 x 1
    #print(f"eig_vec:{min_eig_vec}")
    A = get_camera_intrinsic_from_b(min_eig_vec) # 3 x 3
    return A

def get_transformation_mat(A,lamda,H):
    """
    description:
        calculate rotation and translation matrices for each image
    input:
    output:
    """
    A_inv = np.linalg.inv(A) # should be perfectly invertible
    lamda1 = 1/np.linalg.norm(A_inv @ H[:,0],ord=2)
    lamda2 = 1/np.linalg.norm(A_inv @ H[:,1],ord=2)

    r1 = lamda1*A_inv @ H[:,0] # 3 x 1
    r2 = lamda1*A_inv @ H[:,1] # 3 x 1
    r3 = skew(r1) @ r2 # 3 x 1

    t  = lamda1*A_inv @ H[:,2] # 3 x 1

    R  = np.vstack((r1,r2,r3)).T
    r  = scipyRot.from_matrix(R).as_mrp() # 3,

    rt = np.concatenate((r,t.flatten()),axis=0).tolist() # 6,
    return rt

def distort_corners(x_c,k1,k2):
    """
    description:
        distort corners based on the k1 and k2
    input:
        x_c - 3 x M
        k1 - distortion coefficient 1
    output:
        x_d - 3 x M
    """
    x_c   = x_c/x_c[2]
    r_c2   = x_c[0]**2 + x_c[1]**2
    factor = k1*r_c2 + k2*(r_c2**2)
    x_h   = x_c + x_c*factor
    x_h[2] = 1
    return x_h

def get_projection(x,A,k1,k2,world_corners):
    R  = scipyRot.from_mrp(x[0:3]).as_matrix() # 3 x 3
    t  = x[3:6].reshape((3,1)) # 3 x 1
    T  = np.hstack((R,t)) # 3 x 4
    
    M = world_corners.shape[0]
    zeros = np.zeros((M,1))
    ones  = np.ones((M,1))
    world_corners = np.hstack((world_corners,zeros,ones)).T # 4 x M

    x_c   = T @ world_corners # 3 x 4 @ 4 x M = 3 x M
    x_h   = distort_corners(x_c,k1,k2) # 3 x M
    m_hat = A @ x_h # 3 x 3 @ 3 x M =  3 x M
    m_hat = m_hat/m_hat[2]
    return m_hat

def projection_error(transformation,A,k1,k2,img_corners,world_corners):
    """
    description:
        computes projection error for an image
    input:
        x - 6, vector of all transformation parameters
        k1,k2 - distortion coefficients
        img_corners - M x 2 
        world_corners - M x 2
    output:
        residuals - 14,1
    """
    m_hat = get_projection(transformation,A,k1,k2,world_corners)
    M = world_corners.shape[0]
    ones  = np.ones((M,1))
    img_corners = np.hstack((img_corners,ones)).T # 3 x M

    error = img_corners - m_hat
    error = np.linalg.norm(error,axis=0,ord=2)
    return error

def get_projections(transformations,A,k1,k2,world_corners):
    projections = []
    for transformation in transformations:
        projections.append(get_projection(transformation,A,k1,k2,world_corners))
    return projections

def compute_residuals(x,imgs_corners,world_corners,per_img=False):
    """
    description: 
        callable functional to calcuate residuals
    input:
        if N - number of images
            M - number of features per image
            nP - number of parameters required for transformation
                = 3(rotation rodrigues) + 3(translation)
        x - 5(intrinsics) + 2(distortion) + N*nP
        imgs_corners - N x M x 2
        world_corners - M x 2
    output:
        residuals - N*M*nP
    """

    n_imgs = len(imgs_corners)
    n_feats = len(world_corners)

    A,k1,k2,transformations = dissect_x_vector(x,n_imgs)

    errors= [] # N
    for i in range(n_imgs):
        error = projection_error(transformations[i,:],A,k1,k2,imgs_corners[i],world_corners) # M,
        if per_img:
            error = np.sum(error)
        errors.append(error)

    if not per_img:
        errors = np.concatenate(errors) # N*M,
    return errors

def filter_coords(u_grid_old,v_grid_old,u_grid,v_grid,img_shape):
    inds_max_v = np.argwhere(v_grid_old>img_shape[0]-1)
    inds_max_u = np.argwhere(u_grid_old>img_shape[1]-1)

    inds_max  = np.concatenate((inds_max_v,inds_max_u)).flatten().tolist()
    exclude_inds = list(set(tuple(inds_max)))
    exclude_inds = np.array(exclude_inds)
    if exclude_inds.shape[0] <=0:
        return u_grid_old,v_grid_old,u_grid,v_grid

    u_grid_old = np.delete(u_grid_old,exclude_inds, axis=0)
    v_grid_old = np.delete(v_grid_old,exclude_inds, axis=0)
    u_grid = np.delete(u_grid,exclude_inds, axis=0)
    v_grid = np.delete(v_grid,exclude_inds, axis=0)

    return u_grid_old,v_grid_old,u_grid,v_grid

def inverse_warp(img,A,k1,k2):
    """
    description:
        inverse warp the img based on the instrinsics and extrinsincs
    input:
        img - original img
        A - camera intrinsics
        k1 - distortion parameter 1
        k2 - distortion parameter 1
    output:
        rectified_img - image after rectification
    """
    u_lin = np.arange(0,img.shape[1],1)
    v_lin = np.arange(0,img.shape[0],1)
    u_grid,v_grid = np.meshgrid(u_lin,v_lin)
    u_grid = u_grid.flatten()
    v_grid = v_grid.flatten()

    alpha,gamma,beta,u0,v0 = convert_A_matrix_to_vector(A)
    res_u = ((u_grid - u0)/alpha)
    res_v = ((v_grid - v0)/beta)
    
    r_2 = res_u**2 + res_v**2
    r_4 = r_2**2
    factor = k1*r_2 + k2*r_4

    u_grid_old = u_grid + (u_grid - u0)*factor
    v_grid_old = v_grid + (v_grid - v0)*factor

    u_grid_old = u_grid_old.astype(int)
    v_grid_old = v_grid_old.astype(int)

    u_grid_old,v_grid_old,u_grid,v_grid = filter_coords(u_grid_old,v_grid_old,u_grid,v_grid,img.shape)

    new_img = np.zeros_like(img)
    new_img[v_grid,u_grid,:] = img[v_grid_old,u_grid_old,:]
    return new_img


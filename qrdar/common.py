import os
import pandas as pd
import numpy as np
import itertools

from sklearn.cluster import DBSCAN
from scipy.spatial import distance_matrix as distance_matrix
from scipy.optimize import curve_fit

import qrdar
from qrdar.io.pcd_io import *

# a bit of hack for Python 2.x
__dir__ = os.path.split(os.path.abspath(qrdar.__file__))[0]

def nn(arr):

    nbrs = NearestNeighbors(n_neighbors=3, algorithm='kd_tree').fit(arr)
    distances, indices = nbrs.kneighbors(arr)
    
    return np.unique(distances[:, 1])

def calculate_R(corners, template):

    stop = False
    
    for N in [4, 3]:
        
        if stop: break
        combinations = [c for c in itertools.combinations(corners.index, N)]
        all_R = []
        all_rmse = []
        all_tcombo = []
        all_combo = []
        exclude = []

        for combo in combinations:

            for t_combo in itertools.permutations(template.index, N):

                test = corners.loc[list(combo)].copy()
                test = test.sort_values(['x', 'y', 'z']).reset_index()
                if set(test.index) in exclude: continue
                t_pd = template.loc[list(t_combo)]

                all_combo.append(combo)
                R = rigid_transform_3D(test[['x', 'y', 'z']].values, t_pd) 
                all_R.append(R)
                test = apply_rotation(R, test)

                RMSE = np.sqrt((np.linalg.norm(test[['x', 'y', 'z']].values - t_pd.values, axis=1)**2).mean())
                all_rmse.append(RMSE)

                if np.any(np.array(all_rmse) < .015):
                    stop = True
                    break
            
    idx = np.where(np.array(all_rmse) == np.array(all_rmse).min())[0][0]
    
    return list(all_combo[idx]), all_R[idx], all_rmse[idx]

def apply_rotation(M, df):
    
    if 'a' not in df.columns:
        df.loc[:, 'a'] = 1
    
    r_ = np.dot(M, df[['x', 'y', 'z', 'a']].T).T
    df.loc[:, ['x', 'y', 'z']] = r_[:, :3]
    
    return df[['x', 'y', 'z']]

def rigid_transform_3D(A, B):
    
    """
    http://nghiaho.com/uploads/code/rigid_transform_3D.py_
    """
    
    assert len(A) == len(B)
    
    A = np.matrixlib.defmatrix.matrix(A)
    B = np.matrixlib.defmatrix.matrix(B)

    N = A.shape[0]; # total points

    centroid_A = np.mean(A, axis=0).reshape(1, 3)
    centroid_B = np.mean(B, axis=0).reshape(1, 3)
    
    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.transpose(AA) * BB

    U, S, Vt = np.linalg.svd(H)

    R = np.dot(Vt.T, U.T)
    
    t = -R*centroid_A.T + centroid_B.T
    
    M, N = np.identity(4), np.identity(4)
    M[:3, :3] = R
    N[:3, 3] = t.reshape(-1, 3)
    
    return np.dot(N, M)

def gauss(x, mu, sigma, A):
    
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

def calculate_cutoff(data, p):

    bins = np.linspace(np.floor(data.min()), np.ceil(data.max()))
    y, x = np.histogram(data, bins=bins)
    x = (x[1:] + x[:-1]) / 2 # for len(x)==len(y)
    
    expected = (np.percentile(bins, 0 + p), 1, len(data) / 2, np.percentile(bins, 100 - p), 1, len(data) / 2)
    params, cov = curve_fit(bimodal, x, y, expected, maxfev=10000)
    sigma = np.sqrt(np.diag(cov))
    
    return np.mean([params[0], params[3]])

def expected_distances(markerTemplate):
    
    # distances between points
    
    dist = set()
    for b in itertools.combinations(markerTemplate, 2):
        dist.add(np.around(np.linalg.norm(b[0] - b[1]), 2))
        
    return np.hstack([0, np.sort(np.array(list(dist)))])

def ensure_square_arr(df, var):
    
    X, Y = np.meshgrid(np.arange(0, 6), np.arange(0, 6))
    x_df = pd.DataFrame(np.vstack([Y.flatten(), X.flatten()]).T, columns=['xx', 'zz'])
        
    b_df = df.groupby(['xx', 'zz'])[var].mean().reset_index()
    img = pd.merge(x_df, b_df, on=['xx', 'zz'], how='left')
    img.loc[np.isnan(img[var]), var] = 0
    img = img[var].values.reshape(6, 6)
    img = img * np.pad(np.ones(np.array(img.shape) - 2), 1, 'constant') # force border to be black
    return img

def load_codes(dic):

    if dic == 'aruco_mip_16h3':
        return np.load(os.path.join(__dir__, 'aruco_mip_16h3_dict.npy'))
    
def template():

    markerTemplate = np.array([[ 0.      ,  0, 0.      ],
                               [ 0.182118,  0, 0.0381  ],
                               [ 0.      ,  0, 0.266446],
                               [ 0.131318,  0, 0.266446]])

    return pd.DataFrame(data=markerTemplate, columns=['x', 'y', 'z'])

def calculate_score(img, codes, N=6):
    
    size_of_code = float(N**2)
    size_of_inner = float((N-2)**2)
    size_diff = size_of_code - size_of_inner

    score = np.array([(np.rot90(img) == codes[:, :, j]).astype(int).sum() for j in range(codes.shape[2])])
    code = int(np.where(score == score.max())[0][0])
    confidence = ((size_of_inner - (size_of_inner - (score.max() - size_diff))) / size_of_inner)

    return code, confidence

def method_1(code):

    """
    method 1 calculates the reflectance threshold between white and black
    areas of the target and uses this to create a binary filter
    """
    
    code.loc[:, 'I_mean'] = code.groupby(['xx', 'zz']).intensity.transform(np.mean)


    for p in np.arange(5, 50, 5):
        try:
            C = calculate_cutoff(code.I_mean, p)
            if code.I_mean.min() < C < code.I_mean.max():
                break
        except:
            C = 0 # required if optimal parameters are not found

    code.loc[:, 'bw1'] = np.where(code.I_mean < C, 0, 1)
    img_1 = ensure_square_arr(code, 'bw1')

    return img_1


def method_2(code, threshold):
    
    """
    method 2 calculate the number of returns on a per grid square
    basis and determines how many are above a threshold creating
    a binary image e.g. if 70% had a reflectance value >7 db then
    this would be a white square
    
    RIEGL specific: todo expose intensity threshold
    """

    code.loc[:, 'N'] = code.groupby(['xx', 'zz']).x.transform(np.size)
    LN = code[(code.intensity < -7)].groupby(['xx', 'zz']).x.size().reset_index(name='LN')
    code = pd.merge(LN, code, on=['xx', 'zz'], how='outer')
    code.loc[:, 'P'] = code.LN / code.N

    code.loc[:, 'bw2'] = code.P.apply(lambda p: 0 if p > threshold else 1)
    img = ensure_square_arr(code, 'bw2')

    return img

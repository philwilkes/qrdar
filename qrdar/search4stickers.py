import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN

from qrdar.io.pcd_io import *
from qrdar.io.ply_io import *

pd.options.mode.chained_assignment = None  # default='warn'

def read(path, refl_field='intensity', refl_filter=0.):

    """
    Read in .pcd point cloud

    Parameters
    ----------
    path: str
        path to point cloud
    refl_field: str
        field containing reflectance / intensity values
    refl_filter: int or float
        value below which points are filtered

    Returns
    -------
    pc: pd.DataFrame
        input points filtered by reflectance value
    """

    # read in points and filter
    pc = read_pcd(path)
    pc = pc[pc[refl_field] >= refl_filter]
    
    return pc


def find(pc, sticker_size=.025, W=50, rgb=False, verbose=False):
    
    """
    Searches a point cloud for bright returns and clusters
    them into potential stickers.

    To improve efficiency, the point cloud is subdivided into
    n x n quadrants.


    Parameters
    ----------
    pc: pd.DataFrame with at least columns ['x', 'y', 'z']
        point cloud with bright points filtered
    sticker_size: float (default 0.025)
        diameter of stickers
    W: int (default 50)
        length of quadrant.
    rgb: boolean (default False)
        colours points according to cluster.
     
    Returns
    -------
    pc: pd.DataFrame 
        points are attributed with arbitary sticker cluster
        number
    """

    pc.loc[:, 'xx'] = (pc.x // W) * W
    pc.loc[:, 'yy'] = (pc.y // W) * W
    label_max = 0
    pc.loc[:, 'sticker_labels_'] = -1
    
    for tx, ty in pc.groupby(['xx', 'yy']).size().index:
       
        sq = pc[(pc.xx == tx) & (pc.yy == ty)]
        if len(sq) > 1e5: 
            raise Exception('more than 1 million points in a tile, reduce W'))
        if verbose: print('processing grid {} {} with length {}'.format(tx, ty, len(sq)))
              
        dbscan = DBSCAN(eps=sticker_size * 1.1, min_samples=2).fit(sq[['x', 'y', 'z']])
        idx = np.where(dbscan.labels_ > -1)[0]
        pc.loc[sq.loc[sq.index[idx]].index, 'sticker_labels_'] = dbscan.labels_[idx] + label_max
        label_max = pc.sticker_labels_.max() + 1 # iterative labelling hence 

    pc = pc[pc.sticker_labels_ != -1] # remove outlier points
           
    # points can be coloured by cluster for visualisation
    if rgb:
        RGB = pd.DataFrame({l:np.random.randint(0, high=255, size=3) for l in pc.sticker_labels_.unique()}).T
        RGB.columns = ['red', 'green', 'blue']
        pc = pd.merge(pc, RGB, left_on=pc.sticker_labels_, right_on=RGB.index.values, how='outer')
   
    return pc 

def filterBySize(pc, max_size=.05, min_size=0, verbose=False):

    """
    removes potential stickers that are too big e.g. reflective targets
    used for co-registration etc.

    Parameters
    ----------
    pc: pd.DataFrame with at least columns ['x', 'y', 'z', 'sticker_labels_']
        point cloud containing clustered potential stickers
    max_size: float
        size above which potential stickers are filtered
    verbose:
        print something
    
    
    Returns
    -------
    potential_dots: pd.DataFrame containing centres of potential stickerss

    """

    # group points in to potential stickets and estimate size and locations
    potential_dots = pc.groupby('sticker_labels_').agg({'x':(np.ptp, np.mean), 'y':(np.ptp, np.mean), 'z':(np.ptp, np.mean)})
    potential_dots.columns = ['y_ptp', 'y', 'x_ptp', 'x', 'z_ptp', 'z']
    N = len(potential_dots)
    if verbose: print("potential stickers found:", N) 
    
    # filter by size
    func = lambda row: np.max([row['x_ptp'], row['y_ptp'], row['z_ptp']])
    potential_dots.loc[:, 'max_ptp'] = potential_dots.apply(func, axis=1)
    potential_dots = potential_dots[potential_dots.max_ptp.between(min_size, max_size)]
    if verbose: print('stickers removed for being too large', N - len(potential_dots)) 
    if verbose: print('number of potential stickers:', len(potential_dots))

    return pc[pc.sticker_labels_.isin(potential_dots.index)] 


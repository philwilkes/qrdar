import pandas as pd
import numpy as np

from qrdar.common import *
from qrdar.io.pcd_io import *
from qrdar.io.ply_io import *

def extractFeatures(marker_df, tile_index, extract_tiles_w_braces, out_dir, verbose=True):
    
    """
    extract features from main dataset that are coincident with the marker.
    Saves them to file
    
    Parameters
    ----------
    marker_df: pd.DataFrame
        output from qrdar.readMarker
    tile_index: pd.DataFrame [required fields are ['x', 'y', 'tile_number']]
        tile index as dataframe
    extract_tiles_w_braces: str with {}
        path to tiles where tile number is replaced with {} e.g. '../tiles/tile_{}.pcd'  
    out_dir: str
        filepath to output directory
    verbose: boolean
        print something
    """
    
    for ix, row in marker_df.iterrows():
        
        if verbose: print('extracting feature:', row.code)
        corners = pd.DataFrame(columns=['x', 'y', 'z'])
        for i, c in enumerate([row.c0, row.c1, row.c2, row.c3]):
            if isinstance(c, str): 
                c = tuple([float(i) for i in c[1:-1].split(',')])
            if i == 0 and isinstance(c, float): break
            if i == 3 and isinstance(c, float): continue
            corners.loc[i, ['x', 'y', 'z']] = list(c)
        if len(corners) == 0: continue
        v = _extract_feature(row.code, corners, tile_index, extract_tiles_w_braces, out_dir, verbose)
#         return v

def _extract_feature(code, corners, tile_index, tile_path, out_dir, verbose):
    
    R = np.identity(4)
    R[:3, 3] = -corners[['x', 'y', 'z']].mean()

    voxel = pd.DataFrame()
    all_tiles = []
   
    # codes may overlap tiles so read in all tiles and append
    for ix, cnr in corners.iterrows():
        # extract tiles with a 10 m buffer
        tile_names = tile_index.loc[np.where((np.isclose(cnr.y, tile_index.y, atol=10) & 
                                              np.isclose(cnr.x, tile_index.x, atol=10)))].tile.values
        for tile_name in tile_names:
            if tile_name in all_tiles: continue
            if verbose: print('        processing tile:', tile_name)
            all_tiles.append(tile_name)
            tile = read_pcd(tile_path.format(tile_name)) if tile_path.endswith('.pcd') else read_ply(tile_path.format(tile_name))
            tile = tile.loc[(tile.x.between(corners.x.min() - 3, corners.x.max() + 3)) & 
                            (tile.y.between(corners.y.min() - 3, corners.y.max() + 3)) &
                            (tile.z.between(corners.z.min() - 2, corners.z.max() + 4))]
            if len(tile) == 0: continue
            # apply rotation
            tile[['x', 'y', 'z']] = apply_rotation(R, tile)
            # filter
            tile = tile[(tile.z.between(0, 4)) &
                        (tile.x.between(-1.5, 1.5)) &
                        (tile.y.between(-2, 2))]
            voxel = voxel.append(tile)    
#             if verbose: print 'number of points extracted before clipping:', len(voxel)
    
#     return voxel
    if verbose: print('    total number of points for voxel:', len(voxel))
    if verbose: print('    running DBSCAN on voxel')
    dbscan = DBSCAN(eps=.1, min_samples=25).fit(voxel[['x', 'y', 'z']])
    print(dbscan.labels_)
    voxel.loc[:, 'labels_'] = dbscan.labels_
    voxel = voxel[voxel.labels_ != -1] 
    voxel[['x', 'y', 'z']] = apply_rotation(np.linalg.inv(R), voxel)
    if verbose: print('    DBSCAN completed')
    
    v = voxel.groupby('labels_').agg([min, max, 'count'])
    stem_cluster = []
    inc = 0
    
    if verbose: print('    incrementing over voxel')
    while len(stem_cluster) == 0:
        stem_cluster = v[(corners.x.min() >= v['x']['min'] - inc) & 
                         (corners.x.max() <= v['x']['max'] + inc) &
                         (corners.y.min() >= v['y']['min'] - inc) & 
                         (corners.y.max() <= v['y']['max'] + inc)].index
        inc += .01
    
    if verbose: print('    finished incremeting')
    
    print('saving feature to:', os.path.join(out_dir, 'cluster_{}.pcd'.format(code)))
    write_pcd(voxel[voxel.labels_.isin(stem_cluster)], 
              os.path.join(out_dir, 'cluster_{}.pcd'.format(code)))  

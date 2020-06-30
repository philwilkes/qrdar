import pandas as pd
import numpy as np

from qrdar.common import * 

def locateTargets(pc, markerTemplate=None, min_intensity=0, rmse_threshold=.15, 
                  check_z=True, verbose=False):

    """ 
    Groups stickers into potential targets, this is required for
    the next stage

    Parameters
    ----------
    potential_dots: pd.DataFrame
        sticker centres
    check_z: boolean (default True)
        assumes targets are upright and removes otherwise
    verbose: boolean (default False)
        print something

    Returns
    -------
    code_centre: pd.DataFrame
        locatin of potential targets
    """
    
    if 'target_labels_' in pc.columns:
        del pc['target_labels_']
        
    if markerTemplate is None:
        markerTemplate = template()
    
    # cluster pc into potential dots
    potential_dots = pc.groupby('sticker_labels_').mean().reset_index()

    # target_centre = pc.loc[pc.labels_.isin(potential_dots.index)].groupby('labels_').mean()
    dbscan = DBSCAN(eps=.4, min_samples=3).fit(potential_dots[['x', 'y', 'z']])
    potential_dots.loc[:, 'target_labels_'] = dbscan.labels_
    potential_dots = potential_dots[potential_dots.target_labels_ != -1]
    if verbose: print('number of potential targets:', len(potential_dots.target_labels_.unique()))
    
    # check if targets are too close
    while np.any(~potential_dots.target_labels_.value_counts().isin([3, 4])):
        
        for labels in potential_dots.target_labels_.unique():
            
            remove = False
            bright_spots_tmp = potential_dots[potential_dots.target_labels_ == labels]
            if verbose: print('processing: {} (number of stickers {})'.format(labels, len(bright_spots_tmp)))
            
            # remove stickers that don't match ~ distance between stickers
            remove_idx = distanceFilter(bright_spots_tmp, markerTemplate)
            potential_dots = potential_dots.loc[~potential_dots.index.isin(remove_idx)]
            bright_spots_tmp = bright_spots_tmp.loc[~bright_spots_tmp.index.isin(remove_idx)]
            if len(remove_idx) > 0:
                if verbose: print("\tstickers removed for wrong distance:", len(remove_idx))
            
            if len(bright_spots_tmp) < 3:
                reason = 'less than 3'
                remove = True
            
            if len(bright_spots_tmp) > 4:
                idx, R, rmse = calculate_R(bright_spots_tmp, markerTemplate)
                if np.isnan(rmse):
                    reason = 'does not fit template'
                    remove = True
                else:
                    labels = potential_dots.target_labels_.max() + 1
                    bright_spots_tmp = bright_spots_tmp.loc[idx]
                    potential_dots.loc[idx, 'target_labels_'] = labels
            
            if 3 <= len(bright_spots_tmp) <= 4:
                if check_z:
                    if np.ptp(bright_spots_tmp.z) < .1 or np.ptp(bright_spots_tmp.z) > .4:
                        reason = 'points are not spread over Z correctly' 
                        remove = True
                idx, R, rmse = calculate_R(bright_spots_tmp, markerTemplate)
                if np.isnan(rmse):
                    reason = 'rmse greater than threshold'
                    remove = True

            if remove:
                if verbose: print("\tremvoing targets labelled: {} {}".format(labels, reason))
                potential_dots = potential_dots[potential_dots.target_labels_ != labels]
    
    # find code centres
    code_centres = potential_dots.groupby('target_labels_')[['x', 'y', 'z']].mean().reset_index()
    if verbose: print('number of potential tree codes:', len(code_centres))

    pc = pd.merge(pc, potential_dots[['sticker_labels_', 'target_labels_']],  on='sticker_labels_', how='right')

    return pc

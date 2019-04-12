import pandas as pd

from qrdar.common import * 

def locateTargets(pc, markerTemplate=None, min_intensity=0, rmse_threshold=.15, verbose=False):

    """ 
    Groups stickers into potential targets, this is required for
    the next stage

    Parameters
    ----------
    potential_dots: pd.DataFrame
        sticker centres
    verbose: boolean (default False)
        print something

    Returns
    -------
    code_centre: pd.DataFrame
        locatin of potential targets
    """
    
    if 'target_labels_' in pc.columns:
        del pc['target_labels_']
        
    if markerTemplate == None:
        markerTemplate = template()
    
    # cluster pc into potential dots
    potential_dots = pc.groupby('sticker_labels_').mean().reset_index()

    # target_centre = pc.loc[pc.labels_.isin(potential_dots.index)].groupby('labels_').mean()
    dbscan = DBSCAN(eps=.4, min_samples=3).fit(potential_dots[['x', 'y', 'z']])
    potential_dots.loc[:, 'target_labels_'] = dbscan.labels_
    potential_dots = potential_dots[potential_dots.target_labels_ != -1] 
    
    # check if targets are too close
    while np.any(potential_dots.target_labels_.value_counts().values > 4):
        for labels in potential_dots.target_labels_.unique():
            bright_spots_tmp = potential_dots[potential_dots.target_labels_ == labels]
            if len(bright_spots_tmp) > 4:
                if verbose: print 'filtering points matching label {} with {} points'.format(labels, len(bright_spots_tmp))
                idx, R, rmse = calculate_R(bright_spots_tmp, markerTemplate)
                potential_dots.loc[idx, 'target_labels_'] = potential_dots.target_labels_.max() + 1
    
    # remove any stickers that are not associated wth at least 2 others
    vc = potential_dots.target_labels_.value_counts()
    potential_dots = potential_dots[potential_dots.target_labels_.isin(vc[vc >= 3].index)]
    
    # and double check (remove any with large error that have been left over)
    for labels in potential_dots.target_labels_.unique():
        bright_spots_tmp = potential_dots[potential_dots.target_labels_ == labels]
        idx, R, rmse = calculate_R(bright_spots_tmp, markerTemplate)
        if rmse > .02 and .1 < bright_spots_tmp.z.ptp() < .4:
            potential_dots = potential_dots.loc[potential_dots.target_labels_ != labels]
    
    # find code centres
    code_centres = potential_dots.groupby('target_labels_')[['x', 'y', 'z']].mean().reset_index()
    if verbose: print 'number of potential tree codes:', len(code_centres)

    pc = pd.merge(pc, potential_dots[['sticker_labels_', 'target_labels_']],  on='sticker_labels_', how='right')

    return pc
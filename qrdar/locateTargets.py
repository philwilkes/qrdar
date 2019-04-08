import pandas as pd
from sklearn.cluster import DBSCAN

def locateTargets(potential_dots, verbose=False):

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

    # target_centre = pc.loc[pc.labels_.isin(potential_dots.index)].groupby('labels_').mean()
    dbscan = DBSCAN(eps=.4, min_samples=3).fit(potential_dots[['x', 'y', 'z']])
    potential_dots.loc[:, 'target_labels_'] = dbscan.labels_
    potential_dots = potential_dots[potential_dots.target_labels_ != -1]
    
    # find code centres
    code_centres = potential_dots.groupby('target_labels_')[['x', 'y', 'z']].mean().reset_index()
    if verbose: print 'number of potential tree codes:', len(code_centres)

    return code_centres

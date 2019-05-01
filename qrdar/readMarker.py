import os
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

import qrdar
from qrdar.common import *
from qrdar.io.pcd_io import *
from qrdar.io.ply_io import *

# a bit of hack for Python 2.x
__dir__ = os.path.split(os.path.abspath(qrdar.__file__))[0]


def singleTile(bright, 
               pc, 
               min_intensity=0,
               markerTemplate=None,
               expected_codes=[],
               codes_dict='aruco_mip_16h3',
               save_to=False,
               print_figure=True, 
               sticker_error =.015,
               verbose=True
               ):

    """
    This reads the potential tree codes and returns there
    marker number with confidence of correct ID.

    An assumption is made that TLS data is tiled
    
    Parameters
    ----------
    bright: pd.DataFrame [requires fields ['x', 'y', 'z', 'sticker_labels_', 'target_labels_']]
        Dataframe containing output from locateTargets
    pc:
        full point cloud from which to extract targets
    min_intensity: float or int (default 0)
        minimum intensity for stickers, the lower the number the more likely erroneous points
        will be identified and the longer the runtime will be.
    markerTemplate: None or 4 x 3 np.array (default None)
        relative xyz location of stickers on targets, default loads template used in example 
    expected_codes: None or list (default None)
        a list of expected targets, if None all codes in codes_dict expected
    codes_dict: str ('aruco_mip_16h3') or n x n x m array
        defaults to 'aruco_mip_16h3' but another dictionary can be provided
    print_figure: boolean (default True)
        creates images of extracted markers, can be useful for identifying codes that were
        not done so automatically. Saves to file.
    sticker_error: float (default .015)
        accpetable rmse for a target stickers to match the template
    verbose: boolean (default True)
        print something
    
    Returns
    -------
    marker_df: pd.DataFrame
        Dataframe of marker number and other metadata
    
    """

    if isinstance(codes_dict, str):
        codes = load_codes(codes_dict)
    else:
        codes = codes_dict
        
    if len(expected_codes) == 0:
        expected_codes = np.arange(codes.shape[2])
    codes = codes[:, :, expected_codes]

    if markerTemplate == None:
        markerTemplate = template()
    
    # create a database to store output metadata
    marker_df = pd.DataFrame(index=bright.target_labels_.unique(), 
                             columns=['x', 'y', 'z', 'rmse', 'code', 'confidence', 'c0', 'c1', 'c2', 'c3'])
    
    for i in np.sort(bright.target_labels_.unique().astype(int)):

        if verbose: print 'processing targets:', i
            
        # locate stickers
        corners = bright[bright.target_labels_ == i].groupby('sticker_labels_').mean()
        marker_df.loc[i, ['x', 'y', 'z']] = corners[['x', 'y', 'z']].mean()
        
        # extract portion of tile containing code
        code = pc[(pc.x.between(corners.x.min() - .1, corners.x.max() + .1)) &
                  (pc.y.between(corners.y.min() - .1, corners.y.max() + .1)) &
                  (pc.z.between(corners.z.min() - .1, corners.z.max() + .1))][['x', 'y', 'z', 'intensity']]
        
        # create axis for plotting
        if print_figure:
            f = plt.figure(figsize=(10, 5))
            f.text(.01, .05, 'cluster: {}'.format(i), ha='left')
            ax1 = f.add_axes([0, 0, .32, 1])
            ax2 = f.add_axes([.33, .5, .32, .49])
            ax3 = f.add_axes([.33, 0, .32, .49])
            ax4 = f.add_axes([.66, 0, .32, .49])
            ax5 = f.add_axes([.66, .5, .32, .49])
            [ax.axis('off') for ax in [ax1, ax2, ax3, ax4, ax5]]
        
        # identify stickers
        if verbose: print '    locating stickers'
        idx, R, rmse = calculate_R(corners, markerTemplate)
        sticker_centres = corners.loc[idx]
        #sticker_centres, R, rmse = identify_stickers(code, markerTemplate, min_intensity, sticker_error)
        marker_df.loc[i, 'rmse'] = rmse
        marker_df.at[i, 'c0'] = tuple(sticker_centres[['x', 'y', 'z']].loc[sticker_centres.index[0]].round(2))  
        marker_df.at[i, 'c1'] = tuple(sticker_centres[['x', 'y', 'z']].loc[sticker_centres.index[1]].round(2))  
        marker_df.at[i, 'c2'] = tuple(sticker_centres[['x', 'y', 'z']].loc[sticker_centres.index[2]].round(2))  
        if len(sticker_centres) == 4:
            marker_df.at[i, 'c3'] = tuple(sticker_centres[['x', 'y', 'z']].loc[sticker_centres.index[3]].round(2))  

        if verbose: print '    sticker rmse:', rmse
    
        # applying rotation matrix
        if verbose: print '    applying rotation matrix'
        sticker_centres.loc[:, ['x', 'y', 'z']] = apply_rotation(R, sticker_centres)
        code.loc[:, ['x', 'y', 'z']] = apply_rotation(R, code)
    
        # set up and plot point cloud
        if print_figure:
            code.sort_values('y', inplace=True, ascending=False)
            ax1.scatter(code.x, code.z, c=code.intensity, edgecolor='none', s=1, cmap=plt.cm.Spectral_r)
            ax1.scatter(markerTemplate.x, markerTemplate.z, s=30, edgecolor='b', facecolor='none')
            ax1.scatter(sticker_centres.x, sticker_centres.z, s=30, edgecolor='r', facecolor='none')
            
        if len(sticker_centres) == 0 or rmse > sticker_error:
            if verbose: print "    could not find 3 bright targets that match the markerTemplate"
            if print_figure: 
                code.sort_values('y', inplace=True)
                if verbose: print '    saving images:', '{}.png'.format(i)
                f.savefig('{}.png'.format(i))
            continue    

        # extracting fiducial marker
        # TODO: make this a function
        if verbose: print '    extracting fiducial marker'
        code_ = code.copy()
        code = code.loc[(code.x.between(-.01, .18)) & 
                        (code.y.between(-.01, .01)) &
                        (code.z.between(.06, .25))]
        code.x = code.x - code.x.min()
        code.z = code.z - code.z.min()
        code.loc[:, 'xx'] = code.x // 0.032
        code.loc[:, 'zz'] = code.z // 0.032
    
        # correct for non-flat target
        #code.loc[:, 'yt'] = code.groupby(['xx', 'zz']).y.transform(np.percentile, 75)
        #code.loc[:, 'yn'] = code.y - code.yt
        #code = code.loc[code.yn.between(-.01, .01)]
    
        code.sort_values('intensity', inplace=True)
        if print_figure: ax2.scatter(code.x, code.z, c=code.intensity, edgecolor='none', 
                                     s=10, cmap=plt.cm.Greys_r, vmin=-10, vmax=-5)
    
        # matrix for holding estimated codes and confidence 
        scores = np.zeros((3, 2))

        # method 1
        try:
            img_1 = method_1(code)
            scores[0, :] = calculate_score(img_1, codes)
            if print_figure: ax3.imshow(np.rot90(img_1, 1), cmap=plt.cm.Greys_r, interpolation='none')
        except Exception as err:
            if verbose: print '    {}'.format(err)      
        
        # method 2 .4 threshold
        try:
            img_2 = method_2(code, .4)
            scores[1, :] = calculate_score(img_2, codes)
            if print_figure: ax4.imshow(np.rot90(img_2, 1), cmap=plt.cm.Greys_r, interpolation='none')
        except Exception as err:
            if verbose: print '    {}'.format(err)
            
        # method 2 .6 threshold
        try:
            img_3 = method_2(code, .6)
            scores[2, :] = calculate_score(img_3, codes)
            if print_figure: ax5.imshow(np.rot90(img_3, 1), cmap=plt.cm.Greys_r, interpolation='none') 
        except Exception as err:
            if verbose: print '    {}'.format(err)

        number = np.unique(scores[np.where(scores[:, 1] == scores[:, 1].max())][:, 0])
        if len(number) > 1:
            if verbose: print '    more than one code identified with same confidence:', number
            if verbose: print '    value of -1 set for code in marker_df'
            if verbose: print '    writing these to {}'.format(os.path.join(os.getcwd(), str(i) + '.log'))
            read_code = [int(expected_codes[int(n)]) for n in number]
            confidence = scores[np.where(scores[:, 1] == scores[:, 1].max())][0, 1]
            with open(os.path.join(os.getcwd(), str(i) + '.log'), 'w') as fh:
                fh.write(' '.join([str(n) for n in number]))
                fh.write(' {}'.format(confidence))
        else:
            number, confidence = scores[np.where(scores[:, 1] == scores[:, 1].max())][0, :]
            read_code = int(expected_codes[int(number)])
        if verbose: print '    tag identified (ci): {} ({})'.format(read_code, confidence)

        if print_figure:
            f.text(.01, .01, 'code: {} ({})'.format(read_code, confidence))
            f.savefig('{}.png'.format(i))
            if verbose: print '    saved image to:', '{}.png'.format(i)

        marker_df.loc[i, 'code'] = read_code
        marker_df.loc[i, 'confidence'] = confidence
        
    return marker_df    


def fromTiles(pc, 
              tile_index, 
              refl_tiles_w_braces,
              min_intensity=0,
              markerTemplate=None,
              expected_codes=[],
              codes_dict='aruco_mip_16h3',
              save_to=False,
              print_figure=True, 
              sticker_error =.015,
              verbose=True
              ):

    """
    This reads the potential tree codes and returns there
    marker number with confidence of correct ID.

    An assumption is made that TLS data is tiled
    
    Parameters
    ----------
    pc: pd.DataFrame [requires fields ['x', 'y', 'z', 'sticker_labels_', 'target_labels_']]
        Dataframe containing points
    tile_index: pd.DataFrame [required fields are ['x', 'y', 'tile_number']]
        tile index as dataframe
    refl_tiles_w_braces: str with {}
        path to tiles where tile number is replaced with {} e.g. '../tiles/tile_{}.pcd'
    min_intensity: float or int (default 0)
        minimum intensity for stickers, the lower the number the more likely erroneous points
        will be identified and the longer the runtime will be.
    markerTemplate: None or 4 x 3 np.array 
        relative xyz location of stickers on targets
    expected_codes: None or list (default None)
        a list of expected targets
    codes_dict: str ('aruco_mip_16h3') or n x n x m array
        defaults to 'aruco_mip_16h3' but another dictionary can be provided
    print_figure: boolean (default True)
        creates images of extracted markers, can be useful for identifying codes that were
        not done so automatically
    sticker_error: float (default .005)
        accpetable rmse for a target stickers to match the template
    verbose: boolean (default True)
        print something
    
    Returns
    -------
    marker_df: pd.DataFrame
        Dataframe of marker number and other metadata
    
    """

    if isinstance(codes_dict, str):
        codes = load_codes(codes_dict)
    else:
        codes = codes_dict
        
    if len(expected_codes) == 0:
        expected_codes = np.arange(codes.shape[2])
    codes = codes[:, :, expected_codes]

    if markerTemplate == None:
        markerTemplate = template()

    # create a database to store output metadata
    marker_df = pd.DataFrame(index=pc.target_labels_.unique(), 
                             columns=['x', 'y', 'z', 'rmse', 'code', 
                                      'confidence', 'c0', 'c1', 'c2', 'c3'])

    for i in np.sort(pc.target_labels_.unique().astype(int)):

        if verbose: print 'processing targets:', i
            
        # locate stickers
        corners = pc[pc.target_labels_ == i].groupby('sticker_labels_').mean()
        marker_df.loc[i, ['x', 'y', 'z']] = corners[['x', 'y', 'z']].mean()
        
        # extract tile with full-res data
        code = extract_tile(corners, tile_index, refl_tiles_w_braces)
        
        # add bright points that may have been removed with deviation
        # RIEGL SPECFIC!
        code = code.append(pc[(pc.x.between(corners.x.min(), corners.x.max())) &
                              (pc.y.between(corners.y.min(), corners.y.max())) &
                              (pc.z.between(corners.z.min(), corners.z.max())) &
                              (pc.intensity > 0)][['x', 'y', 'z', 'intensity']])
        
        # create axis for plotting
        if print_figure:
            f = plt.figure(figsize=(10, 5))
            f.text(.01, .05, 'cluster: {}'.format(i), ha='left')
            ax1 = f.add_axes([0, 0, .32, 1])
            ax2 = f.add_axes([.33, .5, .32, .49])
            ax3 = f.add_axes([.33, 0, .32, .49])
            ax4 = f.add_axes([.66, 0, .32, .49])
            ax5 = f.add_axes([.66, .5, .32, .49])
            [ax.axis('off') for ax in [ax1, ax2, ax3, ax4, ax5]]
        
        # identify stickers
        if verbose: print '    locating stickers'
        idx, R, rmse = calculate_R(corners, markerTemplate)
        sticker_centres = corners.loc[idx]
        #sticker_centres, R, rmse = identify_stickers(code, markerTemplate, min_intensity, sticker_error)
        marker_df.loc[i, 'rmse'] = rmse
        marker_df.at[i, 'c0'] = tuple(sticker_centres[['x', 'y', 'z']].loc[sticker_centres.index[0]].round(2))  
        marker_df.at[i, 'c1'] = tuple(sticker_centres[['x', 'y', 'z']].loc[sticker_centres.index[1]].round(2))  
        marker_df.at[i, 'c2'] = tuple(sticker_centres[['x', 'y', 'z']].loc[sticker_centres.index[2]].round(2))  
        if len(sticker_centres) == 4:
            marker_df.at[i, 'c3'] = tuple(sticker_centres[['x', 'y', 'z']].loc[sticker_centres.index[3]].round(2))  

        if verbose: print '    sticker rmse:', rmse
    
        # applying rotation matrix
        if verbose: print '    applying rotation matrix'
        sticker_centres.loc[:, ['x', 'y', 'z']] = apply_rotation(R, sticker_centres)
        code.loc[:, ['x', 'y', 'z']] = apply_rotation(R, code)
    
        # set up and plot point cloud
        if print_figure:
            code.sort_values('y', inplace=True, ascending=False)
            ax1.scatter(code.x, code.z, c=code.intensity, edgecolor='none', s=1, cmap=plt.cm.Spectral_r)
            ax1.scatter(markerTemplate.x, markerTemplate.z, s=30, edgecolor='b', facecolor='none')
            ax1.scatter(sticker_centres.x, sticker_centres.z, s=30, edgecolor='r', facecolor='none')
            
        if len(sticker_centres) == 0 or rmse > sticker_error:
            if verbose: print "    could not find 3 bright targets that match the markerTemplate"
            if print_figure: 
                code.sort_values('y', inplace=True)
                ax1.scatter(code.x, code.z, c=code.intensity, edgecolor='none', s=1)
                if verbose: print '    saving images:', '{}.png'.format(i)
                f.savefig('{}.png'.format(i))
            continue    
    
        # extracting fiducial marker
        # TODO: make this a function
        if verbose: print '    extracting fiducial marker'
        code_ = code.copy()
        code = code.loc[(code.x.between(-.01, .18)) & 
                        (code.y.between(-.01, .01)) &
                        (code.z.between(.06, .25))]
        code.x = code.x - code.x.min()
        code.z = code.z - code.z.min()
        code.loc[:, 'xx'] = code.x // 0.032
        code.loc[:, 'zz'] = code.z // 0.032
    
        # correct for non-flat target
        #code.loc[:, 'yt'] = code.groupby(['xx', 'zz']).y.transform(np.percentile, 75)
        #code.loc[:, 'yn'] = code.y - code.yt
        #code = code.loc[code.yn.between(-.01, .01)]
    
        code.sort_values('intensity', inplace=True)
        if print_figure: ax2.scatter(code.x, code.z, c=code.intensity, edgecolor='none', s=10, cmap=plt.cm.Greys_r, vmin=-10, vmax=-5)
    
        # matrix for holding estimated codes and confidence 
        scores = np.zeros((3, 2))

        # method 1
        try:
            img_1 = method_1(code)
            scores[0, :] = calculate_score(img_1, codes)
            if print_figure: ax3.imshow(np.rot90(img_1, 1), cmap=plt.cm.Greys_r, interpolation='none')
        except Exception as err:
            if verbose: print '    {}'.format(err)      
        
        # method 2 .4 threshold
        try:
            img_2 = method_2(code, .4)
            scores[1, :] = calculate_score(img_2, codes)
            if print_figure: ax4.imshow(np.rot90(img_2, 1), cmap=plt.cm.Greys_r, interpolation='none')
        except Exception as err:
            if verbose: print '    {}'.format(err)
            
        # method 2 .6 threshold
        try:
            img_3 = method_2(code, .6)
            scores[2, :] = calculate_score(img_3, codes)
            if print_figure: ax5.imshow(np.rot90(img_3, 1), cmap=plt.cm.Greys_r, interpolation='none') 
        except Exception as err:
            if verbose: print '    {}'.format(err)

        number = np.unique(scores[np.where(scores[:, 1] == scores[:, 1].max())][:, 0])
        if len(number) > 1:
            read_code = [int(expected_codes[int(n)]) for n in number]
            if verbose: print '    more than one code identified with same confidence:', read_code
            if verbose: print '    value of -1 set for code in marker_df'
            if verbose: print '    writing these to {}'.format(os.path.join(os.getcwd(), str(i) + '.log'))
            confidence = scores[np.where(scores[:, 1] == scores[:, 1].max())][0, 1]
            with open(os.path.join(os.getcwd(), str(i) + '.log'), 'w') as fh:
                fh.write(' '.join([str(n) for n in number]))
                fh.write(' {}'.format(confidence))
        else:
            number, confidence = scores[np.where(scores[:, 1] == scores[:, 1].max())][0, :]
            read_code = int(expected_codes[int(number)])
        if verbose: print '    tag identified (ci): {} ({})'.format(read_code, confidence)

        if print_figure:
            f.text(.01, .01, 'code: {} ({})'.format(read_code, confidence))
            f.savefig('{}.png'.format(i))
            if verbose: print '    saved image to:', '{}.png'.format(i)

        marker_df.loc[i, 'code'] = read_code
        marker_df.loc[i, 'confidence'] = confidence
        
    return marker_df

def extract_tile(corners, tile_centres, filepath):
    
    tile_names = []
    code = pd.DataFrame()   
    
    # codes may overlap tiles so read in all tiles and append
    for ix, cnr in corners.iterrows():
        tile_name = tile_centres.loc[np.where((np.isclose(cnr.y, tile_centres.y, atol=5) & 
                                               np.isclose(cnr.x, tile_centres.x, atol=5)))].tile.values[0]
        if tile_name not in tile_names:
            tile = read_pcd(filepath.format(tile_name))
            tile = tile.loc[(tile.x.between(corners.x.min() - .1, corners.x.max() + .1)) & 
                            (tile.y.between(corners.y.min() - .1, corners.y.max() + .1)) &
                            (tile.z.between(corners.z.min() - .1, corners.z.max() + .1))]
            code = code.append(tile)
            tile_names.append(tile_name)
             
    return code
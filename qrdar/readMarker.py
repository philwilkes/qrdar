
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

import qrdar
from qrdar.common import *
from qrdar.io.pcd_io import *
from qrdar.io.ply_io import *

# a bit of hack for Python 2.x
# __dir__ = os.path.split(os.path.abspath(qrdar.__file__))[0]


def readCodes(bright, 
              pc=None,
              tile_index=None,
              refl_tiles_w_braces='',
              min_intensity=0,
              reflectance_field='intensity',
              markerTemplate=None,
              expected_codes=[],
              codes_dict='aruco_mip_16h3',
              save_to=False,
              print_figure=True, 
              sticker_error =.015,
              code_dims={'edge':.03, 'x':(-.01, .18), 'y':(-.05, .05), 'z':(.06, .25)},
              return_marker_df=True,
              save_pc=False,
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
    tile_index: pd.DataFrame [required fields are ['x', 'y', 'tile_number']]
        tile index as dataframe
    refl_tiles_w_braces: str with {} (default '')
        path to tiles where tile number is replaced with {} e.g. '../tiles/tile_{}.pcd'. If pc
        is pd.DataFrame this is required.
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
    code_dims: dict('edge':.032, 'x':(-.01, .18), 'y':(-.1, .1), 'z':(.06, .25))  
        dimensions of the code where edge is the edge length of the qr code squares
        and x, y and z are the corner locations. Defaults are for the standard 
        aruco_mip_16h3 dictionary of codes.
    save_pc: boolean (default False)
        save point clouds of markers
    verbose: boolean (default True)
        print something
    
    Returns
    -------
    marker_df: pd.DataFrame
        Dataframe of marker number and other metadata
    
    """
    
    assert isinstance(pc, pd.DataFrame) or isinstance(tile_index, pd.DataFrame), \
        'pc or tile_index needs to be specified'
    assert not (isinstance(pc, pd.DataFrame) and isinstance(tile_index, pd.DataFrame)), \
        'a point cloud and tile index have been specified'

    if isinstance(codes_dict, str):
        codes = load_codes(codes_dict)
    else:
        codes = codes_dict
        
    if len(expected_codes) == 0:
        expected_codes = np.arange(codes.shape[2])
    codes = codes[:, :, expected_codes]

    if markerTemplate is None:
        markerTemplate = template()
    
    # create a database to store output metadata
    marker_df = pd.DataFrame(index=bright.target_labels_.unique(), 
                             columns=['x', 'y', 'z', 'rmse', 'code', 'confidence', 'c0', 'c1', 'c2', 'c3'])
    
    bright.loc[:, 'intensity'] = bright[reflectance_field]
    if isinstance(pc, pd.DataFrame):
        pc.loc[:, 'intensity'] = pc[reflectance_field]
       
    
    for i, target in enumerate(np.sort(bright.target_labels_.unique().astype(int))):
        
        if verbose: print('processing targets:', target)
            
        # locate stickers
        corners = bright[bright.target_labels_ == target].groupby('sticker_labels_').mean()
        marker_df.loc[target, ['x', 'y', 'z']] = corners[['x', 'y', 'z']].mean()
        
        # extract portion of tile containing code
        if isinstance(tile_index, pd.DataFrame):
            assert refl_tiles_w_braces != '' and '{}' in refl_tiles_w_braces, 'refl_tiles_w_braces needs to be a path with {}'
            code = extract_tile(corners, tile_index, refl_tiles_w_braces)
        else:
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
        if verbose: print('    locating stickers')
        idx, R, rmse = calculate_R(corners, markerTemplate)
        if np.isnan(rmse): continue # need to investiage why this is needed - very rarely though!
        sticker_centres = corners.loc[idx]
        marker_df.loc[target, 'rmse'] = rmse
        marker_df.at[target, 'c0'] = tuple(sticker_centres[['x', 'y', 'z']].loc[sticker_centres.index[0]].round(2))  
        marker_df.at[target, 'c1'] = tuple(sticker_centres[['x', 'y', 'z']].loc[sticker_centres.index[1]].round(2))  
        marker_df.at[target, 'c2'] = tuple(sticker_centres[['x', 'y', 'z']].loc[sticker_centres.index[2]].round(2))  
        if len(sticker_centres) == 4:
            marker_df.at[target, 'c3'] = tuple(sticker_centres[['x', 'y', 'z']].loc[sticker_centres.index[3]].round(2))  

        if verbose: print('    sticker rmse:', rmse)

        # applying rotation matrix
        if verbose: print('    applying rotation matrix')
        sticker_centres.loc[:, ['x', 'y', 'z']] = apply_rotation(R, sticker_centres)
        code.loc[:, ['x', 'y', 'z']] = apply_rotation(R, code)
        
        if len(sticker_centres) == 0 or rmse > sticker_error:
            if verbose: print("    could not find 3 bright targets that match the markerTemplate")
            if print_figure: 
                code.sort_values('y', inplace=True)
                if verbose: print('    saving images:', '{}.png'.format(i))
                f.savefig('{}.png'.format(i))
            continue   
            
        # set up and plot point cloud
        if print_figure:
            code.sort_values('y', inplace=True, ascending=False)
            ax1.scatter(code.x, code.z, c=code.intensity, edgecolor='none', s=1, cmap=plt.cm.Spectral_r)
            ax1.scatter(markerTemplate.x, markerTemplate.z, s=30, edgecolor='b', facecolor='none')
            ax1.scatter(sticker_centres.x, sticker_centres.z, s=30, edgecolor='r', facecolor='none')      

        # extracting fiducial marker
        # TODO: make this a function
        if verbose: print('    extracting fiducial marker')
        code_ = code.copy()
        code = code.loc[(code.x.between(*code_dims['x'])) & 
                        (code.y.between(*code_dims['y'])) &
                        (code.z.between(*code_dims['z']))]
        xmin, zmin = code.x.min(), code.z.min()
        # save pc
        if save_pc:
            if verbose: print('    saving point cloud to: {}.ply'.format(i))
            write_ply('{}.ply'.format(i), apply_rotation(np.linalg.inv(R), code.copy())) 
            np.savetxt('{}.rot.txt'.format(i), R)   
        code.x = code.x - code.x.min()
        code.z = code.z - code.z.min()
        code.loc[:, 'xx'] = code.x // code_dims['edge']
        code.loc[:, 'zz'] = code.z // code_dims['edge']
    

        # TODO: correct for non-flat target
        #code.loc[:, 'yt'] = code.groupby(['xx', 'zz']).y.transform(np.percentile, 75)
        #code.loc[:, 'yn'] = code.y - code.yt
        #code = code.loc[code.yn.between(-.01, .01)]
    
        code.sort_values('intensity', inplace=True)
        if print_figure: 
            cbar = ax2.scatter(code.x, code.z, c=code.intensity, edgecolor='none', 
                        s=10, cmap=plt.cm.Greys_r, vmin=-10, vmax=0)
            [ax2.axhline(z, c='r') for z in np.arange(code_dims['z'][0], code_dims['z'][1], code_dims['edge']) - zmin]
            [ax2.axvline(z, c='r') for z in np.arange(code_dims['x'][0], code_dims['x'][1], code_dims['edge']) - xmin]         

        # matrix for holding estimated codes and confidence 
        scores = np.zeros((3, 2))
        
        # method 1
        try:
            img_1 = method_1(code)
            scores[0, :] = calculate_score(img_1, codes)
            if print_figure: ax3.imshow(np.rot90(img_1, 1), cmap=plt.cm.Greys_r, interpolation='none')
        except Exception as err:
            if verbose: print(('\t{}'.format(err)))    
        
        # method 2 .4 threshold
        try:
            img_2 = method_2(code, .4)
            scores[1, :] = calculate_score(img_2, codes)
            if print_figure: ax4.imshow(np.rot90(img_2, 1), cmap=plt.cm.Greys_r, interpolation='none')
        except Exception as err:
            if verbose: print('\t{}'.format(err))
            
        # method 2 .6 threshold
        try:
            img_3 = method_2(code, .6)
            scores[2, :] = calculate_score(img_3, codes)
            if print_figure: ax5.imshow(np.rot90(img_3, 1), cmap=plt.cm.Greys_r, interpolation='none') 
        except Exception as err:
            if verbose: print('\t{}'.format(err))

        number = np.unique(scores[np.where(scores[:, 1] == scores[:, 1].max())][:, 0])
        if len(number) > 1:
            if verbose: print('\tmore than one code identified with same confidence:', number)
            if verbose: print('\tvalue of -1 set for code in marker_df')
            if verbose: print('\twriting these to {}'.format(os.path.join(os.getcwd(), str(i) + '.log')))
            read_code = [int(expected_codes[int(n)]) for n in number]
            confidence = scores[np.where(scores[:, 1] == scores[:, 1].max())][0, 1]
            with open(os.path.join(os.getcwd(), str(i) + '.log'), 'w') as fh:
                fh.write(' '.join([str(n) for n in number]))
                fh.write(' {}'.format(confidence))
        else:
            number, confidence = scores[np.where(scores[:, 1] == scores[:, 1].max())][0, :]
            read_code = int(expected_codes[int(number)])
        if verbose: print('    tag identified (ci): {} ({})'.format(read_code, confidence))

        if print_figure:
            f.text(.01, .01, 'code: {} ({})'.format(read_code, confidence))
            f.savefig('{}.png'.format(i))
            if verbose: print('    saved image to:', '{}.png'.format(i))

        marker_df.loc[target, 'code'] = read_code
        marker_df.loc[target, 'confidence'] = confidence

    
    if return_marker_df:
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

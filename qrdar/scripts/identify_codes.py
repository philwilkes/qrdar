import qrdar
import sys
import argparse

def identify_codes_in_pc(pc,  
                         expected=[], 
                         print_figure=False,
                         codes_dict='aruco_mip_16h3',
                         marker_template=None,
                         verbose=False):

    bright = qrdar.search4stickers.find(pc)
    bright = qrdar.search4stickers.filterBySize(bright)
    bright = qrdar.locateTargets(bright, check_z=False, verbose=False)
    marker_df = qrdar.readMarker.readCodes(bright, pc=pc, 
                                           expected_codes=expected,
                                           print_figure=print_figure,
                                           codes_dict=codes_dict,
                                           markerTemplate=marker_template,
                                           verbose=verbose)
    
    print(marker_df[['code', 'confidence', 'x', 'y', 'z']])
    return marker_df
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pc', '-p', type=str, required=True, help='path to point cloud')
    parser.add_argument('--min_reflectance', '-m', type=float, default=-1, required=False, 
                        help='minimum reflectance values')
    parser.add_argument('--refl_field', '-r', default='intensity', help='name of reflectance field in data')
    parser.add_argument('--expected', '-e', default=[], nargs='+', help='list of expected codes')
    parser.add_argument('--figures', '-f', action='store_true', help='generate images from scans')
    parser.add_argument('--verbose', action='store_true', help='print something')
    args = parser.parse_args()
  
    pc = qrdar.io.read_ply(args.pc)
    if args.refl_field != 'intensity':
        pc.rename(columns={args.refl_field:'intensity'}, inplace=True)
    pc = pc[pc.intensity > args.min_reflectance]
    assert len(pc) > 0, 'pc has no points after filtering, try reducing min_reflectance value (defualt -1)'
    
    # run code
    marker_df = identify_codes_in_pc(pc, 
                                     expected=[int(e) for e in args.expected],
                                     print_figure=args.figures,
                                     verbose=args.verbose)
    
    marker_df.to_csv(args.pc.replace('.ply', '.csv'))
    
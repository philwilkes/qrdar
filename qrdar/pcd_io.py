import sys
import argparse
import numpy as np
import pandas as pd

def read_pcd(fp):

    with open(fp) as pcd:

        length = 0

        for i, line in enumerate(pcd.readlines()):
            length += len(line)
            if 'WIDTH' in line: N = int(line.split()[1])
            if 'FIELDS' in line: F = line.split()[1:]
            if 'DATA' in line:
                fmt = line.split()[1]
                break

        if fmt == 'binary':
            pcd.seek(length)
            arr = np.fromfile(pcd, dtype='f')

            arr = arr[:N*len(F)].reshape(-1, len(F))
            df = pd.DataFrame(arr, columns=F)

    if fmt == 'ascii':
        df = pd.read_csv(fp, sep=' ', names=F, skiprows=11)

    return df

def write_pcd(df, path, binary=True):

    columns = ['x', 'y', 'z', 'intensity']
    if 'intensity' not in df.columns: columns = columns[:3]

    with open(path, 'wb') as pcd:

        pcd.write('# .PCD v0.7 - Point Cloud Data file format\n')
        pcd.write('VERSION 0.7\n')
        pcd.write('FIELDS ' + ' '.join(columns + ['\n']))
        pcd.write('SIZE ' + '4 ' * len(columns) + '\n')
        pcd.write('TYPE ' + 'F ' * len(columns) + '\n')
        pcd.write('COUNT ' + '1 ' * len(columns) + '\n')
        pcd.write('WIDTH {}\n'.format(len(df)))
        pcd.write('HEIGHT 1\n')
        pcd.write('VIEWPOINT 0 0 0 1 0 0 0\n')
        pcd.write('POINTS {}\n'.format(len(df)))
        if binary:
            pcd.write('DATA binary\n')
            df[columns].values.astype('f4').tofile(pcd)
        else:
            pcd.write('DATA ascii\n')
            df[columns].to_csv(pcd, sep=' ', index=False, header=False)


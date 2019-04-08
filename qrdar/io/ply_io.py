import pandas as pd
import numpy as np

def read_ply(path):
    
    with open(path) as ply:
    
        length = 0
        prop = []
        dtype_map = {'float': 'f4', 'uchar': 'B', 'int':'i'}
        dtype = []
        fmt = 'binary'
    
        for i, line in enumerate(ply.readlines()):
            length += len(line)
            if i == 0:
                if 'ascii' in line:
                    fmt = 'ascii' 
            if 'element vertex' in line: N = int(line.split()[2])
            if 'property' in line: 
                dtype.append(dtype_map[line.split()[1]])
                prop.append(line.split()[2])
            if 'end_header' in line: break
    
        ply.seek(length)
        if fmt == 'binary':
            arr = np.fromfile(ply, dtype=','.join(dtype))
        else:
            arr = pd.read_csv(ply, sep=' ')
        df = pd.DataFrame(arr)
        df.columns = prop
        
    return df

def write_ply(output_name, pc):

    cols = ['x', 'y', 'z']
    pc[['x', 'y', 'z']] = pc[['x', 'y', 'z']].astype('f4')

    with open(output_name, 'w') as ply:

        ply.write("ply\n")
        ply.write('format binary_little_endian 1.0\n')
        ply.write("comment Author: Phil Wilkes\n")
        ply.write("obj_info generated with pcd2ply.py\n")
        ply.write("element vertex {}\n".format(len(pc)))
        ply.write("property float x\n")
        ply.write("property float y\n")
        ply.write("property float z\n")
        if 'red' in pc.columns:
            cols += ['red', 'green', 'blue']
            pc[['red', 'green', 'blue']] = pc[['red', 'green', 'blue']].astype('i')
            ply.write("property int red\n")
            ply.write("property int green\n")
            ply.write("property int blue\n")
        for col in pc.columns:
            if col in cols: continue
            cols += [col]
            pc[col] = pc[col].astype('f4')
            ply.write("property float {}\n".format(col))
        ply.write("end_header\n")

        ply.write(pc[cols].to_records(index=False).tobytes()) 

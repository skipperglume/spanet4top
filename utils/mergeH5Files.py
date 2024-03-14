#This is an addaptation of an script created by Marina Galchenkova, all credits go to her.

import os
import sys
import h5py as h5
import numpy as np
import re
import math
from numpy.ctypeslib import ndpointer
from collections import defaultdict
from itertools import groupby
import glob


m1 = '/INPUTS/Jets/MASK'
m2 =  '/INPUTS/Jets/btag'
m3 =  '/INPUTS/Jets/cos_phi'
m4 =  '/INPUTS/Jets/e'
m5 =  '/INPUTS/Jets/eta'
m6 =   '/INPUTS/Jets/mass'
m7 =   '/INPUTS/Jets/pt'
m8 =   '/INPUTS/Jets/sin_phi'
m9 =  '/INPUTS/Leptons/MASK'
m10 =   '/INPUTS/Leptons/cos_phi'
m11 =   '/INPUTS/Leptons/e'
m12 =   '/INPUTS/Leptons/eta'
m13 =   '/INPUTS/Leptons/etag'
m14 =   '/INPUTS/Leptons/mass'
m15 =   '/INPUTS/Leptons/mutag'
m16 =   '/INPUTS/Leptons/pt'
m17 =   '/INPUTS/Leptons/sin_phi'
m18 =   '/INPUTS/Met/cos_phi'
m19 =   '/INPUTS/Met/met'
m20 =   '/INPUTS/Met/sin_phi'
m21 =   '/REGRESSIONS/EVENT/neutrino_e'
m22 =   '/REGRESSIONS/EVENT/neutrino_eta'
m23 =   '/REGRESSIONS/EVENT/neutrino_phi'
m24 =   '/REGRESSIONS/EVENT/neutrino_pt'
m25 =   '/TARGETS/extrajet_parton/extrajet'
m26 =   '/TARGETS/th/b'
m27 =   '/TARGETS/th/q1'
m28 =   '/TARGETS/th/q2'
m29 =   '/TARGETS/tl/b'
m30 =   '/TARGETS/tl/l'

def traverse_datasets(hdf_file):
    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = '{}/{}'.format(prefix, key)
            if isinstance(item, h5.Dataset): # test for dataset
                yield (path, item)
            elif isinstance(item, h5.Group): # test for group (go down)
                yield from h5py_dataset_iterator(item, path)
    with h5.File(hdf_file, 'r') as f:
        for (path, dset) in h5py_dataset_iterator(f):
            yield path



def append_to_dataset(dataset, data):   
    data = np.asanyarray(data)
    dataset.resize(len(dataset) + len(data), axis=0)
    dataset[-len(data):] = data

def create_for_append(h5file, name, data):
    data = np.asanyarray(data)
    return h5file.create_dataset(
          name, data=data, maxshape=(None,) + data.shape[1:])

if __name__ == '__main__':
    input_path = sys.argv[1]
    mode = sys.argv[2]
    filenames_odd = glob.glob(os.path.join(input_path,f'*{mode}.h5'))
    outputh5Filename = f'joined_{mode}.h5'
    file = h5.File(filenames_odd[0], 'r')

    with h5.File(outputh5Filename, "w") as outputDataFile:
        create_for_append(outputDataFile, m1, file[m1])
        create_for_append(outputDataFile, m2, file[m2])
        create_for_append(outputDataFile, m3, file[m3])
        create_for_append(outputDataFile, m4, file[m4])
        create_for_append(outputDataFile, m5, file[m5])
        create_for_append(outputDataFile, m6, file[m6])
        create_for_append(outputDataFile, m7, file[m7])
        create_for_append(outputDataFile, m8, file[m8])
        create_for_append(outputDataFile, m9, file[m9])
        create_for_append(outputDataFile, m10, file[m10])
        create_for_append(outputDataFile, m11, file[m11])
        create_for_append(outputDataFile, m12, file[m12])
        create_for_append(outputDataFile, m13, file[m13])
        create_for_append(outputDataFile, m14, file[m14])
        create_for_append(outputDataFile, m15, file[m15])
        create_for_append(outputDataFile, m16, file[m16])
        create_for_append(outputDataFile, m17, file[m17])
        create_for_append(outputDataFile, m18, file[m18])
        create_for_append(outputDataFile, m19, file[m19])
        create_for_append(outputDataFile, m20, file[m20])
        create_for_append(outputDataFile, m21, file[m21])
        create_for_append(outputDataFile, m22, file[m22]) 
        create_for_append(outputDataFile, m23, file[m23]) 
        create_for_append(outputDataFile, m24, file[m24]) 
        create_for_append(outputDataFile, m25, file[m25]) 
        create_for_append(outputDataFile, m26, file[m26]) 
        create_for_append(outputDataFile, m27, file[m27]) 
        create_for_append(outputDataFile, m28, file[m28]) 
        create_for_append(outputDataFile, m29, file[m29]) 
        create_for_append(outputDataFile, m30, file[m30]) 
        
    file.close()                
    for filename in filenames_odd[1:]:

        h5df5paths = set(traverse_datasets(filename))

        file = h5.File(filename, 'r')

        
        #outputhdf5paths = set()
        #if os.path.exists(outputh5Filename):
        #    outputhdf5paths = set(traverse_datasets(outputh5Filename))
        

        with h5.File(outputh5Filename, "a") as outputDataFile:
            #append_to_dataset(outputDataFile[hdf5path], file[hdf5path][()])
            append_to_dataset(outputDataFile[m1], file[m1])
            append_to_dataset(outputDataFile[m2], file[m2])
            append_to_dataset(outputDataFile[m3], file[m3])
            append_to_dataset(outputDataFile[m4], file[m4])
            append_to_dataset(outputDataFile[m5], file[m5])
            append_to_dataset(outputDataFile[m6], file[m6])
            append_to_dataset(outputDataFile[m7], file[m7])
            append_to_dataset(outputDataFile[m8], file[m8])
            append_to_dataset(outputDataFile[m9], file[m9])
            append_to_dataset(outputDataFile[m10], file[m10])
            append_to_dataset(outputDataFile[m11], file[m11])
            append_to_dataset(outputDataFile[m12], file[m12])
            append_to_dataset(outputDataFile[m13], file[m13])
            append_to_dataset(outputDataFile[m14], file[m14])
            append_to_dataset(outputDataFile[m15], file[m15])
            append_to_dataset(outputDataFile[m16], file[m16])
            append_to_dataset(outputDataFile[m17], file[m17])
            append_to_dataset(outputDataFile[m18], file[m18])
            append_to_dataset(outputDataFile[m19], file[m19])
            append_to_dataset(outputDataFile[m20], file[m20])
            append_to_dataset(outputDataFile[m21], file[m21])
            append_to_dataset(outputDataFile[m22], file[m22]) 
            append_to_dataset(outputDataFile[m23], file[m23]) 
            append_to_dataset(outputDataFile[m24], file[m24]) 
            append_to_dataset(outputDataFile[m25], file[m25]) 
            append_to_dataset(outputDataFile[m26], file[m26]) 
            append_to_dataset(outputDataFile[m27], file[m27]) 
            append_to_dataset(outputDataFile[m28], file[m28]) 
            append_to_dataset(outputDataFile[m29], file[m29]) 
            append_to_dataset(outputDataFile[m30], file[m30]) 
            
            outputhdf5paths = set(traverse_datasets(outputh5Filename))
        
        
        
        file.close()
  

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
import argparse

from h5Test import compactKeyDict, printCompactKeyDict, recursiveHDF5List
class hdf5Structure:
    def __init__(self, name='struct') -> None:
        self.structure = {}
        self.short = False
        self.name = name
    def evaluateStructure(self, file):
        self.file = file
        # print('before:',self.structure)
        compactKeyDict(self.file, self.structure, 0)
        # print('after:',self.structure)

    def setDisplayOption(self, short=False):
        self.short = short

    def displayCompactKeyDict(self, res= [''], key_dict=None, spaceVal=0, short=False):
        if key_dict is None:
            key_dict = self.structure
        for key in key_dict.keys():
        
            # print(spaceVal*'\t',key,sep='')
            res[0] += (spaceVal*'\t')+key+'\n'
            if type(key_dict[key]) == dict:
                # printCompactKeyDict(key_dict[key], spaceVal+1)
                self.displayCompactKeyDict(res, key_dict[key], spaceVal+1, short=short)
            else:
                if not short :
                    result = ''
                    for k in key_dict[key][1].keys():
                        result += f'{k} : {key_dict[key][1][k]}'
                    res[0] += (spaceVal+1)*'\t'+' '+result + '\n'
        return res
    def __repr__(self) -> str:
        result = ''
        result += f'Structure of the HDF5 {self.file} file\n'
        result += '--------------------------------------\n'
        result += self.displayCompactKeyDict(res= [''],key_dict=self.structure,spaceVal=0,short=self.short)[0]

        result += '--------------------------------------\n'

        return result
    def getBranchesList(self, currentBranch=[''], key_dict=None, spaceVal=0, res=[]):
        if key_dict is None:
            key_dict = self.structure

        for key in key_dict:
            currentBranch[0] += key+'/'
            if type(key_dict[key]) == dict:
                self.getBranchesList(currentBranch, key_dict[key], spaceVal+1, res)
                currentBranch[0] = '/'.join(currentBranch[0].split('/')[:-2] ) + '/'
            elif type(key_dict[key]) == list or key_dict[key].empty() :
                currentBranch[0] = currentBranch[0][:-1]
                res.append(currentBranch[0])
                currentBranch[0] = '/'.join(currentBranch[0].split('/')[:-1] ) + '/'
        return res

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
    # Resizing dataset 
    # print(h5file)
    # recursiveHDF5List(h5file)
    if name in h5file:
        # Resizing dataset 
        h5file[name].resize(h5file[name].shape[0] + data.shape[0], axis=0)
        h5file[name][-data.shape[0]:] = data
    else:
        # Create the dataset if it doesn't exist
        h5file.create_dataset(name, data=data, maxshape=(None,) + data.shape[1:])
    # h5file[name].resize(h5file[name].shape[0] + data.shape[0], axis=0)
    # h5file[name][-data.shape[0]:] = data
    # return h5file.create_dataset(name, data=data, maxshape=(None,) + data.shape[1:])

if __name__ == '__main__':

    parser=argparse.ArgumentParser()
    # four_top_SPANET_input_even.h5
    parser.add_argument('-i', '--inloc', default='/home/timoshyd/spanet4Top/ntuples/four_top_SPANET_input/', type=str, help='Input file location')
    parser.add_argument('-o', '--outloc' , default='/home/timoshyd/spanet4Top/ntuples/four_top_SPANET_input/output1.h5', type=str, help='Output location')
    parser.add_argument('-t', '--test', action='store_true', help='Test the code')
    parser.add_argument('-m', '--mode', default='even', type=str, help='Mode e.g. even or odd')

    args = parser.parse_args()

    input_path = args.inloc
    mode = args.mode
    filenames_odd = glob.glob(os.path.join(input_path,f'*{mode}.h5'))
    print(filenames_odd)
    
    file = h5.File(filenames_odd[0], 'r')
    
    struct = hdf5Structure('default')
    struct.evaluateStructure(file)
    recursiveHDF5List(file)
    branchesList = struct.getBranchesList()

    problematicBranches = []
    for branch in branchesList:
        if branch not in file:
            print(f'ERROR: Branch {branch} not found in the file')
            problematicBranches.append(branch)
        else:  
            if type(file[branch]) == h5._hl.dataset.Dataset:
                print(f'Branch {branch} found in the file is [Dataset]',file[branch].shape, file[branch].dtype)
            else:
                print(f'ERROR: Branch {branch} found in the file is ', type(file[branch]))
                problematicBranches.append(branch)
    
    if len(problematicBranches) > 0:
        print('ERROR: The following branches are problematic')
        for branch in problematicBranches:
            branchesList.remove(branch)
            print(f'Branch {branch} removed from the list of branches')
    print('Branches in the file:', branchesList)
    
    
    # Cleaning old output file
    if os.path.exists(args.outloc):
        os.system(f'rm -f {args.outloc}')
    with h5.File(args.outloc, "w") as outputDataFile:
        for branch in branchesList:
            create_for_append(outputDataFile, branch, file[branch])
        
    file.close()                
    for filename in filenames_odd[1:]:
        print(f'Appending file: {filename}')
        h5df5paths = set(traverse_datasets(filename))

        file = h5.File(filename, 'r')
        
        outputhdf5paths = set()
        if os.path.exists(args.outloc):
           outputhdf5paths = set(traverse_datasets(args.outloc))

        print(outputhdf5paths)
        # print(branchesList)
        with h5.File(args.outloc, "a") as outputDataFile:
            print(args.outloc)
            for branch in branchesList:
                print(branch)
                create_for_append(outputDataFile, branch, file[branch])
            
            outputhdf5paths = set(traverse_datasets(args.outloc))
        
        
        file.close()
    
    for filename in filenames_odd:
        print(filename)
        print('+++++++++++++++++')
        iterStruct = hdf5Structure('new struct')
        # print(iterStruct.name)
        iterStruct.evaluateStructure(h5.File(filename, 'r'))
        # recursiveHDF5List(h5.File(filename, 'r'))
        print(iterStruct)
        branchesList = iterStruct.getBranchesList()
        # print(branchesList)
        # print('+++++++++++++++++')
  
    outStruct = hdf5Structure('outputstruct')
    outStruct.evaluateStructure(h5.File(args.outloc, 'r'))
    print(outStruct)
    branchesList = outStruct.getBranchesList()
    print(branchesList)
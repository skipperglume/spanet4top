import argparse
import h5py
import os
import numpy as np
filename = "/home/timoshyd/spanet4Top/SPANet/data/full_hadronic_ttbar/example.h5"
filename = "/home/timoshyd/spanet4Top/ntuples/four_top_SPANET_input_even.h5"


parser=argparse.ArgumentParser()
parser.add_argument('-fn','--filename', default='four_top_SPANET_input/four_top_SPANET_input_even.h5', type=str, help='Input file location')
parser.add_argument('-n','--nPoints', default=1, type=int, help='Number of Data points to diplay')
parser.add_argument('-s', '--shift' , default=0, type=int, help='Shift the data points to display')
args = parser.parse_args()

def recursiveHDF5List(currentContainer, spaceVal=0):
    for key in currentContainer.keys():
        print(' '*spaceVal, key)
        # print(' '*spaceVal, currentContainer[key])
        if type(currentContainer[key]) == h5py._hl.dataset.Dataset:
            # print(' '*spaceVal, currentContainer[key].shape)
            # print(' '*spaceVal, currentContainer[key].dtype)
            continue
        
        if type(currentContainer[key]) == h5py._hl.group.Group:
            recursiveHDF5List(currentContainer[key], spaceVal+1)

def compactKeyDict(currentContainer, key_dict, spaceVal=0):
    for key in currentContainer.keys():
        if type(currentContainer[key]) == h5py._hl.dataset.Dataset:
            key_dict[key] = [
                    {   'shape':currentContainer[key].shape,
                        'dtype':currentContainer[key].dtype,
                        'ndim':currentContainer[key].ndim,
                        'size':currentContainer[key].size,
                        'nbytes':currentContainer[key].nbytes,
                    },
                    {'shape':currentContainer[key].shape}
                ]
            # return
        if type(currentContainer[key]) == h5py._hl.group.Group:
            key_dict[key] = {}
            for sub_key in currentContainer[key].keys():
                compactKeyDict(currentContainer[key], key_dict[key], spaceVal+1)
    # return key_dict
def printCompactKeyDict(key_dict, spaceVal=0):
    for key in key_dict.keys():
        
        print(spaceVal*'\t',key,sep='')
        if type(key_dict[key]) == dict:
            printCompactKeyDict(key_dict[key], spaceVal+1)
        else:
            print((spaceVal+1)*'\t',key_dict[key][1],sep='')
if __name__ == "__main__":
    args = parser.parse_args()
    # Check if file exists
    if not os.path.exists(args.filename):
        print("File does not exist")
        exit(1)
    # Open hdf5 file using h5py
    f = h5py.File(args.filename, 'r')

    fileName = args.filename.split('/')[-1]
    fileDir = '/'.join(args.filename.split('/')[:-1])
    print(f'File directory: {fileDir}')
    print(f'File Name: {fileName}')
    # List all keys:
    # Old way:
    # print("Main Keys: %s" % f.keys())
    # recursiveHDF5List(f)
    print('--------------------')
    print('Recursive printing of keys and data types')
    key_dict = {}
    printCompactKeyDict(key_dict)
    compactKeyDict(f, key_dict)
    printCompactKeyDict(key_dict)
    print('Finished')
    # print(key_dict)
    print('--------------------')
    # List of the keys recursively as well as data types
    # def printname(name):
        # print(name)
    # f.visit(printname)

    # Display firt n data points
    print('--------------------')
    print(f'Display {args.nPoints} data points from position {args.shift}:')

    # for index in range(f['TARGETS']['t1']['b'].shape[0]):
    for index in range(args.nPoints):
        # check  = [False]*f['INPUTS'][list(key_dict['INPUTS'].keys())[0]]['MASK'][index:index+1].shape[1]
        # print('Mask for real vs duds:\n',f['INPUTS'][list(key_dict['INPUTS'].keys())[0]]['MASK'][index:index+1])
        for key in key_dict.keys():
            print(f'{key}:')
            for sub_key in key_dict[key].keys():
                print(f'\t{sub_key}:')
                for sub_sub_key in key_dict[key][sub_key].keys():
                    print(f'\t\t{sub_sub_key}:',f[key][sub_key][sub_sub_key][index:index+1])
        # print(f[dic])
        # if f['TARGETS']['t1']['b'][index:index+1][0] >= 0:
        #     check[f['TARGETS']['t1']['b'][index:index+1][0]] = True
        # if f['TARGETS']['t1']['q1'][index:index+1][0] >= 0:
        #     check[f['TARGETS']['t1']['q1'][index:index+1][0]] = True
        # if f['TARGETS']['t1']['q2'][index:index+1][0] >= 0:
        #     check[f['TARGETS']['t1']['q2'][index:index+1][0]] = True
        # if f['TARGETS']['t2']['b'][index:index+1][0] >= 0:
        #     check[f['TARGETS']['t2']['b'][index:index+1][0]] = True
        # if f['TARGETS']['t2']['q1'][index:index+1][0] >= 0:
        #     check[f['TARGETS']['t2']['q1'][index:index+1][0]] = True
        # if f['TARGETS']['t2']['q2'][index:index+1][0] >= 0:
        #     check[f['TARGETS']['t2']['q2'][index:index+1][0]] = True
        # print('Mask of jets found to be truth:\n', np.array([check]))
        # diff = np.logical_xor(f['INPUTS']['Source']['MASK'][index:index+1], np.array([check]))
        # print(f['TARGETS']['t1']['b'][index:index+1])
        # print(f['TARGETS']['t1']['q1'][index:index+1])
        # print(f['TARGETS']['t1']['q2'][index:index+1])
        # print(f['TARGETS']['t2']['b'][index:index+1])
        # print(f['TARGETS']['t2']['q1'][index:index+1])
        # print(f['TARGETS']['t2']['q2'][index:index+1])
    print('Finished')

    exit(0)
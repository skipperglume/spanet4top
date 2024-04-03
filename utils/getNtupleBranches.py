import re
import argparse
import ROOT
import math
import h5py
import numpy as np
import sys, os
import glob



def displayFoundFiles(fPaths : list):
        numFile = len(fPaths)
        print('==============================================================')
        print(f'Found {numFile} files:')
        print('==============================================================')
        for i in range(numFile):
            name = os.path.basename(fPaths[i])
            if name == '':
                print(f'ERROR: {fPaths[i]} is a directory')
                exit(1)
            else:
                print(name)
if __name__ == "__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument('-i', '--inloc', default='/home/timoshyd/RAC_4tops_analysis/ntuples/v06_BDT_SPANET_Input/nom', type=str, help='Input file location')
    parser.add_argument('-r', '--regex', default='', type=str, help='Regex to filter variables')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('-t', '--tree', type=str, default='nominal', help='Name of a tree in the root file')
    args = parser.parse_args()

    file_paths = []
    if os.path.isfile(args.inloc):
        file_paths.append(args.inloc)
    elif os.path.isdir(args.inloc):
        for filename in  glob.iglob(args.inloc+"**/**.root", recursive=True):
            if os.path.isfile(filename):
                file_paths.append(filename)
    if len(file_paths) == 0:
        print(f'No files found in {args.inloc}')
        exit(1)
    if args.verbose:
        displayFoundFiles(file_paths)
    f = ROOT.TFile(file_paths[0],'READ')
    objectNameList = []
    for i in f.GetListOfKeys():
        objectNameList.append(str(i.GetName()))
    print(objectNameList)
    if args.tree not in objectNameList:
        print(f'ERROR: {args.tree} not found in {file_paths[0]}')
        exit(1)
    nominal = f.Get(args.tree)
    branchNameList = []
    for i in nominal.GetListOfBranches():
        branchNameList.append(str(i.GetName()))

    if args.regex != '':
        branchNameList = [x for x in branchNameList if re.match(args.regex, x)]
        print(branchNameList)
    else:
        print(branchNameList)
        print('Try regex to filter variables:')
        print('1: \'\w{4,10}\d{1,2}.*\'')
        print('2: \'.*(t|T)op.*\'')
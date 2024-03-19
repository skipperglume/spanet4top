import argparse
import ROOT
import math
import h5py
import numpy as np
import sys, os
import glob



def displayFoundFiles(fPaths : list):
       numFile = len(fPaths)
       print(f'Found {numFile} files:')
       for i in range(numFile):
              name = fPaths[i][fPaths[i].find(args.inloc)+len(args.inloc):]
              if name[0] == '.':
                     name = name[1:]
              print(name)
if __name__ == "__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument('-i', '--inloc', default='/home/timoshyd/RAC_4tops_analysis/ntuples/v06_BDT_SPANET_Input/nom', type=str, help='Input file location')
    args = parser.parse_args()

    file_paths = []
    if os.path.isfile(args.inloc):
        file_paths.append(args.inloc)
    elif os.path.isdir(args.inloc):
        for filename in  glob.iglob(args.inloc+"**/**.root", recursive=True):
            if os.path.isfile(filename):
                file_paths.append(filename)
    displayFoundFiles(file_paths)
    f = ROOT.TFile(file_paths[0],'READ')
    print(f.GetListOfKeys())
    nominal = f.Get('nominal')
    branchNameList = []
    for  i in nominal.GetListOfBranches():
        branchNameList.append(str(i.GetName()))
    print(branchNameList)

import sys, os, glob
import argparse
import ROOT
import math
import h5py
import numpy as np
from tqdm import tqdm


def jetE_Eval(jet_e):
    return [ float(_) for _ in jet_e]
def mask_Eval(jet_e, maxJets):
    if len(jet_e) >= maxJets:
        return [True]*maxJets
    else:
        return [True] *len(jet_e) + [False] * (maxJets-len(jet_e))

def sin_phi_Eval( jet_phi ):
    return [math.sin(_) for _ in jet_phi]
def cos_phi_Eval( jet_phi ):
    return [math.cos(_) for _ in jet_phi]

def ptx_Eval(jet_pt, jet_phi ):
    result = []
    for i in range(len(jet_pt)):
        result.append(jet_pt[i]*math.cos(jet_phi[i]))
    return result
def pty_Eval(jet_pt, jet_phi ):
    result = []
    for i in range(len(jet_pt)):
        result.append(jet_pt[i]*math.sin(jet_phi[i]))
    return result

def identityListFunc(**kwargs):
    if len(kwargs) == 1:
        x = list(kwargs.values())[0]
        return list(x)
    else:
        print('Error: identityListFunc takes only one argument')
        return None

def identityFunc(**kwargs):
    if len(kwargs) == 1:
        x = list(kwargs.values())[0]
        return x
    else:
        print('Error: identityListFunc takes only one argument')
        return None

def identityIntFunc(**kwargs):
    if len(kwargs) == 1:
        x = list(kwargs.values())[0]
        return int(x)
    else:
        print('Error: identityListFunc takes only one argument')
        return None

def metx_Func(met_met, met_phi):
    return met_met * math.cos(met_phi)
def mety_Func(met_met, met_phi):
    return met_met * math.sin(met_phi)

def cos_Func(**kwargs):
    if len(kwargs) == 1:
        return math.cos(kwargs[list(kwargs.keys())[0]])
    else:
        print('Error: cos_Func takes only one argument')
        return None
def sin_Func(**kwargs):
    if len(kwargs) == 1:
        return math.sin(kwargs[list(kwargs.keys())[0]])
    else:
        print('Error: cos_Func takes only one argument')
        return None

variablesFuncDict = {
    'INPUTS/Source/MASK' : mask_Eval,
    'INPUTS/Source/jet_e' : jetE_Eval,
    'INPUTS/Source/jet_pt' : identityListFunc,
    'INPUTS/Source/jet_eta' : identityListFunc,
    'INPUTS/Source/sin_phi' : sin_phi_Eval,
    'INPUTS/Source/cos_phi' : cos_phi_Eval,
    'INPUTS/Source/jet_ptx' : ptx_Eval,
    'INPUTS/Source/jet_pty' : pty_Eval,
    'INPUTS/Source/jet_mass' : identityListFunc,
    'INPUTS/Source/jet_btag' : identityListFunc,
    'INPUTS/Met/cos_phi' : cos_Func,
    'INPUTS/Met/sin_phi' : sin_Func,
    'INPUTS/Met/met' :  identityFunc,
    'INPUTS/Met/met_x' : metx_Func,
    'INPUTS/Met/met_y' : mety_Func,

    'TARGETS/t1/b' : identityIntFunc,
    'TARGETS/t1/q1' : identityIntFunc,
    'TARGETS/t1/q2' : identityIntFunc,
    'TARGETS/t2/b' : identityIntFunc,
    'TARGETS/t2/q1' : identityIntFunc,
    'TARGETS/t2/q2' : identityIntFunc,
    'TARGETS/t3/b' : identityIntFunc,
    'TARGETS/t3/q1' : identityIntFunc,
    'TARGETS/t3/q2' : identityIntFunc,
    'TARGETS/t4/b' : identityIntFunc,
    'TARGETS/t4/q1' : identityIntFunc,
    'TARGETS/t4/q2' : identityIntFunc,
}
# Setting 
MAX_JETS = 18

class Container:
    def __init__(self, name , size):
        self.name = name
        self.size = size
    def trimList( self, input, name='' ):
        # check that input is indeed a list:
        if self.size == -1:
            return input
        if not isinstance(input, list):
            print('Error: trimList input is not a list')
            if name != '':
                print(f'Error in {name}')
            exit(0)
        if len(input) == self.size:
            return input
        elif len(input) > self.size:
            return input[:self.size]
        else:
            return input + [0.0]*(self.size-len(input))
    def setVarList(self, varArgDict : dict, varFuncDcit : dict):
        self.varArgsFunc = {}
        for var in varArgDict:
            self.varArgsFunc[var] = {}
            self.varArgsFunc[var]['varName'] = var
            self.varArgsFunc[var]['varValues'] = []
            self.varArgsFunc[var]['args'] = varArgDict[var]
            self.varArgsFunc[var]['func'] = varFuncDcit[var]
    def setTree(self, tree):
        self.tree = tree
    def evaluate(self, ievent):
        self.tree.GetEntry(ievent)
        # print(getattr(self.tree, 'jet_e'))
        for var in self.varArgsFunc:
            # if var == 'INPUTS/Source/jet_e':
                # print(self.varArgsFunc)
            argsFuncDict = {}
            for arg in self.varArgsFunc[var]['args']:
                if arg == 'maxJets':
                    argsFuncDict[arg] = self.size 
                else:
                    argsFuncDict[arg] = getattr(self.tree, arg)
                # self.varArgsFunc[var]['varValues'].append(getattr(self.tree, arg))
            # print(var, argsFunc)
            self.varArgsFunc[var]['varValues'].append(
                self.trimList(self.varArgsFunc[var]['func'](**argsFuncDict), var)
                )

    def __repr__(self) -> str:
        result = ''
        result += f'Container {self.name}\n'
        # for key in self.__dict__.keys():
            # result += f'{key} : {self.__dict__[key]}\n'
        return result

# These are varaibles to save for spanet input
# As well as their respective list of branch names in ntuples to calculate them from 
maskVariablesDict = {
    'INPUTS/Source/MASK'            :       ['jet_e', 'maxJets', ],
}
sequentialVariablesDict = {
    'INPUTS/Source/jet_e'           :       ['jet_e', ],
    'INPUTS/Source/jet_pt'          :       ['jet_pt', ],
    'INPUTS/Source/jet_eta'         :       ['jet_eta', ],
    'INPUTS/Source/cos_phi'         :       ['jet_phi', ],
    'INPUTS/Source/sin_phi'         :       ['jet_phi', ],
    'INPUTS/Source/jet_ptx'         :       ['jet_phi', 'jet_pt', ],
    'INPUTS/Source/jet_pty'         :       ['jet_phi', 'jet_pt', ],
    # 'INPUTS/Source/jet_mass'        :       ['jet_m', ],
    'INPUTS/Source/jet_btag'        :       ['jet_tagWeightBin_DL1dv01_Continuous', ],
}

globalVariablesDict = {
    'INPUTS/Met/cos_phi'        :           ['met_phi', ],
    'INPUTS/Met/sin_phi'         :          ['met_phi', ],
    'INPUTS/Met/met'        :               ['met_met', ],
    'INPUTS/Met/met_x'        :               ['met_met', 'met_phi', ],
    'INPUTS/Met/met_y'        :               ['met_met', 'met_phi', ],

}

targetsDict = {    
    'TARGETS/t1/b'  :                       ['t1_b_Jeti', ],
    'TARGETS/t1/q1'     :                   ['t1_w0_Jeti', ],
    'TARGETS/t1/q2'     :                   ['t1_w1_Jeti', ],
    'TARGETS/t2/b'  :                       ['t2_b_Jeti', ],
    'TARGETS/t2/q1'     :                   ['t2_w0_Jeti', ],
    'TARGETS/t2/q2'     :                   ['t2_w1_Jeti', ],
    'TARGETS/t3/b'  :                       ['t3_b_Jeti', ],
    'TARGETS/t3/q1'     :                   ['t3_w0_Jeti', ],
    'TARGETS/t3/q2'     :                   ['t3_w1_Jeti', ],
    'TARGETS/t4/b'  :                       ['t4_b_Jeti', ],
    'TARGETS/t4/q1'     :                   ['t4_w0_Jeti', ],
    'TARGETS/t4/q2'     :                   ['t4_w1_Jeti', ],
}


def source(root_files : list, args: argparse.Namespace):
    eventNumber_list = []
    containerList = []
    nfiles=len(root_files)
    seqContainer = Container('Sequential', args.maxjets)
    seqContainer.setVarList(sequentialVariablesDict, variablesFuncDict)
    containerList.append(seqContainer)
    
    gloContainer = Container('Global', -1)
    gloContainer.setVarList(globalVariablesDict, variablesFuncDict)
    containerList.append(gloContainer)

    maskContainer = Container('Mask', args.maxjets)
    maskContainer.setVarList(maskVariablesDict, variablesFuncDict)
    containerList.append(maskContainer)
    
    targContainer = Container('Target', -1)
    targContainer.setVarList(targetsDict, variablesFuncDict)
    containerList.append(targContainer)
    
    

    for ifile, rf in enumerate(root_files):
            print("Processing file: "+rf+" ("+str(ifile+1)+"/"+str(nfiles)+")")
            f = ROOT.TFile(rf, 'READ')

            tree = f.Get(args.treename)
            events = tree.GetEntries()

            seqContainer.setTree(tree)
            gloContainer.setTree(tree)
            maskContainer.setTree(tree)
            targContainer.setTree(tree)
            events = min(events, 3)
            for i in tqdm(range(events)):
                eventNumber_list.append(tree.eventNumber)
                seqContainer.evaluate(i)
                gloContainer.evaluate(i)
                maskContainer.evaluate(i)
                targContainer.evaluate(i)
                pass
            f.Close()
    outfiles = []
    modulusList = []
    remainderList = []
    if args.oddeven: # cross training based on eventNumber
        outfiles.append(args.outloc+"_even.h5")
        modulusList.append(2)
        remainderList.append(0)
        outfiles.append(args.outloc+"_odd.h5")
        modulusList.append(2)
        remainderList.append(1)
    else:
        outfiles.append(args.outloc+".h5")
        modulusList.append(1)
        remainderList.append(0)

    # Create "source" group in HDF5 file, adding feature data sets
    for out, modulus, remainder in zip(outfiles, modulusList, remainderList):
        indices = np.where(np.array(eventNumber_list) % modulus == remainder)
        print(f'Outpufile: {out}')
        print(f'indices for {remainder} after division by {modulus}:',indices)
        # Print out only elements that are in indices:
        print('eventNumber_list:',np.array(eventNumber_list)[indices])
        indices = np.where(np.array(eventNumber_list) % modulus == remainder)
        write(out, args.topo, indices, containerList)

    return containerList

def write(outloc, topo, indices, containerList):
    HDF5 = h5py.File(outloc, 'w')
    inputs_group = HDF5.create_group('INPUTS')
    jet_group = inputs_group.create_group('Source')
    pass
def displayFoundFiles(fPaths : list):
       numFile = len(fPaths)
       print(f'Found {numFile} files:')
       for i in range(numFile):
              name = fPaths[i][fPaths[i].find(args.inloc)+len(args.inloc):]
              if name[0] == '.':
                     name = name[1:]
              print(name)
if __name__ == '__main__':

    file_paths = []

    parser=argparse.ArgumentParser()
    parser.add_argument('-i', '--inloc', default='/home/timoshyd/RAC_4tops_analysis/ntuples/v06_BDT_SPANET_Input/nom', type=str, help='Input file location')
    parser.add_argument('-t', '--topo', default='ttbar', type=str, help='Topology of underlying event')
    parser.add_argument('-o', '--outloc', default='/home/timoshyd/spanet4Top/ntuples/four_top_SPANET_input/four_top_SPANET_input', type=str, help='Output location')
    parser.add_argument('-m', '--maxjets', default=18, type=int, help='Max number of jets')
    parser.add_argument('-tr', '--treename', default='nominal', type=str, help='Name of nominal tree')
    parser.add_argument('-b', '--btag', type=str, default='DL1r', help='Which btag alg to use')
    parser.add_argument('--oddeven', action='store_false' , help='Split into odd and even events')
    parser.add_argument('--test', action='store_true', help='Test the code')
    parser.add_argument('-p','--prefix',default='modeALLHAD_', help='Prefix to separate files')
    parser.add_argument('-s','--suffix',default='_year18', help='Prefix to separate files')
    args = parser.parse_args()

    if args.prefix != '' and args.suffix != '':
        print('Old output location:', args.outloc)
        if args.prefix != '':
            listNameParts = args.outloc.split('/')
            listNameParts[-1] = args.prefix + listNameParts[-1]
            args.outloc = '/'.join(listNameParts)
        if args.suffix != '':
            listNameParts = args.outloc.split('/')
            listNameParts[-1] = listNameParts[-1] + args.suffix
            args.outloc = '/'.join(listNameParts)
        print('New output location:', args.outloc)    
    else: 
        print('Output location:', args.outloc)
    print('===========================================================')
    if os.path.isfile(args.inloc):
        file_paths.append(args.inloc)
    elif os.path.isdir(args.inloc):
        for filename in  glob.iglob(args.inloc+"**/**.root", recursive=True):
            #print(filename)
            if os.path.isfile(filename):
                file_paths.append(filename)
    displayFoundFiles(file_paths)

    selectionSubstring = [ 
                                   # 'user.nhidic.412043.aMcAtNloPythia8EvtGen.DAOD_PHYS.e7101_a907_r14859_p5855.4thad26_240130_v06.3_output_root.nominal.root',
                                   # 'user.nhidic.412043.aMcAtNloPythia8EvtGen.DAOD_PHYS.e7101_a907_r14860_p5855.4thad26_240130_v06.3_output_root.nominal.root',
                                   'user.nhidic.412043.aMcAtNloPythia8EvtGen.DAOD_PHYS.e7101_a907_r14861_p5855.4thad26_240130_v06.3_output_root.nominal.root',
                                   ]

    if not len(selectionSubstring) ==0 :
        file_paths = [x  for x in file_paths  if any(substring in x for substring in selectionSubstring)]
        print('===========================================================')
        print('Final set of files:')
        displayFoundFiles(file_paths)

    result = source(file_paths, args)
    print(result)
    print(result[0])
    print(result[1])
    print(result[1].varArgsFunc.keys())
    print(result[0].varArgsFunc)
    print(result[1].varArgsFunc)
    print(result[2].varArgsFunc)
    print(result[3].varArgsFunc)
    # dict_keys(['varName', 'varValues', 'args', 'func'])
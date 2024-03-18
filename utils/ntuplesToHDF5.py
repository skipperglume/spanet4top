import sys, os
import argparse
import ROOT
import math
import h5py
import numpy as np
from tqdm import tqdm

# Setting 
class Contaner:
    def __init__(self, **kwargs):
        self.varaibles = kwargs.keys()
        
        self.__dict__.update(kwargs)
    def __repr__(self) -> str:
        result = ''
        for key in self.__dict__.keys():
            result += f'{key} : {self.__dict__[key]}\n'
        return result

# These are varaibles to save for spanet input
# As well as their respective list of branch names in ntuples to calculate them from 
sequentialVariablesDict = {
    'INPUTS/Source/MASK'            :       ['jet_e', ],
    'INPUTS/Source/jet_e'           :       ['jet_e', ],
    'INPUTS/Source/jet_pt'          :       ['jet_pt', ],
    'INPUTS/Source/jet_eta'         :       ['jet_eta', ],
    'INPUTS/Source/cos_phi'         :       ['jet_phi', ],
    'INPUTS/Source/sin_phi'         :       ['jet_phi', ],
    'INPUTS/Source/jet_ptx'         :       ['jet_phi', 'jet_pt', ],
    'INPUTS/Source/jet_pty'         :       ['jet_phi', 'jet_pt', ],
    
    'INPUTS/Source/jet_mass'        :       ['jet_mass', ],
    'INPUTS/Source/jet_btag'        :       ['jet_btag', ],
}

globalVariablesDict = {
    'INPUTS/Met/cos_phi'        :           ['jet_mass', ],
    'INPUTS/Met/sin_phi'         :          ['jet_phi', ],
    'INPUTS/Met/met'        :               ['jet_btag', ],
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

def jetE_Eval(jet_e):
    return jet_e
def mask_Eval(jet_e):
    return jet_e

def px_Eval(jet_pt, jet_eta, jet_phi ):
    return jet_pt

variablesFuncDict = {
    'INPUTS/Source/MASK' : mask_Eval,
    'jet_e' : jetE_Eval,
    'jet_pt' : jetE_Eval,
    'jet_eta' : jetE_Eval,
    'cos_phi' : jetE_Eval,
    'sin_phi' : jetE_Eval,
    'jet_mass' : jetE_Eval,
    'jet_btag' : jetE_Eval,
}
if __name__ == '__main__':

    list1 = [1,1]

    initContainer = {
        'name' : 'contForDHF5',
    }
    for varName in variablesDict:
        initContainer[varName] = []
    cont = Contaner( **initContainer)
    list1.append(2)
    print( cont)
    import time
    for i in tqdm(range(10)):
        print(i)
        # wait for 1 sec
        time.sleep(1)




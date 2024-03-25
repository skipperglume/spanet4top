#!/usr/bin/env python3
import argparse
import ROOT
import math
import h5py
import numpy as np
import sys, os
import glob
import re
from tqdm import tqdm

def eTOm(feats):
       """
       Find jet masses by using TLorentzVector().SetPtEtaPhiE(pt, eta, phi, e), 
       where e = nominal.jet_e since there is no jet_m, then finding the
       mass of the LorentzVector with the M() method.
       """
       # Creating TLorentzVectors for each jet
       vects = [ROOT.TLorentzVector() for i in feats[0]]
       # SetPtEtaPhiE for each jet
       for i in range(len(feats[0])):
              vects[i].SetPtEtaPhiE(feats[0][i], feats[1][i], feats[2][i], feats[3][i])
       # Using .M() method on each jet to get jet masses
       m_ls = [j.M() for j in vects]
       return m_ls

def get_max_jets(root_files, treename='nominal', topo='ttbar'):
    """
    Find MAX_JETS for the data set
    """
    # Iterating over files to find MAX_JETS of data set
    print("Finding max number of jets.")

    MAX_JETS = 0
    nfiles=len(root_files)
    
    for ifile, rf in enumerate(root_files):    
        f = ROOT.TFile(rf)
        print(str(ifile)+"/"+str(nfiles))
        nominal=f.Get(treename)

        events = nominal.GetEntries()
        print("events:", events)
        for i in range(events):
            nominal.GetEntry(i)
            # Use pt lengths to determine amount of jets in event
            pt_ls = list(nominal.jet_pt)

            njets = len(pt_ls)
            # if ljets, add space for leptons and neutrinos (potentially two nu solns!)
            if 'ljets' in topo: njets = njets
                    
            # If # of jets+leptons is greater than previous events
            if njets > MAX_JETS:
                MAX_JETS = njets
        print("MAXJETS="+str(MAX_JETS))
        maxjetsfile=open("maxjets"+topo+".txt", "w")
        maxjetsfile.write(str(MAX_JETS))
        f.Close()

    return(MAX_JETS)


def number_of_jets( topTargets : list ) -> int:
       result = 0 
       for i in range(len(topTargets)):
              if topTargets[i] != -1:
                     result += 1
       return result

def goodEvent( tree, assignmentDict : dict, args : argparse.Namespace) -> bool:
    
    if tree.n_jet < 8: return False
    for jetParton in assignmentDict:
        if assignmentDict[jetParton] >= args.maxjets:
            return False
    
    uniqueSet = set([assignmentDict[jetParton] for jetParton in assignmentDict if assignmentDict[jetParton] != -1])
    notUnique = [assignmentDict[jetParton] for jetParton in assignmentDict if assignmentDict[jetParton] != -1]
    indexDict = {}
    for jetParton in assignmentDict:
        if assignmentDict[jetParton] != -1:
            if assignmentDict[jetParton] not in indexDict:
                indexDict[assignmentDict[jetParton]] = 1
            else:
                indexDict[assignmentDict[jetParton]] += 1

    if len(uniqueSet) != len(notUnique):
        # print('Not unique:  ', sorted(notUnique))
        # print('Unique:      ',uniqueSet)
        # print(indexDict)
        return False

    


    # if (number_of_jets([t1q1, t1q2, t1b] ) < 2 ) or (number_of_jets([t2q1, t2q2, t2b] ) < 2 ) or (number_of_jets([t3q1, t3q2, t3b] ) < 2 ) or (number_of_jets([t4q1, t4q2, t4b] ) < 2 ): 
        # return False
    for targetTuple in [('t1/b','t1/q1','t1/q2'),('t2/b','t2/q1','t2/q2'),('t3/b','t3/q1','t3/q2'),('t4/b','t4/q1','t4/q2')]:
        if all([assignmentDict[target] == -1 for target in targetTuple]):
            # print(targetTuple, [assignmentDict[target]  for target in targetTuple])
            return False
        if sum([assignmentDict[target] == -1 for target in targetTuple]) > 1:
            # print(targetTuple, [assignmentDict[target]  for target in targetTuple])
            return False
                     
    # if not uniqness: 
    #     return False
    return True

def source(root_files : list, args: argparse.Namespace):
        """
        Create HDF5 file and create the "source" group in the HDF5 file.
        """
        # Creating HDF5 file, setting MAX_JETS to 0, and opening all provided ROOT files
       
        #files = [ROOT.TFile(rf) for rf in root_files]

        # List of feature name to save:
        featuresToSave = [
            'INPUTS/Source/MASK',
            'INPUTS/Source/pt_x',
            'INPUTS/Source/pt_y',
            # 'INPUTS/Source/pt',
            'INPUTS/Source/eta',
            'INPUTS/Source/e',
            # 'INPUTS/Source/sin_phi',
            # 'INPUTS/Source/cos_phi',
            'INPUTS/Source/btag',
            # 'INPUTS/Met/met',
            'INPUTS/Met/met_x',
            'INPUTS/Met/met_y',
            # 'INPUTS/Met/sin_phi',
            # 'INPUTS/Met/cos_phi',
            'TARGETS/t1/b',
            'TARGETS/t1/q1',
            'TARGETS/t1/q2',
            'TARGETS/t2/b',
            'TARGETS/t2/q1',
            'TARGETS/t2/q2',
            'TARGETS/t3/b',
            'TARGETS/t3/q1',
            'TARGETS/t3/q2',
            'TARGETS/t4/b',
            'TARGETS/t4/q1',
            'TARGETS/t4/q2',
        ]

        # INPUTS for SPANet       
        # Jets
        # Features lists that will be appended to and will have dimensions (EVENTS, MAX_JETS)
        inputDict = {}
        inputDict['INPUTS/Source/MASK'] = []
        # inputDict['INPUTS/Source/mass'] = []
        inputDict['INPUTS/Source/pt_x'] = []
        inputDict['INPUTS/Source/pt_y'] = []
        inputDict['INPUTS/Source/pt'] = []
        inputDict['INPUTS/Source/eta'] = []
        inputDict['INPUTS/Source/e'] = []
        inputDict['INPUTS/Source/sin_phi'] = []
        inputDict['INPUTS/Source/cos_phi'] = []
        inputDict['INPUTS/Source/btag'] = []

        # Only one values per event 
        # MET
        inputDict['INPUTS/Met/met'] = []
        inputDict['INPUTS/Met/met_x'] = []
        inputDict['INPUTS/Met/met_y'] = []
        inputDict['INPUTS/Met/sin_phi'] = []
        inputDict['INPUTS/Met/cos_phi'] = []
        # Auxiliary variables for the output preparation
        inputDict['AUX/aux/eventNumber'] = []
        inputDict['AUX/aux/decayType'] = []
        # inputDict['AUX/aux/extrajet_list'] = []
        # inputDict['AUX/aux/empty_list'] = []


        # TARGETS
        # Instatiating jet and mask lists
        inputDict['TARGETS/t1/b'] = []
        inputDict['TARGETS/t1/q1'] = []
        inputDict['TARGETS/t1/q2'] = []
        inputDict['TARGETS/t2/b'] = []
        inputDict['TARGETS/t2/q1'] = []
        inputDict['TARGETS/t2/q2'] = []
        inputDict['TARGETS/t3/b'] = []
        inputDict['TARGETS/t3/q1'] = []
        inputDict['TARGETS/t3/q2'] = []
        inputDict['TARGETS/t4/b'] = []
        inputDict['TARGETS/t4/q1'] = []
        inputDict['TARGETS/t4/q2'] = []
        


        # TODO: fix this part for more events
        # SPANet requires all particles to decay to at least two, to introduce a single jet, you add an empty particle
       

        nfiles=len(root_files)
        # Iterating over files to extract data, organize it, and add it to HDF5 file
        for ifile, rf in enumerate(root_files):
            print("Processing file: "+rf+" ("+str(ifile+1)+"/"+str(nfiles)+")")

            # Get ROOT TFile
            f = ROOT.TFile(rf)

            nominal = f.Get(args.treename)

            # this code might be useful if the truth info is stored in a different tree
            # truth = f1.Get("truth")
            # truth_events, nominal_events = truth.GetEntries(), nominal.GetEntries()
            # truth.BuildIndex("runNumber", "eventNumber")
                     
            events = nominal.GetEntries()

            # Counters to display need info about our samples
            all_had_count = 0
            non_all_had_count = 0
            for i in tqdm(range(events)):
                # Early stopping for testing
                if i > 30 and args.test: break
                #print(i)
                if i % 50000 == 0: 
                    print(str(i)+"/"+str(events))
                    print('Currently collected events: ',len(inputDict['AUX/aux/eventNumber']))
                nominal.GetEntry(i)
                # Now do the particle groups, ie the truth targets
                assignmentDict = assignIndicesljetsttbar(nominal, args)
                # One could apply cuts here if desired, but usually inclusive training is best!
                # print(assignmentDict)
                if not goodEvent(nominal, assignmentDict, args): continue
                
                # TODO: check variable - truthTop_isHadTauDecay
                isLepDecay = [ float(x.encode("utf-8").hex())  for x in list(nominal.truthTop_isLepDecay)]
                if sum(isLepDecay) > 0:
                    non_all_had_count += 1
                else:
                    all_had_count += 1

                # Feature lists for: pt, eat, phi, ls - that hold this info for ONLY CURRENT EVENT
                eventDict = {}
                eventDict['INPUTS/Source/pt_x'] = [float(nominal.jet_pt[_]) * math.cos(float(nominal.jet_phi[_])) for _ in range(len(nominal.jet_phi))]
                eventDict['INPUTS/Source/pt_y'] = [float(nominal.jet_pt[_]) * math.sin(float(nominal.jet_phi[_])) for _ in range(len(nominal.jet_phi))]
                eventDict['INPUTS/Source/pt'] = [float(_) for _ in nominal.jet_pt]
                eventDict['INPUTS/Source/eta'] = [float(_) for _ in nominal.jet_eta]
                eventDict['INPUTS/Source/e'] = [float(_) for _ in nominal.jet_e]
                eventDict['INPUTS/Source/sin_phi'] = [math.sin(_) for _ in nominal.jet_phi]
                eventDict['INPUTS/Source/cos_phi'] = [math.cos(_) for _ in nominal.jet_phi]
                eventDict['INPUTS/Source/btag'] = [float(_) for _ in nominal.jet_tagWeightBin_DL1dv01_Continuous]
                
                eventDict['INPUTS/Met/met'] = nominal.met_met
                eventDict['INPUTS/Met/met_phi'] = nominal.met_phi
                eventDict['INPUTS/Met/met_x'] = nominal.met_met * math.cos(nominal.met_phi)
                eventDict['INPUTS/Met/met_y'] = nominal.met_met * math.sin(nominal.met_phi)
                eventDict['INPUTS/Met/sin_phi'] = math.sin(nominal.met_phi)
                eventDict['INPUTS/Met/cos_phi'] = math.cos(nominal.met_phi)
                        
                eventDict['AUX/aux/eventNumber'] = nominal.eventNumber
                eventDict['AUX/aux/decayType'] = sum([ float(x.encode("utf-8").hex())  for x in list(nominal.truthTop_isLepDecay)])
                # Adding btag values according to WP
                # Continuous DL1r tagger score
                                   
                # Getting mass of each jet(or lep) with eTOm function
                # jet_m_ls = eTOm([jet_pt_ls, jet_eta_ls, jet_phi_ls, jet_e_ls])
                # eventDict['INPUTS/Source/mass'] = 
                # TODO:

                # Source's "mask" is given True for positions with a jet(or lep) and False when empty
                if len(nominal.jet_pt) >= MAX_JETS:
                    eventDict['INPUTS/Source/MASK'] = [True] * MAX_JETS
                else:
                    eventDict['INPUTS/Source/MASK'] = [True] * len(nominal.jet_pt) + [False] *  (MAX_JETS-len(nominal.jet_pt))
                     
                # Padding: Adding 0.0 to feature lists until they are all the same length
                # Maybe we want to revisit this to set sin and cos to some incorrect values (-100.0)
                # Making all feature lists the same length so they can be np.array later
                for varName in ['pt_x','pt_y', 'pt','eta','e','sin_phi','cos_phi','btag'] :
                    if len(eventDict[f'INPUTS/Source/{varName}']) >= MAX_JETS:
                        eventDict[f'INPUTS/Source/{varName}'] = eventDict[f'INPUTS/Source/{varName}'][:MAX_JETS]
                    while len(eventDict[f'INPUTS/Source/{varName}']) < MAX_JETS:
                        eventDict[f'INPUTS/Source/{varName}'].append(0.0)
                # Appending event feature lists to data set ((EVENTS, MAX_JETS)) feature lists


                for varName in ['pt_x','pt_y', 'pt','eta','e','sin_phi','cos_phi','btag', 'MASK'] :
                    inputDict[f'INPUTS/Source/{varName}'].append(eventDict[f'INPUTS/Source/{varName}'])

                inputDict['AUX/aux/eventNumber'].append(eventDict['AUX/aux/eventNumber'])
                inputDict['AUX/aux/decayType'].append(eventDict['AUX/aux/decayType'])
                # Global feature lists for: met, met_phi, ls - ONLY one value per event
                # Appending event features with onlhy one entry per event (MET)
                inputDict['INPUTS/Met/met'].append(eventDict['INPUTS/Met/met'])
                inputDict['INPUTS/Met/sin_phi'].append(eventDict['INPUTS/Met/sin_phi'])
                inputDict['INPUTS/Met/cos_phi'].append(eventDict['INPUTS/Met/cos_phi'])
                inputDict['INPUTS/Met/met_x'].append(eventDict['INPUTS/Met/met_x'])
                inputDict['INPUTS/Met/met_y'].append(eventDict['INPUTS/Met/met_y'])

                #Get the REGRESSIONS that you want to estimate

                # Appending the variables for the REGRESSIONS
                
                # Appending the variables for the TARGETS
                for targetName in ['t1/b','t1/q1','t1/q2','t2/b','t2/q1','t2/q2','t3/b','t3/q1','t3/q2','t4/b','t4/q1','t4/q2']:
                    inputDict[f'TARGETS/{targetName}'].append(assignmentDict[targetName])

                # empty = -1
                # empty_list.append(empty)

            # Close ROOT files
            f.Close()
        print('Number of All hadronic:', all_had_count)
        print('Number of Non All hadronic:', non_all_had_count)
              
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
            indices = np.where(np.array(inputDict['AUX/aux/eventNumber']) % modulus == remainder)
            print(f'Outpufile: {out}')
            print(f'indices for {remainder} after division by {modulus}:', len(indices[0]))
            # Print out only elements that are in indices:
            print('eventNumber_list:',len(np.array(inputDict['AUX/aux/eventNumber'])[indices]))
            decayDict = { }
            if args.ignoreDecay:
                ...
            else:
                allHad_indices = np.where(np.array(inputDict['AUX/aux/decayType']) == 0.0)
                onePlusSemi_indices = np.where(np.array(inputDict['AUX/aux/decayType']) != 0)
                decayDict['allHad'] = allHad_indices
                decayDict['onePlusSemi'] = onePlusSemi_indices
                
            write( out, args.topo, featuresToSave, indices, inputDict, decayDict )
                     



def write(outloc : str, topo : str, featuresToSave : list, indices : np.array, inputDict, decayDict : dict):
    """
        Function to create and write to the HDF5 file
    """
    # print('Indices Size: ',len(indices[0]))
    # print('Mask Size: ',len(jet_mask_list),len(jet_mask_list[0]))
    # print('Mass Size: ',len(jet_mass_list))
       
    HDF5 = h5py.File(outloc, 'w')
    for featureName in featuresToSave:
        if featureName not in inputDict:
            print(f'ERROR: {featureName} not found in inputDict')
            exit(1)
        if 'AUX/' in featureName:
            continue
        if 'MASK' in featureName:
            HDF5.create_dataset(featureName, data=np.array(inputDict[featureName], dtype='bool')[indices])
        elif 'TARGETS' in featureName:
            HDF5.create_dataset(featureName, data=np.array(inputDict[featureName], dtype=np.int32)[indices])
        else:
            HDF5.create_dataset(featureName, data=np.array(inputDict[featureName], dtype=np.float32)[indices])

    #INPUTS group
    # inputsGroup = HDF5.create_group('INPUTS')


    # sourceGroup = inputsGroup.create_group('Source')
    # jet_mask_set = sourceGroup.create_dataset('MASK', data=np.array(jet_mask_list, dtype='bool')[indices])
       
    # jet_mass_set = jet_group.create_dataset('mass', data=np.array(jet_mass_list, dtype=np.float32)[indices])


    HDF5.close()
def targetsAreUnique(targets : list) -> bool:
    for i in targets: 
        if targets.count(i) > 1 and i != -1:
            # mem = []
            # for valIndex in range(len(targets)): 
                            # if targets[valIndex] == i:
                                   # mem.append(valIndex)
                                   # print(f'Index: {valIndex} Value: {i}')
                     # if len( set([_//3 for _ in mem]) )!= 1:
                            # print('Incorect:',mem)
                            # print(targets)
            return False
    return True

def compareResults( spaNet, root):
    ['topCandidate_smallJetIndexReco', 'topCandidate_largeJetIndexReco']
    pass
def getRadius(eta : float):
    defaultR = 0.4
    if abs(eta) < 2.5:
        return defaultR
    else:
        return defaultR + 0.1*(abs(eta)-2.5)
def findPartonJetPairs(nominal, args) -> dict:
    '''
    Method to find pairings between parton with detected jets. 
    The assignment target arrays contain the indices of each assignment. 
    Only sequential input indices may be assignment targets. 
    Each reconstruction target should be associated with exactly one input vector. 
    These indices must also be strictly unique. 
    Any targets which are missing within an event should be marked with -1.
    '''
    # ['truthTop_pt', 'truthTop_eta', 'truthTop_phi', 'truthTop_e']
    # ['truthTop_b_pt', 'truthTop_b_eta', 'truthTop_b_phi', 'truthTop_b_e'] 
    # ['truthTop_W_pt', 'truthTop_W_eta', 'truthTop_W_phi', 'truthTop_W_e']
    # ['truthTop_W_child1_pt', 'truthTop_W_child1_eta', 'truthTop_W_child1_phi', 'truthTop_W_child1_e']
    # 'truthTop_W_child1_pdgId'
    # ['truthTop_W_child2_pt', 'truthTop_W_child2_eta', 'truthTop_W_child2_phi', 'truthTop_W_child2_e']
    # 'truthTop_W_child2_pdgId'
    # ['truthTop_truthJet_pt', 'truthTop_truthJet_eta', 'truthTop_truthJet_phi', 'truthTop_truthJet_e']
    # 'truthTop_truthJet_index' 'truthTop_truthJet_flavor',  
    result = {}    
    partonLVs = {}
    jetLVs = {}
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('Targets to Reconstruct:', args.reconstruction.split('|'))
    groupedNames = {}
    for group in args.reconstruction.split('|'):
        subgroups = group.split('(')
        groupedNames[subgroups[0]] = subgroups[1][:-1].split(',')
    print(groupedNames)
    print(f'Total partons to match:', sum( [len(groupedNames[_]) for _ in groupedNames]) )
    for particle in groupedNames.keys():
        topIter = int(particle[1])-1
        # partonLVs[f'{particle}/type'] = int(nominal.truthTop_isLepDecay[topIter].encode("utf-8").hex())
        
        for parton in groupedNames[particle]:
            partonLVs[f'{particle}/{parton}'] = ROOT.TLorentzVector()
            if parton == 'b':
                partonLVs[f'{particle}/{parton}'].SetPtEtaPhiE(nominal.truthTop_b_pt[topIter]  , nominal.truthTop_b_eta[topIter], nominal.truthTop_b_phi[topIter], nominal.truthTop_b_e[topIter])
            elif parton == 'q1':
                partonLVs[f'{particle}/{parton}'].SetPtEtaPhiE(nominal.truthTop_W_child1_pt[topIter]  , nominal.truthTop_W_child1_eta[topIter], nominal.truthTop_W_child1_phi[topIter], nominal.truthTop_W_child1_e[topIter])
            elif parton == 'q2':
                partonLVs[f'{particle}/{parton}'].SetPtEtaPhiE(nominal.truthTop_W_child2_pt[topIter]  , nominal.truthTop_W_child2_eta[topIter], nominal.truthTop_W_child2_phi[topIter], nominal.truthTop_W_child2_e[topIter])
    for jetIter in range(len(nominal.jet_pt)):
        jetLVs[jetIter] = ROOT.TLorentzVector()
        jetLVs[jetIter].SetPtEtaPhiE(nominal.jet_pt[jetIter], nominal.jet_eta[jetIter], nominal.jet_phi[jetIter], nominal.jet_e[jetIter])
    
    
    costMatrix = {}

    # print(jetLVs, partonLVs)
    for jetLabel in jetLVs:
        for partonLabel in partonLVs:
            if 'type' in partonLabel: continue
            if partonLabel not in costMatrix:
                costMatrix[partonLabel] = {}
            if partonLVs[partonLabel].DeltaR(jetLVs[jetLabel]) < getRadius(partonLVs[partonLabel].Eta())  :
                costMatrix[partonLabel][jetLabel] = partonLVs[partonLabel].DeltaR(jetLVs[jetLabel])
                if nominal.jet_tagWeightBin_DL1dv01_Continuous[jetLabel] >= 3 and 'b' in partonLabel:
                    costMatrix[partonLabel][jetLabel] *= (-1)
                if nominal.jet_tagWeightBin_DL1dv01_Continuous[jetLabel] >= 4 and 'b' not in partonLabel:
                    costMatrix[partonLabel][jetLabel] *= (-1)
    partonToJets = {}
    jetToPartons = {}
    
    return result

    for partonLabel in costMatrix:
        for jetLabel in costMatrix[partonLabel]:
            if partonLabel not in partonToJets:
                partonToJets[partonLabel] = [jetLabel]
            else:
                partonToJets[partonLabel].append(jetLabel)
            if jetLabel not in jetToPartons:
                jetToPartons[jetLabel] = [partonLabel]
            else:
                jetToPartons[jetLabel].append(partonLabel)

    print(f'Cost Matrix: {len(costMatrix)}:-')
    print(costMatrix)
    print(f'Parton-Jet pairs: {len(partonToJets)}')
    print(partonToJets)
    print(f'Jet-Parton pairs: {len(jetToPartons)}')
    print(jetToPartons)
    print(f'Number of definitive matches: {min( len(partonToJets), len(jetToPartons) )}')
    print('Topology: ')
    print(partonLVs['t1/type'],partonLVs['t2/type'], partonLVs['t3/type'], partonLVs['t4/type'])
    usedJets = []
    for partonLabel in costMatrix:
        if len(costMatrix[partonLabel])==0:
            result[partonLabel] = -1
            continue
        
        if len(partonToJets[partonLabel]) == 1 and partonToJets[partonLabel][0] not in usedJets:
            result[partonLabel] = partonToJets[partonLabel][0]
            usedJets.append(result[partonLabel])
    for partonLabel in costMatrix:
        if partonLabel in result: continue
        if len(partonToJets[partonLabel]) > 1 and partonLabel not in  result:
            for jetLabel in partonToJets[partonLabel]:
                if jetLabel not in usedJets:
                    result[partonLabel] = jetLabel
                    usedJets.append(jetLabel)
                    print('Happend \'ere!')
                    break
        if partonLabel not in result:
            result[partonLabel] = -1 
    topLVs = {}
    for partonLabel in result:
        if 't' in partonLabel:
            if result[partonLabel] != -1:
                if partonLabel.split('/')[0] not in topLVs :
                    topLVs[partonLabel.split('/')[0]] = jetLVs[result[partonLabel]]
                else:
                    topLVs[partonLabel.split('/')[0]] += jetLVs[result[partonLabel]]
    print(topLVs)
    for topIter in range(len(nominal.truthTop_pt)):
        mass = (partonLVs[f't{topIter+1}/b']+partonLVs[f't{topIter+1}/q2']+partonLVs[f't{topIter+1}/q1']).M()*0.001

        matchedMass = 0

        print(f't{topIter+1}',partonLVs[f't{topIter+1}/type'],',',mass)
        if f't{topIter+1}' in topLVs:
            print( ', ',topLVs[f't{topIter+1}'].M()*0.001)
    numberAssigned = 0
    for partonLabel in result:
        if result[partonLabel] != -1:
            numberAssigned += 1
    if numberAssigned != min(len(partonToJets), len(jetToPartons)):
        print(f'CHECK THIS ONE: {numberAssigned} | {min(len(partonToJets), len(jetToPartons))}')
    if abs(numberAssigned - min(len(partonToJets), len(jetToPartons))) > 1 :
        print(f'Serious! - {abs(numberAssigned - min(len(partonToJets), len(jetToPartons)))}')
    return result



def assignIndicesljetsttbar( nominal, args : argparse.Namespace) -> dict:
    """
    Here is where you need to do the truth matching
    """
    result = {
        't1/b' : -1,
        't1/q1' : -1,
        't1/q2' : -1,
        't2/b' : -1,
        't2/q1' : -1,
        't2/q2' : -1,
        't3/b' : -1,
        't3/q1' : -1,
        't3/q2' : -1,
        't4/b' : -1,
        't4/q1' : -1,
        't4/q2' : -1,
    }
    
    for topIter in range(4):
        topIter +=1
        for parton in ('b','w0','w1'):
            # result[f't{topIter}/{parton}'] = nominal.t1_b_Jeti
            spaNetParton = f't{topIter}/'
            if parton == 'b':
                spaNetParton += 'b'
            if parton[0] == 'w':
                spaNetParton += f'q{1+int(parton[1])}'
            rootParton = f't{topIter}_{parton}_Jeti'
            # print(spaNetParton, rootParton)
            result[spaNetParton] = getattr(nominal, rootParton)
    

    newAss = findPartonJetPairs(nominal, args)
    print('New Assignment:')
    print( newAss)
    print('Old Assignment:')
    print(result)
    return result

        
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
       
    file_paths = []

    parser=argparse.ArgumentParser()
    parser.add_argument('-i', '--inloc', default='/home/timoshyd/RAC_4tops_analysis/ntuples/v06_BDT_SPANET_Input/nom', type=str, help='Input file location')
    parser.add_argument('-r', '--reconstruction', default='t1(b,q1,q2)|t2(b,q1,q2)|t3(b,q1,q2)|t4(b,q1,q2)', type=str, help='Topology of underlying event')
    parser.add_argument('-t', '--topo', default='tttt', type=str, help='Topology of underlying event')
    parser.add_argument('-o', '--outloc', default='/home/timoshyd/spanet4Top/ntuples/four_top_SPANET_input/four_top_SPANET_input', type=str, help='Output location')
    parser.add_argument('-m', '--maxjets', default=18, type=int, help='Max number of jets')
    parser.add_argument('-tr', '--treename', default='nominal', type=str, help='Name of nominal tree')
    parser.add_argument('-b', '--btag', type=str, default='DL1r', help='Which btag alg to use')
    parser.add_argument('--oddeven', action='store_false' , help='Split into odd and even events')
    parser.add_argument('--test', action='store_true', help='Test the code')
    parser.add_argument('--ignoreDecay', action='store_true', help='Ignore the decay type of tops')
    parser.add_argument('-p','--prefix',default='year17_', choices=['year15+16_','year17_','year18_'],help='Prefix to separate files')
    parser.add_argument('-s','--suffix',default='_tttt', choices=['_tttt',], help='Suffix to separate files')
    args = parser.parse_args()

    if args.prefix != '':
        listNameParts = args.outloc.split('/')
        listNameParts[-1] = args.prefix + listNameParts[-1]
        print('Old output location:', args.outloc)
        args.outloc = '/'.join(listNameParts)
        print('New output location:', args.outloc)
    if args.suffix != '':
        listNameParts = args.outloc.split('/')
        listNameParts[-1] = listNameParts[-1] + args.suffix
        print('Old output location:', args.outloc)
        args.outloc = '/'.join(listNameParts)
        print('New output location:', args.outloc)
    # exit(1)
    #print(args.inloc)
    if os.path.isfile(args.inloc):
        file_paths.append(args.inloc)
    elif os.path.isdir(args.inloc):
        for filename in  glob.iglob(args.inloc+"**/**.root", recursive=True):
            #print(filename)
            if os.path.isfile(filename):
                file_paths.append(filename)
    displayFoundFiles(file_paths)
    # print(file_paths)
    """
    file_paths is a list of paths to the ATLAS ROOT files.
    source() function builds the HDF5 file.
    """

    MAX_JETS = args.maxjets
    if not MAX_JETS:
        MAX_JETS = get_max_jets(file_paths, args.treename, args.topo)

    # For now we choose only two files
    # rTagDict = {15 : ['r13167','r14859'], 16 : ['r13167','r14859'], 17 : ['r13144','r14860'], 18 : ['r13145','r14861']}
    selectionSubStrDict = {
        'tttt' : { 
            '15+16' :       [r'user.*.412043.aMcAtNloPythia8EvtGen.DAOD_PHYS.*r14859.*nominal.root'],
            '17' :          [r'user.*.412043.aMcAtNloPythia8EvtGen.DAOD_PHYS.*r14860.*nominal.root'],
            '18' :          [r'user.*.412043.aMcAtNloPythia8EvtGen.DAOD_PHYS.*r14861.*nominal.root'],
        },
        'tt_allhad' : {
            '15+16' : [],
            '17' : [],
            '18' : [],
        },
        'tt_lep' : {
            '15+16' : [],
            '17' : [],
            '18' : [],
        }, 
    }

    selectedPaths = []
    for pattern in selectionSubStrDict[args.topo][args.prefix.replace('year','').replace('_','')]:
        for i in range(len(file_paths)):
            filePath = os.path.basename(file_paths[i])
            match = re.search(pattern, filePath) 
            if match:
                print(pattern)
                selectedPaths.append(file_paths[i])
    file_paths = selectedPaths
    
    print(file_paths)
    displayFoundFiles(file_paths)
    # Run production:
    source(file_paths, args)



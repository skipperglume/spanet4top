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
import pickle
import datetime
from sharedMethods import *

from rich import pretty
pretty.install()

# TODO: implement Py Parsing
# import pyparsing

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

def goodEvent( tree : ROOT.TTree, args : argparse.Namespace) -> bool:
    result = True

    # Selection on number of jets
    if not tree.n_jet >= 8: 
        result = False
    # Selection on number of b-tagged jets
    if not tree.n_bjet >= 1:
        result = False
    # Check that event indeed has 4 top quarks (happens that some events have 3)
    if len(list(tree.truthTop_isLepDecay)) != 4 : 
        result = False
    # Check that the event is all hadronic:
    isNotAllHad = 1 if sum([float(x.encode("utf-8").hex()) for x in list(tree.truthTop_isLepDecay)]) > 0 else 0
    if isNotAllHad: 
        result = False
    # print('Is Not All Hadronic:', isNotAllHad)
    
    return result
def goodAssignment(tree, assignmentDict : dict, args : argparse.Namespace ) -> bool:
    # Check that jet indices are above value of maximum number of jets
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
    # Check uniqueness of the assignments. Same jet should not correspond to different partons
    if len(uniqueSet) != len(notUnique):
        return False

    # args.reconstruction

    groupedNames = getParticlesGroupName(args.reconstruction)
    for particle in groupedNames:
        # Check that each particle has at least one jet assigned
        if all([ assignmentDict[particle +'/'+target] == -1 for target in groupedNames[particle]]):
            return False
        # if sum([assignmentDict[particle +'/'+target] == -1 for target in groupedNames[particle]]) > 1:
            # return False
    return True

def getParticlesGroupName(reconstructionTargets) -> dict :
    '''
    This method returns a dictionary of particle groups and their corresponding partons.
    E.g 't1(b,q1,q2)|t2(b,q1,q2)' -> {'t1': ['b', 'q1', 'q2'], 't2': ['b', 'q1', 'q2']}
    '''
    groupedNames = {}
    for group in reconstructionTargets.split('|'):
        subgroups = group.split('(')
        groupedNames[subgroups[0]] = subgroups[1][:-1].split(',')                
    return groupedNames

def source(root_files : list, args: argparse.Namespace):
        """
        Create HDF5 file and create the "source" group in the HDF5 file.
        """
        # Creating HDF5 file, setting MAX_JETS to 0, and opening all provided ROOT files
       
        #files = [ROOT.TFile(rf) for rf in root_files]

        # List of feature name to save:
        featuresToSave = [
            'INPUTS/Source/MASK',
            # 'INPUTS/Source/pt_x',
            # 'INPUTS/Source/pt_y',
            'INPUTS/Source/pt',
            'INPUTS/Source/eta',
            'INPUTS/Source/e',
            'INPUTS/Source/sin_phi',
            'INPUTS/Source/cos_phi',
            'INPUTS/Source/btag',
            'INPUTS/Met/met',
            # 'INPUTS/Met/met_x',
            # 'INPUTS/Met/met_y',
            'INPUTS/Met/sin_phi',
            'INPUTS/Met/cos_phi',
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

        # These variables are needed to check the selection criteria
        countDict = {}
        countDict['passJetNumber'] = []
        countDict['passDecayType'] = []
        countDict['passAssignemntCheck'] = []
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
       
        # Set histograms
        histoDict = {}
        histoDict['MatchedMass'] = ROOT.TH1F('MatchedMass', 'MatchedMass', 100, 0, 500)
        histoDict['TruthMass'] = ROOT.TH1F('TruthMass', 'TruthMass', 100, 0, 500)
        histoDict['DeltaR'] = ROOT.TH1F('DeltaR', 'DeltaR', 100, -1, 4)
        histoDict['PtRatio'] = ROOT.TH1F('PtRatio', 'PtRatio', 100, -1, 4)
        histoDict['MatchedJetsMass'] = ROOT.TH1F('MatchedJetsMass', 'MatchedJetsMass', 100, 0, 500)

        histoDict['topTruthRecoDeltaR'] = ROOT.TH1F('topTruthRecoDeltaR', 'topTruthRecoDeltaR', 100, -1, 4)
        histoDict['topTruthRecoPtRatio'] = ROOT.TH1F('topTruthRecoPtRatio', 'topTruthRecoPtRatio', 100, -1, 4)
        histoDict['partonRecoDeltaR'] = ROOT.TH1F('partonRecoDeltaR', 'partonRecoDeltaR', 100, -1, 4)
        histoDict['partonRecoPtRatio'] = ROOT.TH1F('partonRecoPtRatio', 'partonRecoPtRatio', 100, -1, 4)

        for histo in histoDict:
            histoDict[histo].Sumw2()

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
                     
            eventNumber = nominal.GetEntries()

            # Counters to display need info about our samples
            all_had_count = 0
            non_all_had_count = 0
            eventRange = range(0, eventNumber)
            # if args.test: eventRange = range(3550000+ 28000, eventNumber)
            for i in tqdm(eventRange):
                # Early stopping for testing
                if args.test and i-eventRange[0] > 5000 : break
                #print(i)
                if i % 50000 == 0: 
                    print(str(i)+"/"+str(eventNumber))
                    print('Currently collected events: ',len(inputDict['AUX/aux/eventNumber']), '-', len(inputDict['AUX/aux/eventNumber'])/(i+1))
                nominal.GetEntry(i)
                # Now do the particle groups, ie the truth targets
                # One could apply cuts here if desired, but usually inclusive training is best!
                # print(assignmentDict)
                if not goodEvent(nominal, args): continue
                
                # Old parton jets assignment. Done in RAC. Has problems: no protection against repetitions
                assignmentDict = assignIndicesljetsttbar(nominal, args)
                
                # New parton jets assignment. Done in RAC. Has problems: no protection against repetitions
                assignment, quality = findPartonJetPairs(nominal, args)

                # Here we set which assignment to use
                assignmentDict = assignment
                # Quality keys:
                # 'nAssigned', 'unique', 'nCompleteMatch', 
                # 't@_TruthMass', 't@_MatchMass', 't@_DeltaR', 't@_PtRatio'
                
                # if quality['nCompleteMatch'] != 4 : continue
                
                if not True:
                    print('Old | New Assignments:')
                    for group in getParticlesGroupName(args.reconstruction):
                        for parton in getParticlesGroupName(args.reconstruction)[group]:
                            oldMatch = assignmentDict[group+'/'+parton] if group+'/'+parton in assignmentDict else 'N'
                            newMatch = assignment[group+'/'+parton] if group+'/'+parton in assignment else 'N'
                            print(group+'/'+parton+":", oldMatch, '|', newMatch)
                
                if not goodAssignment(nominal, assignmentDict, args): 
                    # print(assignmentDict)
                    continue
                
                for group in getParticlesGroupName(args.reconstruction):

                    histoDict['MatchedMass'].Fill( quality[f'{group}_MatchMass']  , nominal.weight_final)
                    histoDict['TruthMass'].Fill( quality[f'{group}_TruthMass']  , nominal.weight_final)
                    histoDict['DeltaR'].Fill( quality[f'{group}_DeltaR']  , nominal.weight_final)
                    histoDict['PtRatio'].Fill( quality[f'{group}_PtRatio']  , nominal.weight_final)
                    histoDict['MatchedJetsMass'].Fill( quality[f'{group}_MatchedJetsMass']  , nominal.weight_final)
                    histoDict['topTruthRecoDeltaR'].Fill( quality[f'{group}_truthRecoDeltaR']  , nominal.weight_final)
                    histoDict['topTruthRecoPtRatio'].Fill( quality[f'{group}_truthRecoPtRatio']  , nominal.weight_final)
                    histoDict['partonRecoDeltaR'].Fill( quality[f'{group}_partonRecoDeltaR']  , nominal.weight_final)
                    histoDict['partonRecoPtRatio'].Fill( quality[f'{group}_partonRecoPtRatio']  , nominal.weight_final)

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
        collectedEvents = len(inputDict['AUX/aux/eventNumber'])
        print('Collected events: ', collectedEvents)
        print('Number of All hadronic:', all_had_count, all_had_count/collectedEvents )
        print('Number of Non All hadronic:', non_all_had_count, non_all_had_count/collectedEvents)
        
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
            write( out, args.topo, featuresToSave, indices, inputDict, {} )
        plotHistograms( histoDict, args)


def write(outloc : str, topo : str, featuresToSave : list, indices : np.array, inputDict, decayDict : dict):
    """
        Function to create and write to the HDF5 file
    """
    # print('Indices Size: ',len(indices[0]))
       
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

def plotHistograms( histoDict: dict, args : argparse.Namespace) -> None:
    '''
    Save the histograms as pickle dump file and also as png files.
    '''
    # Get Date and Time in format: dd-mm-yyyy
    dateTimeVal = datetime.datetime.now().strftime("%d-%m-%Y")
    if args.test : dateTimeVal += '_test'
    pickle.dump(histoDict, open(f'pickles/histograms_{args.prefix}_{args.topo}_truth_{dateTimeVal}.pkl', 'wb'))
    for histo in histoDict:
        c = ROOT.TCanvas()
        histoDict[histo].Draw()
        c.SaveAs(f'plots/{histo}.png')

def targetsAreUnique(targets : list) -> bool:
    for target in targets: 
        if target != -1 and targets.count(target) > 1:
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

def massFromLVs(LVs : list, scale=1.0 ) -> float:
    if len(LVs) == 0:
        print('Size of LVs should not be zero')
        exit(1)
    sumLV = ROOT.TLorentzVector( LVs[0] )
    for lvIndex in range(1, len(LVs)):
        sumLV += LVs[lvIndex]
    mass = sumLV.M()
    return mass * scale

def jetSetMass(nominal , jetSet : list, args : argparse.Namespace) -> float:
    '''
    This function evaluates the invariant mass of a set of jets. Returns the mass of the set of jets. 
    As well does checks that the set of jets is reasonable. 
    '''
    mass = 0.0
    if len(jetSet) == 0:
        print('Size of jets set should not be zero')
        exit(1)
    
    LVs = []
    for jetIndex in jetSet:
        # count number of repeated jetIndex in jetSet:
        if jetIndex != -1 and jetSet.count(jetIndex) > 1:
            print(f'Repeated jetIndex {jetIndex} in jetSet {jetSet}')
            exit(1)
        lv = ROOT.TLorentzVector()
        lv.SetPtEtaPhiE(nominal.jet_pt[jetIndex], nominal.jet_eta[jetIndex], nominal.jet_phi[jetIndex], nominal.jet_e[jetIndex])
        LVs.append(lv)
    mass = massFromLVs(LVs)


    return mass

def getTrue_N_DetectedMasses(partonLVs : dict, jetLVs : dict, assignemnt : dict, args : argparse.Namespace) -> dict:
    '''
    This function returns the masses of the parton level and the detected level.
    '''
    result = {}
    for group in args.reconstruction.split('|'):
        subgroups = group.split('(')
        particleName = subgroups[0]
        partons = subgroups[1][:-1].split(',')
        partonSet = []
        jetSet = []
        jetDict = {}
        particleMass = 0
        detectedMass = 0
        for parton in partons:
            if f'{particleName}/{parton}' in partonLVs:
                partonSet.append(partonLVs[f'{particleName}/{parton}'])
            
            if f'{particleName}/{parton}' in assignemnt and assignemnt[f'{particleName}/{parton}'] != -1:
                jetSet.append(  jetLVs[assignemnt[f'{particleName}/{parton}']] )
                jetDict[parton] = assignemnt[f'{particleName}/{parton}']
        
        particleMass = massFromLVs(partonSet, 0.001)
        
        # print('Jet Lorentz Vectors:')
        # for lv in jetSet:
        #     print( lv.Pt()*0.001, lv.Eta(), lv.Phi(), lv.E()*0.001)

        detectedMass = massFromLVs(jetSet, 0.001) if len(jetSet)>0 else -1.0
        result[ particleName ] = {'particle': particleMass, 'detected': detectedMass, 'jets': jetDict}
    return result

def printMassInfo(partonLVs : dict, jetLVs : dict, assignemnt : dict, args : argparse.Namespace) -> None:
    '''
    This function prints the mass information of parton level and matched one.
    '''
    compressedDict = getTrue_N_DetectedMasses(partonLVs, jetLVs, assignemnt, args)
    for particle in compressedDict:
        particleMass = compressedDict[particle]['particle']
        detectedMass = compressedDict[particle]['detected']
        jetSet = compressedDict[particle]['jets']
        print(f'{particle} : True Mass: {particleMass}; Detected Mass {detectedMass} with {jetSet} jets matched.')
    return

def printPartonJetMatches(costMatrix : dict, partonToJets : dict, jetToPartons : dict):
    print(f'Cost Matrix: {len(costMatrix)}:-')
    print(costMatrix)
    print(f'Parton-Jet pairs: {len(partonToJets)}')
    for partonLabel in partonToJets:
        print(partonLabel, ":", partonToJets[partonLabel], '-', len(partonToJets[partonLabel]))
    print(f'Jet-Parton pairs: {len(jetToPartons)}')
    for jetLabel in jetToPartons:
        print(jetLabel, ":", jetToPartons[jetLabel], '-', len(jetToPartons[jetLabel]))

def getPartonLVs(nominal, args : argparse.Namespace) -> dict:
    partonLVs = {}    
    groupedNames = getParticlesGroupName(args.reconstruction)
    
    # print(f'Total partons to match:', sum( [len(groupedNames[_]) for _ in groupedNames]) )
        
    for particle in groupedNames.keys():
        topIter = int(particle[1:])-1
        
        partonLVs[f'{particle}/type'] = int(nominal.truthTop_isLepDecay[topIter].encode("utf-8").hex())
        
        for parton in groupedNames[particle]:
            partonLVs[f'{particle}/{parton}'] = ROOT.TLorentzVector()
            if parton == 'b':
                partonLVs[f'{particle}/{parton}'].SetPtEtaPhiE(nominal.truthTop_b_pt[topIter]  , nominal.truthTop_b_eta[topIter], nominal.truthTop_b_phi[topIter], nominal.truthTop_b_e[topIter])
            elif parton == 'q1':
                partonLVs[f'{particle}/{parton}'].SetPtEtaPhiE(nominal.truthTop_W_child1_pt[topIter]  , nominal.truthTop_W_child1_eta[topIter], nominal.truthTop_W_child1_phi[topIter], nominal.truthTop_W_child1_e[topIter])
            elif parton == 'q2':
                partonLVs[f'{particle}/{parton}'].SetPtEtaPhiE(nominal.truthTop_W_child2_pt[topIter]  , nominal.truthTop_W_child2_eta[topIter], nominal.truthTop_W_child2_phi[topIter], nominal.truthTop_W_child2_e[topIter])
    
    return partonLVs

def getJetLVs(nominal, args : argparse.Namespace) -> dict:
    jetLVs = {}
    for jetIter in range(len(nominal.jet_pt)):
        jetLVs[jetIter] = ROOT.TLorentzVector()
        jetLVs[jetIter].SetPtEtaPhiE(nominal.jet_pt[jetIter], nominal.jet_eta[jetIter], nominal.jet_phi[jetIter], nominal.jet_e[jetIter])
    return jetLVs

def getCostMatrix(nominal, partonLVs : dict, jetLVs : dict, args : argparse.Namespace) -> dict:
    ''' 
    Create a cost matrix for each parton-jet pair with penalties and reewards
    e.g. if a jet is a b-jet and the parton is a b-jet ...
    '''
    costMatrix = {}
    for jetLabel in jetLVs:
        for partonLabel in partonLVs:
            if 'type' in partonLabel: continue
            if partonLabel not in costMatrix:
                costMatrix[partonLabel] = {}
            if partonLVs[partonLabel].DeltaR(jetLVs[jetLabel]) < getRadius(partonLVs[partonLabel].Eta())  :
                costMatrix[partonLabel][jetLabel] = partonLVs[partonLabel].DeltaR(jetLVs[jetLabel])
                if nominal.jet_tagWeightBin_DL1dv01_Continuous[jetLabel] >= 3 and '/b' in partonLabel:
                    costMatrix[partonLabel][jetLabel] += (-1)
                if nominal.jet_tagWeightBin_DL1dv01_Continuous[jetLabel] >= 4 and '/b' not in partonLabel:
                    costMatrix[partonLabel][jetLabel] += (1)
    return costMatrix

def getPairWiseDicts(costMatrix : dict) -> list:
    '''
    Create dictionaries to store the parton-jet and jet-parton pairs
    '''
    partonToJets = {}
    jetToPartons = {}
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
    return (partonToJets, jetToPartons)

def getParticleLVs(nominal, args : argparse.Namespace) -> dict:
    '''
    Returns the principal particles Lorentz Vectors (t1,t2,t3,t4) in form of:
    {'t1' : ROOT.TLorentzVector,..}
    '''
    particleLVs = {}
    for particle in getParticlesGroupName(args.reconstruction):
        particleLVs[particle] = ROOT.TLorentzVector()
        particleIter = int(particle[1:])-1
        particleLVs[particle].SetPtEtaPhiE(nominal.truthTop_pt[particleIter], nominal.truthTop_eta[particleIter], nominal.truthTop_phi[particleIter], nominal.truthTop_e[particleIter])
    return particleLVs
def getParticleMatchedLVs( partonLVs : dict, assignmentDict : dict, args : argparse.Namespace) -> dict:
    '''
    Returns the Lorentz vectors of the matched partons for each principal particles in form of:
    {'t1' : ROOT.TLorentzVector,..}. If no parton is matched, the Lorentz vector is set to 0.
    '''
    result = {}
    particleGroups = getParticlesGroupName(args.reconstruction)
    for particle in particleGroups:
        result[particle] = ROOT.TLorentzVector()
        result[particle].SetPxPyPzE(0,0,0,0) # Empty vector
        for parton in particleGroups[particle]:
            partonName = particle+'/'+parton
            if partonName in assignmentDict and assignmentDict[partonName] != -1:
                result[particle] += partonLVs[partonName]
    return result
        
def getLVofMatchedJets(jetLVs : dict, assignmentDict : dict, args : argparse.Namespace) -> dict:
    '''
    Calculate the Lorentz vectors of the matched jets for each principal particles
    '''
    result = {}
    particlePartons = getParticlesGroupName(args.reconstruction)
    for particle in particlePartons:
        result[particle] = ROOT.TLorentzVector()
        result[particle].SetPxPyPzE(0,0,0,0)
        for parton in particlePartons[particle]:
            partonName = particle+'/'+parton
            if partonName in assignmentDict and assignmentDict[partonName] != -1:
                result[particle] += jetLVs[assignmentDict[partonName]]
    return result

def evaluateAssignmentQuality(nominal, partonLVs : dict, jetLVs : dict, assignmentDict : dict, args : argparse.Namespace) -> dict:
    wellness = {}
    # Number of jets that were matched to partons
    numberAssigned = 0
    for partonLabel in assignmentDict:
        if assignmentDict[partonLabel] != -1:
            numberAssigned += 1
    wellness['nAssigned'] = numberAssigned
    
    # Flag that all jets are unique
    wellness['unique'] = targetsAreUnique([assignmentDict[partonLabel] for partonLabel in assignmentDict])

    # i.e. particles for which all partons are matched to jets
    numberComplete = 0
    # getParticlesGroupName(args.reconstruction)
    
    # Calculating the number of completly matched particles (tops)
    particlePartons = getParticlesGroupName(args.reconstruction)
    for particle in particlePartons:
        partons = particlePartons[particle]
        if all([assignmentDict[particle+'/'+parton] != -1 for parton in partons]):
            numberComplete += 1
    wellness['nCompleteMatch'] = numberComplete

    # Calculating the masses of the partons and the matched jets
    # Delta R and ratio of transverse momentums

    # Dictionary of Lontze Vectors for:
    # - Target Particles (4top)
    particleLVs = getParticleLVs(nominal, args)
    # - Target Partons that have a matching jet
    particleMatchedLVs = getParticleMatchedLVs( partonLVs, assignmentDict, args)
    # - Matched Jet to partons
    lvOfMatchedJets = getLVofMatchedJets(jetLVs, assignmentDict, args)
    
    for particle in particleLVs:
        wellness[f'{particle}_TruthMass'] = particleLVs[particle].M()*0.001
        wellness[f'{particle}_MatchMass'] = particleMatchedLVs[particle].M()*0.001
        wellness[f'{particle}_MatchedJetsMass'] = lvOfMatchedJets[particle].M()*0.001
        
        wellness[f'{particle}_DeltaR'] = particleLVs[particle].DeltaR(particleMatchedLVs[particle])
        wellness[f'{particle}_PtRatio'] = particleMatchedLVs[particle].Pt()/particleLVs[particle].Pt()

        wellness[f'{particle}_truthRecoDeltaR'] = particleLVs[particle].DeltaR(lvOfMatchedJets[particle])
        wellness[f'{particle}_truthRecoPtRatio'] = lvOfMatchedJets[particle].Pt()/particleLVs[particle].Pt()

        wellness[f'{particle}_partonRecoDeltaR'] = particleMatchedLVs[particle].DeltaR(lvOfMatchedJets[particle]) if particleMatchedLVs[particle].Pt() > 0 else -0.5
        # particleMatchedLVs[particle].DeltaR(lvOfMatchedJets[particle]) if particle in lvOfMatchedJets else -1
        wellness[f'{particle}_partonRecoPtRatio'] = lvOfMatchedJets[particle].Pt() / particleMatchedLVs[particle].Pt() if particleMatchedLVs[particle].Pt() > 0 else -0.5
        # lvOfMatchedJets[particle].Pt()/particleMatchedLVs[particle].Pt() if particleMatchedLVs[particle].Pt() > 0 else -1

        # wellness[f'{particle}_topTruthRecoDeltaR'] = lvOfMatchedJets[particle].M()*0.001 if particle in lvOfMatchedJets else -1
    

    # for particle in particlePartons:
    #     print(particle, ':', end='')
    #     print( getParticleMatchedLVs( partonLVs, assignmentDict, args)[particle].M()*0.001, ' / ', end=''  )
    #     print( particleLVs[particle].M()*0.001, ' / ', end=''  )
    #     print( lvOfMatchedJets[particle].M()*0.001, ' / ', end=''  )
    #     if particle in particleMatchedLVs:
    #         print( particle, ':', particleMatchedLVs[particle].M()*0.001, ' / ', end=''   )
    #         if getParticleMatchedLVs( partonLVs, assignmentDict, args)[particle].M()*0.001 != particleMatchedLVs[particle].M()*0.001:
    #             print('!+!+!+!+')

    #     print()
    return wellness

def findPartonJetPairs(nominal, args) -> tuple:
    '''
    Method to find pairings between parton with detected jets. 
    The assignment target arrays contain the indices of each assignment. 
    Only sequential input indices may be assignment targets. 
    Each reconstruction target should be associated with exactly one input vector. 
    These indices must also be strictly unique. 
    Any targets which are missing within an event should be marked with -1.
    '''
    # ['truthTop_W_pt', 'truthTop_W_eta', 'truthTop_W_phi', 'truthTop_W_e']
    # 'truthTop_W_child1_pdgId'
    # 'truthTop_W_child2_pdgId'
    # ['truthTop_truthJet_pt', 'truthTop_truthJet_eta', 'truthTop_truthJet_phi', 'truthTop_truthJet_e']
    # 'truthTop_truthJet_index' 'truthTop_truthJet_flavor',  

    # This final matching between partons and jets in form of { parton : jet }
    result = {}
    # Parton and jet Lorentz Vectors Dictionary {Name : LV}
    partonLVs = getPartonLVs(nominal, args)
    jetLVs = getJetLVs(nominal, args)

    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++')

    # Get the cost matrix for each parton-jet pair
    costMatrix = getCostMatrix(nominal, partonLVs, jetLVs, args)

    partonToJets , jetToPartons = getPairWiseDicts(costMatrix)

    # Sort entries for each parton label by the cost
    for partonLabel in partonToJets:
        if len(partonToJets[partonLabel]) > 1 :
            partonToJets[partonLabel] = sorted(partonToJets[partonLabel], key=lambda x: costMatrix[partonLabel][x])
            # print('Sorted \'ere')

    # Sort entries for each jet by the cost
    for jetLabel in jetToPartons:
        if len(jetToPartons[jetLabel]) > 1:
            jetToPartons[jetLabel] = sorted(jetToPartons[jetLabel], key=lambda x: costMatrix[x][jetLabel])
            # print('Sorted \'ere')
    
    # Printing of the cost matrix and the parton-jet and jet-parton matches
    if not True:
        printPartonJetMatches(costMatrix, partonToJets, jetToPartons)
    
    # List which keeps track of which jets have been used to prevent from using same jets
    usedJets = []
    # Filling the result dictionary with the definitive matches (1 jet - 1 parton correspondence as weel as missing jets)
    for partonLabel in costMatrix:
        # If no jets are available for a parton, assign -1
        if len(costMatrix[partonLabel])==0:
            result[partonLabel] = -1
            continue
        
        # If only one jet is available for a parton and it has not been used yet and the jet has only one parton, assign it
        if len(partonToJets[partonLabel]) == 1 and partonToJets[partonLabel][0] not in usedJets and len(jetToPartons[partonToJets[partonLabel][0]]) == 1:
            result[partonLabel] = partonToJets[partonLabel][0]
            usedJets.append(result[partonLabel])

    for partonLabel in costMatrix:
        # Assign the first jet that has not been used yet to the parton
        if partonLabel in partonToJets and len(partonToJets[partonLabel]) > 0:
            for jetLabel in partonToJets[partonLabel]:
                if jetLabel not in usedJets:
                    result[partonLabel] = jetLabel
                    usedJets.append(jetLabel)
                    break
            
        # If no jets are left for a parton, assign -1
        if partonLabel not in result:
            result[partonLabel] = -1
        
    # Use this information for later quality control 
    wellness = evaluateAssignmentQuality(nominal, partonLVs, jetLVs, result, args)
    
    if not True:
        print('Wellness:', wellness)
        printMassInfo(partonLVs, jetLVs, result, args)

    return (result, wellness)

def assignIndicesljetsttbar( nominal, args : argparse.Namespace) -> dict:
    """
    This funciton returns the indices of the partons and jets in the SPA-Net format
    """
    groupedNames = getParticlesGroupName(args.reconstruction)
    result = {}
    for group in groupedNames:
        for parton in groupedNames[group]:
            result[group+'/'+parton] = -1
    
    
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
    
    return result

if __name__ == "__main__":
       
    file_paths = []

    parser=argparse.ArgumentParser()
    parser.add_argument('-i', '--inloc', default='/home/timoshyd/RAC_4tops_analysis/ntuples/v06_BDT_SPANET_Input/nom', type=str, help='Input file location.')
    parser.add_argument('-r', '--reconstruction', default='t1(b,q1,q2)|t2(b,q1,q2)|t3(b,q1,q2)|t4(b,q1,q2)', type=str, help='Topology of underlying event.')
    parser.add_argument('-t', '--topo', default='tttt', type=str, help='Topology of underlying event.')
    parser.add_argument('-o', '--outloc', default='/home/timoshyd/spanet4Top/ntuples/four_top_SPANET_input/four_top_SPANET_input', type=str, help='Output location. File type of .h5 will be added automatically.')
    parser.add_argument('-m', '--maxjets', default=18, type=int, help='Max number of jets.')
    parser.add_argument('-tr', '--treename', default='nominal', type=str, help='Name of nominal tree.')
    parser.add_argument('-b', '--btag', type=str, default='DL1r', help='Which btag alg to use.')
    parser.add_argument('--oddeven', action='store_false' , help='Split into odd and even events.')
    parser.add_argument('--test', action='store_true', help='Test the code.')
    parser.add_argument('--ignoreDecay', action='store_true', help='Ignore the decay type of tops.')
    parser.add_argument('-p','--prefix',default='year18_', choices=['year15+16_','year17_','year18_'],help='Prefix to separate files Via looking which exacylt to use.')
    parser.add_argument('-s','--suffix',default='_tttt', choices=['_tttt',], help='Suffix to separate files.')
    parser.add_argument('-c','--cuts',default='allHad==1;jets>=8;bjets>=1;', choices=['_tttt',], help='Suffix to separate files.')
    parser.add_argument('--tag',default='_8JETS', help='Suffix to differentiate files.')

    args = parser.parse_args()
    args.particlesGroupName = getParticlesGroupName(args.reconstruction)

    if args.prefix != '':
        listNameParts = args.outloc.split('/')
        listNameParts[-1] = args.prefix + listNameParts[-1]
        print('Old output location:', args.outloc)
        args.outloc = '/'.join(listNameParts)
        print('New output location:', args.outloc)
    if args.suffix != '' or args.tag != '':
        listNameParts = args.outloc.split('/')
        listNameParts[-1] = listNameParts[-1] + args.suffix + args.tag
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
    print(args.particlesGroupName)


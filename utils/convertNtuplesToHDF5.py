#!/usr/bin/env python3
import argparse
import ROOT
import math
import h5py
import numpy as np
import sys, os
import glob

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

def nuReconstruction(nominal):
       ''' 
       reconstruct neutrino candidates using W mass assumption
       see https://arxiv.org/pdf/1502.05923.pdf
       '''

       neutrinos = []
       
       lepton = ROOT.TLorentzVector()
       if len(nominal.el_pt) > 0:
              #print("electron")
              lepton.SetPtEtaPhiE(nominal.el_pt[0], nominal.el_eta[0], nominal.el_phi[0], nominal.el_e[0])
       elif (len(nominal.mu_pt) >0):
              #print("muon")
              lepton.SetPtEtaPhiE(nominal.mu_pt[0], nominal.mu_eta[0], nominal.mu_phi[0], nominal.mu_e[0])

       met_met = nominal.met_met
       met_phi = nominal.met_phi


       met_x = met_met * math.cos(met_phi)
       met_y = met_met * math.sin(met_phi)
       
       k = ((80400*80400 - lepton.M()*lepton.M()) / 2) + (lepton.Px()*met_x + lepton.Py()*met_y)
       a = lepton.E()*lepton.E() - lepton.Pz()*lepton.Pz()
       b = -2 * k * lepton.Pz()
       c = lepton.E()*lepton.E()*met_met*met_met - k*k
       disc = b*b - 4*a*c

       if disc < 0:
              nu_pz = -b/(2*a)
              nu = ROOT.TLorentzVector()
              nu.SetPxPyPzE(met_x, met_y, nu_pz, np.sqrt(met_met*met_met+nu_pz*nu_pz))
              neutrinos.append(nu)
       else:
              nu_pz_1 = (-b + np.sqrt(disc)) / (2 * a)
              nu_pz_2 = (-b - np.sqrt(disc)) / (2 * a)
              nu1, nu2 = ROOT.TLorentzVector(), ROOT.TLorentzVector()
              nu1.SetPxPyPzE(met_x, met_y, nu_pz_1, np.sqrt(met_met*met_met+nu_pz_1*nu_pz_1))
              nu2.SetPxPyPzE(met_x, met_y, nu_pz_2, np.sqrt(met_met*met_met+nu_pz_2*nu_pz_2))

              if abs(nu_pz_1) > abs(nu_pz_2):
                     neutrinos.append(nu2)
              else:
                     neutrinos.append(nu1)
              
              #neutrinos.append(nu1)
              #neutrinos.append(nu2)

       #if len(neutrinos) > 1: print("Found"+str(len(neutrinos)))
       return(neutrinos)

def getTruthNeutrino(nominal):
       """
       Get the four momenta of the truth neutrino in the ljets channel. Used for the REGRESSION prediction.
       """

       isLjetsEvent = False

       #Truth neutrino
       truth_nu = ROOT.TLorentzVector()

       #Make sure that it is a ljets event and not dileptonic
       #https://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf
       if(abs(nominal.truth_MC_Wdecay1_from_tbar_pdgId) > 6) and (abs(nominal.truth_MC_Wdecay1_from_t_pdgId) > 6):

           isLjetsEvent=False

       else:

           isLjetsEvent=True

           #Need to check first which W decayed into a lepton and a neutrino
           if(abs(nominal.truth_MC_Wdecay1_from_tbar_pdgId) > 6):
               #Now we know that the tbar decays leptonically 

               if(abs(nominal.truth_MC_Wdecay1_from_tbar_pdgId) == 12) or (abs(nominal.truth_MC_Wdecay1_from_tbar_pdgId) == 14) or (abs(nominal.truth_MC_Wdecay1_from_tbar_pdgId) == 16):
                   truth_nu.SetPtEtaPhiM(nominal.truth_MC_Wdecay1_from_tbar_pt, nominal.truth_MC_Wdecay1_from_tbar_eta, nominal.truth_MC_Wdecay1_from_tbar_phi, nominal.truth_MC_Wdecay1_from_tbar_m)

               elif(abs(nominal.truth_MC_Wdecay2_from_tbar_pdgId) == 12) or (abs(nominal.truth_MC_Wdecay2_from_tbar_pdgId) == 14) or (abs(nominal.truth_MC_Wdecay2_from_tbar_pdgId) == 16):

                   truth_nu.SetPtEtaPhiM(nominal.truth_MC_Wdecay2_from_tbar_pt, nominal.truth_MC_Wdecay2_from_tbar_eta, nominal.truth_MC_Wdecay2_from_tbar_phi, nominal.truth_MC_Wdecay2_from_tbar_m)

           elif(abs(nominal.truth_MC_Wdecay1_from_t_pdgId) > 6):
               #the top decays leptonically
               if(abs(nominal.truth_MC_Wdecay1_from_t_pdgId) == 12) or (abs(nominal.truth_MC_Wdecay1_from_t_pdgId) == 14) or (abs(nominal.truth_MC_Wdecay1_from_t_pdgId) == 16):

                   truth_nu.SetPtEtaPhiM(nominal.truth_MC_Wdecay1_from_t_pt, nominal.truth_MC_Wdecay1_from_t_eta, nominal.truth_MC_Wdecay1_from_t_phi, nominal.truth_MC_Wdecay1_from_t_m)


               elif(abs(nominal.truth_MC_Wdecay2_from_t_pdgId) == 12) or (abs(nominal.truth_MC_Wdecay2_from_t_pdgId) == 14) or (abs(nominal.truth_MC_Wdecay2_from_t_pdgId) == 16):

                   truth_nu.SetPtEtaPhiM(nominal.truth_MC_Wdecay2_from_t_pt, nominal.truth_MC_Wdecay2_from_t_eta, nominal.truth_MC_Wdecay2_from_t_phi, nominal.truth_MC_Wdecay2_from_t_m)


#       print("Truth neutrino")
#       print("Exists = " + str(isLjetsEvent))
#       print(truth_nu.Pt())
#       print(truth_nu.Eta())
#       print(truth_nu.Phi())
#       print(truth_nu.M())
#       if(isLjetsEvent==False):
#           print(nominal.)


       #Get the neutrino four momenta
       return (isLjetsEvent, truth_nu)
def number_of_jets( topTargets : list ) -> int:
       result = 0 
       for i in range(len(topTargets)):
              if topTargets[i] != -1:
                     result += 1
       return result
def source(root_files : list, args: argparse.Namespace):
       """
       Create HDF5 file, find MAX_JETS for the data set,
       and create the "source" group in the HDF5 file.
       """
       # Creating HDF5 file, setting MAX_JETS to 0, and opening all provided ROOT files
       
       #files = [ROOT.TFile(rf) for rf in root_files]

       #INPUTS for SPANet       
       #Jets
       # Features lists that will be appended to and will have dimensions (EVENTS, MAX_JETS)
       jet_mask_list, jet_mass_list, jet_pt_list, jet_eta_list, jet_phi_list, jet_sin_phi_list, jet_cos_phi_list, jet_btag_list, jet_e_list = [], [], [], [], [], [], [], [], []
       jet_btag60_list, jet_btag70_list, jet_btag77_list, jet_btag85_list = [], [], [], []
       jet_ptx_list, jet_pty_list =[], []
       leptag_list = []    # If topo is ttZ > 6jet2lep, create leptag feature list

       #Leptons
       lepton_mask_list, lepton_mass_list, lepton_pt_list, lepton_eta_list, lepton_phi_list, lepton_sin_phi_list, lepton_cos_phi_list, lepton_e_list = [], [], [], [], [], [], [], []
       lepton_etag_list, lepton_mutag_list = [], []

       #MET
       #Only one values per event 
       met_met_list, met_phi_list, met_sin_phi_list, met_cos_phi_list = [], [], [], []
       met_x_list, met_y_list = [], []

       if 'allhad_ttH' in args.topo: qgtag_list = []

       eventNumber_list = []
       allHad_list = []

       nfiles=len(root_files)

       # REGRESSIONS
       neutrino_pt_list, neutrino_eta_list, neutrino_phi_list, neutrino_sin_phi_list, neutrino_cos_phi_list, neutrino_e_list = [], [], [], [], [], []
       #neutrino_px_list, neutrino_py_list, neutrino_pz_list, neutrino_eta_list = [], [], [], []
       #log_neutrino_px_list, log_neutrino_py_list, log_neutrino_pz_list, log_neutrino_eta_list = [], [], [], []

       # TARGETS
       # Instatiating jet and mask lists
       t1q1_list, t1q2_list, t1b_list = [], [], []
       t2q1_list, t2q2_list, t2b_list = [], [], []
       t3q1_list, t3q2_list, t3b_list = [], [], []
       t4q1_list, t4q2_list, t4b_list = [], [], []
       extrajet_list, empty_list = [], []
       #SPANet requires all particles to decay to at least two, to introduce a single jet, you add an empty particle
       
       # Iterating over files to extract data, organize it, and add it to HDF5 file

       for ifile, rf in enumerate(root_files):
              print("Processing file: "+rf+" ("+str(ifile+1)+"/"+str(nfiles)+")")
              f = ROOT.TFile(rf)

              nominal = f.Get(args.treename)

              # this code might be useful if the truth info is stored in a different tree
              #truth = f1.Get("truth")
              #truth_events, nominal_events = truth.GetEntries(), nominal.GetEntries()
              #truth.BuildIndex("runNumber", "eventNumber")
                     
              events = nominal.GetEntries()

              all_had_count = 0
              non_all_had_count = 0
              for i in range(events):
                     # Early stopping for testing
                     if i > 10000 and args.test: break
                     #print(i)
                     if i % 50000 == 0: 
                            print(str(i)+"/"+str(events))
                            print('Currently collected events: ',len(eventNumber_list))
                     nominal.GetEntry(i)
                     
                     
                     #one could apply cuts here if desired, but usually inclusive training is best!
                     if nominal.n_jet < 9: continue
                     #if nominal.nHFJets < 4: continue
                     if (nominal.t1_w0_Jeti >= args.maxjets) or (nominal.t1_w1_Jeti >= args.maxjets) or (nominal.t1_b_Jeti >= args.maxjets): continue
                     if (nominal.t2_w0_Jeti >= args.maxjets) or (nominal.t2_w1_Jeti >= args.maxjets) or (nominal.t2_b_Jeti >= args.maxjets): continue
                     if (nominal.t3_w0_Jeti >= args.maxjets) or (nominal.t3_w1_Jeti >= args.maxjets) or (nominal.t3_b_Jeti >= args.maxjets): continue
                     if (nominal.t4_w0_Jeti >= args.maxjets) or (nominal.t4_w1_Jeti >= args.maxjets) or (nominal.t4_b_Jeti >= args.maxjets): continue
                     # now do the particle groups, ie the truth targets
                     (t1q1, t1q2, t1b, t2q1, t2q2, t2b, t3q1, t3q2, t3b, t4q1, t4q2, t4b),uniqness = assignIndicesljetsttbar(nominal)

                     if (number_of_jets([t1q1, t1q2, t1b] ) < 2 ) or (number_of_jets([t2q1, t2q2, t2b] ) < 2 ) or (number_of_jets([t3q1, t3q2, t3b] ) < 2 ) or (number_of_jets([t4q1, t4q2, t4b] ) < 2 ): 
                            # print('Failed:')
                            # print((t1q1, t1q2, t1b, t2q1, t2q2, t2b, t3q1, t3q2, t3b, t4q1, t4q2, t4b))
                            continue
                     # print('Passed: ')
                     # print((t1q1, t1q2, t1b, t2q1, t2q2, t2b, t3q1, t3q2, t3b, t4q1, t4q2, t4b))
                     # if : continue
                     # if : continue
                     # if : continue

                     if ( t1q1==-1 and t1q2==-1 and t1b ==-1): continue
                     if ( t2q1==-1 and t2q2==-1 and t2b ==-1): continue
                     if ( t3q1==-1 and t3q2==-1 and t3b ==-1): continue
                     if ( t4q1==-1 and t4q2==-1 and t4b ==-1): continue
                     
                     # if t1q1 == -1: continue
                     # if t1q2 == -1: continue
                     # if t1b == -1: continue
                     # if t2q1 == -1: continue
                     # if t2q2 == -1: continue
                     # if t2b == -1: continue
                     # if t3q1 == -1: continue
                     # if t3q2 == -1: continue
                     # if t3b == -1: continue
                     # if t4q1 == -1: continue
                     # if t4q2 == -1: continue
                     # if t4b == -1: continue

                     if not uniqness: 
                            # print("T 1 : ", t1q1, t1q2, t1b)
                            # print("T 2 : ", t2q1,t2q2, t2b)
                            # print("T 3 : ", t3q1,t3q2, t3b)
                            # print("T 4 : ", t4q1,t4q2, t4b)
                            continue
                     isLepDecay = [ float(x.encode("utf-8").hex())  for x in list(nominal.truthTop_isLepDecay)]
                     if sum(isLepDecay) > 0:
                            non_all_had_count += 1
                     else:
                            all_had_count += 1

                     # Feature lists for: pt, eat, phi, ls - that hold this info for ONLY CURRENT EVENT
                     jet_pt_ls,jet_eta_ls,jet_phi_ls,jet_e_ls = list(nominal.jet_pt),list(nominal.jet_eta),list(nominal.jet_phi),list(nominal.jet_e)
                     jet_sin_phi_ls,jet_cos_phi_ls = list(map(math.sin,nominal.jet_phi)),list(map(math.cos,nominal.jet_phi))
                     jet_ptx_ls = [  nominal.jet_pt[i] * math.cos(nominal.jet_phi[i]) for i in range(len(nominal.jet_pt)) ]
                     jet_pty_ls = [  nominal.jet_pt[i] * math.sin(nominal.jet_phi[i]) for i in range(len(nominal.jet_pt)) ]
                     
                     lep_ls = []

                     # Adding btag values according to WP
                     # comment out lines here if any WPs are missing
                     if btag_alg == 'MV2c10':
                            jet_b_ls = [float(i) for i in list(nominal.jet_tagWeightBin_MV2c10_Continuous)]
                            jet_b60_ls = [float(i.encode("utf-8").hex()) for i in list(nominal.jet_isbtagged_MV2c10_60)]
                            jet_b70_ls = [float(i.encode("utf-8").hex()) for i in list(nominal.jet_isbtagged_MV2c10_70)]
                            jet_b77_ls = [float(i.encode("utf-8").hex()) for i in list(nominal.jet_isbtagged_MV2c10_77)]
                            jet_b85_ls = [float(i.encode("utf-8").encode("hex")) for i in list(nominal.jet_isbtagged_MV2c10_85)]
                     elif btag_alg == 'DL1r':
                            jet_b_ls = [float(i) for i in list(nominal.jet_tagWeightBin_DL1dv01_Continuous)]
                            pseudo_btag = [float(i) for i in list(nominal.jet_tagWeightBin_DL1dv01_Continuous)]
                     else:
                            print("ERROR: btagging algorithm not recognized")
                            exit(1)
                     # Add inf for leptons
                     # if 'ljets' in args.topo:
                     #        #add the lepton to the output list
                     #        if len(nominal.el_pt) > 0:
                     #               #print(nominal.el_pt[0])
                     #               lepton_pt_ls = list(nominal.el_pt)
                     #               lepton_eta_ls = list(nominal.el_eta)
                     #               lepton_phi_ls = list(nominal.el_phi)
                     #               lepton_sin_phi_ls = list(map(math.sin,nominal.el_phi))
                     #               lepton_cos_phi_ls = list(map(math.cos,nominal.el_phi))
                     #               lepton_e_ls = list(nominal.el_e)
                     #               lepton_etag_ls = []
                     #               lepton_etag_ls.append(1)
                     #               lepton_mutag_ls = []
                     #               lepton_mutag_ls.append(0)
                     #        elif len(nominal.mu_pt) > 0:
                     #               #print(nominal.mu_pt[0])
                     #               lepton_pt_ls  = list(nominal.mu_pt)
                     #               lepton_eta_ls = list(nominal.mu_eta)
                     #               lepton_phi_ls = list(nominal.mu_phi)
                     #               lepton_sin_phi_ls = list(map(math.sin,nominal.mu_phi))
                     #               lepton_cos_phi_ls = list(map(math.cos,nominal.mu_phi))
                     #               lepton_e_ls = list(nominal.mu_e)
                     #               lepton_etag_ls = []
                     #               lepton_etag_ls.append(0)
                     #               lepton_mutag_ls = []
                     #               lepton_mutag_ls.append(1)

                            #reconstruct the neutrino
                            neutrinos = []
#                            neutrinos = nuReconstruction(nominal)
#                           for nu in neutrinos:
#                                  #print(nu.Print())
#                                  pt_ls.append(nu.Pt())
#                                   eta_ls.append(nu.Eta())
#                                   phi_ls.append(nu.Phi())
#                                   sin_phi_ls.append(math.sin(nu.Phi()))
#                                   cos_phi_ls.append(math.cos(nu.Phi()))
#                                   e_ls.append(nu.E())
#                                   b_ls.append(1.0) #nb 1 is untagged for pcb for some reason
#                                   lep_ls.append(2.0)
                                   
                     # Getting mass of each jet(or lep) with eTOm function
                     jet_m_ls = eTOm([jet_pt_ls, jet_eta_ls, jet_phi_ls, jet_e_ls])

                     # Source's "mask" is given True for positions with a jet(or lep) and False when empty
                     jet_mask_list.append([True if i<len(jet_pt_ls) else False for i in range(MAX_JETS)])
                     #print(mask_list[-1])

                     # lepton_m_ls = eTOm([lepton_pt_ls, lepton_eta_ls, lepton_phi_ls, lepton_e_ls])

                     lepton_mask_list.append([True])
                     
                     # Padding: Adding 0.0 to feature lists until they are all the same length
                     # Maybe we want to revisit this to set sin and cos to some incorrect values (-100.0)
                     # Making all feature lists the same length so they can be np.array later
                     if len(jet_m_ls)> MAX_JETS: jet_m_ls = jet_m_ls[:MAX_JETS]
                     if len(jet_pt_ls)> MAX_JETS: jet_pt_ls = jet_pt_ls[:MAX_JETS]
                     if len(jet_eta_ls)> MAX_JETS: jet_eta_ls = jet_eta_ls[:MAX_JETS]
                     if len(jet_phi_ls)> MAX_JETS: jet_phi_ls = jet_phi_ls[:MAX_JETS]
                     if len(jet_sin_phi_ls)> MAX_JETS: jet_sin_phi_ls = jet_sin_phi_ls[:MAX_JETS]
                     if len(jet_cos_phi_ls)> MAX_JETS: jet_cos_phi_ls = jet_cos_phi_ls[:MAX_JETS]
                     if len(jet_ptx_ls)> MAX_JETS: jet_ptx_ls = jet_ptx_ls[:MAX_JETS]
                     if len(jet_pty_ls)> MAX_JETS: jet_pty_ls = jet_pty_ls[:MAX_JETS]

                     if len(jet_e_ls)> MAX_JETS: jet_e_ls = jet_e_ls[:MAX_JETS]
                     if len(jet_b_ls)> MAX_JETS: jet_b_ls = jet_b_ls[:MAX_JETS]

                     for l in [jet_m_ls, jet_pt_ls, jet_eta_ls, jet_phi_ls, jet_sin_phi_ls, jet_cos_phi_ls, jet_e_ls,
                               jet_b_ls, jet_ptx_ls, jet_pty_ls
                            #    jet_b60_ls,
                            #    jet_b70_ls, 
                            #    jet_b77_ls,
                            #    jet_b85_ls,
                               ]:
                            if len(l) >= MAX_JETS:
                                   l = l[:MAX_JETS]
                            while len(l) < MAX_JETS:
                                   l.append(0.0)
                     # Appending event feature lists to data set ((EVENTS, MAX_JETS)) feature lists

                     jet_mass_list.append(jet_m_ls) 
                     jet_pt_list.append(jet_pt_ls) 
                     jet_eta_list.append(jet_eta_ls)
                     jet_phi_list.append(jet_phi_ls)
                     jet_sin_phi_list.append(jet_sin_phi_ls)
                     jet_cos_phi_list.append(jet_cos_phi_ls)

                     jet_ptx_list.append(jet_ptx_ls)
                     jet_pty_list.append(jet_pty_ls)

                     jet_btag_list.append(jet_b_ls)
                     jet_e_list.append(jet_e_ls)
                     eventNumber_list.append([ float(x.encode("utf-8").hex())  for x in list(nominal.truthTop_isLepDecay)])
                     allHad_list.append
                     # if 'ljets' in args.topo:
                     #     lepton_mass_list.append(lepton_m_ls)
                     #     lepton_pt_list.append(lepton_pt_ls)
                     #     lepton_eta_list.append(lepton_eta_ls)
                     #     lepton_phi_list.append(lepton_phi_ls)
                     #     lepton_sin_phi_list.append(lepton_sin_phi_ls)
                     #     lepton_cos_phi_list.append(lepton_cos_phi_ls)
                     #     lepton_e_list.append(lepton_e_ls)
                     #     lepton_etag_list.append(lepton_etag_ls)
                     #     lepton_mutag_list.append(lepton_mutag_ls)
                         
                     # Feature lists for: met, met_phi, ls - ONLY one value per event

                     met_met_ls, met_phi_ls = nominal.met_met, nominal.met_phi

                     met_sin_phi_ls, met_cos_phi_ls = math.sin(nominal.met_phi),math.cos(nominal.met_phi)

                     # Appending event features with onlhy one entry per event (MET)
                     met_met_list.append(met_met_ls); met_phi_list.append(met_phi_ls); met_sin_phi_list.append(met_sin_phi_ls); met_cos_phi_list.append(met_cos_phi_ls);
                     met_x_list.append( nominal.met_met * math.cos(nominal.met_phi) )
                     met_y_list.append( nominal.met_met * math.sin(nominal.met_phi) )

                     #Get the REGRESSIONS that you want to estimate

                     if 'ljets' in args.topo:
                         isLjetsEvent, truth_nu = getTruthNeutrino(nominal)

                         neutrino_pt_ls, neutrino_eta_ls, neutrino_phi_ls, neutrino_sin_phi_ls, neutrino_cos_phi_ls, neutrino_e_ls = truth_nu.Pt(), truth_nu.Eta(), truth_nu.Phi(), math.sin(truth_nu.Phi()), math.cos(truth_nu.Phi()), truth_nu.E()

                         neutrino_pt_list.append(neutrino_pt_ls); neutrino_eta_list.append(neutrino_eta_ls); neutrino_phi_list.append(neutrino_phi_ls); neutrino_sin_phi_list.append(neutrino_sin_phi_ls); neutrino_cos_phi_list.append(neutrino_cos_phi_ls); neutrino_e_list.append(neutrino_e_ls)
                     else:
                         #mask the regression targets using the value "nan"
                         neutrino_pt_list.append(float("nan")); neutrino_eta_list.append(float("nan")); neutrino_phi_list.append(float("nan")); neutrino_sin_phi_list.append(float("nan")); neutrino_cos_phi_list.append(float("nan")); neutrino_e_list.append(float("nan"))


                     # Appending the variables for the REGRESSIONS
                     #neutrino_px_list(neutrino_px_ls); neutrino_py_list(neutrino_py_ls); neutrino_pz_list(neutrino_pz_ls); neutrino_eta_list(neutrino_eta_ls)

                     t1q1_list.append(t1q1); t1q2_list.append(t1q2); t1b_list.append(t1b)
                     t2q1_list.append(t2q1); t2q2_list.append(t2q2); t2b_list.append(t2b)
                     t3q1_list.append(t3q1); t3q2_list.append(t3q2); t3b_list.append(t3b)
                     t4q1_list.append(t4q1); t4q2_list.append(t4q2); t4b_list.append(t4b)

                     empty = -1
                     empty_list.append(empty)

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
              indices = np.where(np.array(eventNumber_list) % modulus == remainder)
              print(f'Outpufile: {out}')
              print(f'indices for {remainder} after division by {modulus}:',indices)
              # Print out only elements that are in indices:
              print('eventNumber_list:',np.array(eventNumber_list)[indices])
              write(out, args.topo, indices,
               jet_mask_list,
               jet_mass_list,
               jet_pt_list,
               jet_eta_list,
               jet_phi_list,
               jet_sin_phi_list,
               jet_cos_phi_list,
               jet_ptx_list,
               jet_pty_list,
               jet_btag_list,
               jet_btag60_list,
               jet_btag70_list,
               jet_btag77_list,
               jet_btag85_list,
               jet_e_list,
               lepton_mask_list,
               lepton_mass_list,
               lepton_pt_list,
               lepton_eta_list,
               lepton_phi_list,
               lepton_sin_phi_list,
               lepton_cos_phi_list,
               lepton_e_list,
               lepton_etag_list,
               lepton_mutag_list,
               met_met_list,
               met_phi_list,
               met_sin_phi_list,
               met_cos_phi_list,
               met_x_list,
               met_y_list,
               neutrino_pt_list,
               neutrino_eta_list,
               neutrino_phi_list,
               neutrino_sin_phi_list,
               neutrino_cos_phi_list,
               neutrino_e_list,
               t1q1_list,
               t1q2_list,
               t1b_list,
               t2q1_list,
               t2q2_list,
               t2b_list,
               t3q1_list,
               t3q2_list,
               t3b_list,
               t4q1_list,
               t4q2_list,
               t4b_list,
               empty_list,

              )
                     



def write(outloc, topo, indices,
          jet_mask_list,
          jet_mass_list,
          jet_pt_list,
          jet_eta_list,
          jet_phi_list,
          jet_sin_phi_list,
          jet_cos_phi_list,
          jet_ptx_list,
          jet_pty_list,
          jet_btag_list,
          jet_btag60_list,
          jet_btag70_list,
          jet_btag77_list,
          jet_btag85_list,
          jet_e_list,
          lepton_mask_list,
          lepton_mass_list,
          lepton_pt_list,
          lepton_eta_list,
          lepton_phi_list,
          lepton_sin_phi_list,
          lepton_cos_phi_list,
          lepton_e_list,
          lepton_etag_list,
          lepton_mutag_list,
          met_met_list,
          met_phi_list,
          met_sin_phi_list,
          met_cos_phi_list,
          met_x_list,
          met_y_list,
          neutrino_pt_list, 
          neutrino_eta_list, 
          neutrino_phi_list, 
          neutrino_sin_phi_list, 
          neutrino_cos_phi_list, 
          neutrino_e_list,
          t1q1_list,
          t1q2_list,
          t1b_list,
          t2q1_list,
          t2q2_list,
          t2b_list,
          t3q1_list,
          t3q2_list,
          t3b_list,
          t4q1_list,
          t4q2_list,
          t4b_list,

          empty_list,

):

       # print('Indices Size: ',len(indices[0]))
       # print('Mask Size: ',len(jet_mask_list),len(jet_mask_list[0]))
       # print('Mass Size: ',len(jet_mass_list))
       
       HDF5 = h5py.File(outloc, 'w')

       #INPUTS group
       inputs_group = HDF5.create_group('INPUTS')

       jet_group = inputs_group.create_group('Source')
       jet_mask_set = jet_group.create_dataset('MASK', data=np.array(jet_mask_list, dtype='bool')[indices])
       
       jet_mass_set = jet_group.create_dataset('mass', data=np.array(jet_mass_list, dtype=np.float32)[indices])
       
       jet_pt_set = jet_group.create_dataset('pt', data=np.array(jet_pt_list, dtype=np.float32)[indices])
       jet_eta_set = jet_group.create_dataset('eta', data=np.array(jet_eta_list, dtype=np.float32)[indices])
       #jet_phi_set = jet_group.create_dataset('phi', data=np.array(jet_phi_list, dtype=np.float32)[indices])
       jet_sin_phi_set = jet_group.create_dataset('sin_phi', data=np.array(jet_sin_phi_list,dtype=np.float32)[indices])
       jet_cos_phi_set = jet_group.create_dataset('cos_phi', data=np.array(jet_cos_phi_list,dtype=np.float32)[indices])
       print(len(jet_pt_list))
       print(len(jet_ptx_list))
       jet_ptx_set = jet_group.create_dataset('pt_x', data=np.array(jet_ptx_list,dtype=np.float32)[indices])
       jet_pty_set = jet_group.create_dataset('pt_y', data=np.array(jet_pty_list,dtype=np.float32)[indices])

       jet_btag_set = jet_group.create_dataset('btag', data=np.array(jet_btag_list, dtype=np.float32)[indices])
       jet_e_set = jet_group.create_dataset('e', data=np.array(jet_e_list, dtype=np.float32)[indices])

       # Commented out bc we don't are not interested in leptons for our Analysis
       # if 'ljets' in topo:
       #     lepton_group = inputs_group.create_group('Leptons')
       #     lepton_mask_set = lepton_group.create_dataset('MASK', data=np.array(lepton_mask_list, dtype='bool')[indices])
       #     lepton_mass_set = lepton_group.create_dataset('mass', data=np.array(lepton_mass_list, dtype=np.float32)[indices])
       #     lepton_pt_set = lepton_group.create_dataset('pt', data=np.array(lepton_pt_list, dtype=np.float32)[indices])
       #     lepton_eta_set = lepton_group.create_dataset('eta', data=np.array(lepton_eta_list, dtype=np.float32)[indices])
       #     #lepton_phi_set = lepton_group.create_dataset('phi', data=np.array(lepton_phi_list, dtype=np.float32)[indices])
       #     lepton_sin_phi_set = lepton_group.create_dataset('sin_phi', data=np.array(lepton_sin_phi_list, dtype=np.float32)[indices])
       #     lepton_cos_phi_set = lepton_group.create_dataset('cos_phi', data=np.array(lepton_cos_phi_list, dtype=np.float32)[indices])
       #     lepton_e_set = lepton_group.create_dataset('e', data=np.array(lepton_e_list, dtype=np.float32)[indices])
       #     lepton_etag_set = lepton_group.create_dataset('etag', data=np.array(lepton_etag_list, dtype=np.float32)[indices])
       #     lepton_mutag_set = lepton_group.create_dataset('mutag', data=np.array(lepton_mutag_list, dtype=np.float32)[indices])


       met_group = inputs_group.create_group('Met')
       met_met_set = met_group.create_dataset('met', data=np.array(met_met_list, dtype=np.float32)[indices])
       #met_sumet_set  = met_group.create_dataset('sumet',
       #met_phi_set = met_group.create_dataset('phi', data=np.array(met_phi_list, dtype=np.float32)[indices])
       met_sin_phi_set = met_group.create_dataset('sin_phi', data=np.array(met_sin_phi_list, dtype=np.float32)[indices])
       met_cos_phi_set  = met_group.create_dataset('cos_phi', data=np.array(met_cos_phi_list, dtype=np.float32)[indices])
       
       met_sin_phi_set = met_group.create_dataset('met_x', data=np.array(met_x_list, dtype=np.float32)[indices])
       met_cos_phi_set  = met_group.create_dataset('met_y', data=np.array(met_y_list, dtype=np.float32)[indices])


       #print(pt_list)
       # Comment out; First trying out solely matching
       #REGRESSIONS group
       # regressions_group = HDF5.create_group("REGRESSIONS")

       # regression_event_group = regressions_group.create_group("EVENT")

       # neutrino_pt_set = regression_event_group.create_dataset('neutrino_pt', data=np.array(neutrino_pt_list, dtype=np.float32)[indices])
       # neutrino_eta_set = regression_event_group.create_dataset('neutrino_eta', data=np.array(neutrino_eta_list, dtype=np.float32)[indices])
       #neutrino_phi_set = regression_event_group.create_dataset('neutrino_phi', data=np.array(neutrino_phi_list, dtype=np.float32)[indices])
       # neutrino_sin_phi_set = regression_event_group.create_dataset('neutrino_sin_phi', data=np.array(neutrino_sin_phi_list, dtype=np.float32)[indices])
       # neutrino_cos_phi_set = regression_event_group.create_dataset('neutrino_cos_phi', data=np.array(neutrino_cos_phi_list, dtype=np.float32)[indices])
       # neutrino_e_set = regression_event_group.create_dataset('neutrino_e', data=np.array(neutrino_e_list, dtype=np.float32)[indices])

       #ttbar_invariant_mass = regressions_group.create_dataset("invariant_mass", data=np.array(XXXXX, dtype=np.float32)[indices])
       #log_invariant_mass
       #log_neutrino_eta = regressions_group.create_dataset("log_neutrino_eta", data=np.array(
       #log_neutrino_px = regressions_group.create_dataset("log_neutrino_px", data=np.array(
       #log_neutrino_py = regressions_group.create_dataset("log_neutrino_py", data=np.array(
       #log_neutrino_pz = regressions_group.create_dataset("log_neutrino_pz", data=np.array(
       #neutrino_eta = regressions_group.create_dataset("neutrino_eta", data=np.array(
       #neutrino_px = regressions_group.create_dataset("neutrino_px", data=np.array(
       #neutrino_py = regressions_group.create_dataset("neutrino_py", data=np.array(
       #neutrino_pz = regressions_group.create_dataset("neutrino_pz", data=np.array(
        


       #TARGETS group

       targets_group = HDF5.create_group('TARGETS')

       if 'ljets' in topo:   
              print('ERROR: no leptons!')
              exit(1)
              # particle_group_t1 = targets_group.create_group('th')
              # t1q1_set = particle_group_t1.create_dataset('q1', data=np.array(t1q1_list, dtype=np.int64)[indices])
              # t1q2_set = particle_group_t1.create_dataset('q2', data=np.array(t1q2_list, dtype=np.int64)[indices])
              # t1b_set = particle_group_t1.create_dataset('b', data=np.array(t1b_list, dtype=np.int64)[indices])

              # particle_group_t2 = targets_group.create_group('tl')
              # t2l_set = particle_group_t2.create_dataset('l', data=np.array(t2q1_list, dtype=np.int64)[indices])
              #t2nu_set = particle_group_t2.create_dataset('nu', data=np.array(t2q2_list, dtype=np.int64)[indices])
              # t2b_set = particle_group_t2.create_dataset('b', data=np.array(t2b_list, dtype=np.int64)[indices])
       else:
              particle_group_t1 = targets_group.create_group('t1')
              particle_group_t1.create_dataset('q1', data=np.array(t1q1_list, dtype=np.int32)[indices])
              particle_group_t1.create_dataset('q2', data=np.array(t1q2_list, dtype=np.int32)[indices])
              particle_group_t1.create_dataset('b', data=np.array(t1b_list, dtype=np.int32)[indices])

              particle_group_t2 = targets_group.create_group('t2')
              particle_group_t2.create_dataset('q1', data=np.array(t2q1_list, dtype=np.int32)[indices])
              particle_group_t2.create_dataset('q2', data=np.array(t2q2_list, dtype=np.int32)[indices])
              particle_group_t2.create_dataset('b', data=np.array(t2b_list, dtype=np.int32)[indices])
              
              particle_group_t3 = targets_group.create_group('t3')
              particle_group_t3.create_dataset('q1', data=np.array(t3q1_list, dtype=np.int32)[indices])
              particle_group_t3.create_dataset('q2', data=np.array(t3q2_list, dtype=np.int32)[indices])
              particle_group_t3.create_dataset('b', data=np.array(t3b_list, dtype=np.int32)[indices])
              
              particle_group_t4 = targets_group.create_group('t4')
              particle_group_t4.create_dataset('q1', data=np.array(t4q1_list, dtype=np.int32)[indices])
              particle_group_t4.create_dataset('q2', data=np.array(t4q2_list, dtype=np.int32)[indices])
              particle_group_t4.create_dataset('b', data=np.array(t4b_list, dtype=np.int32)[indices])

       # particle_group_extrajet = targets_group.create_group('extrajet_parton')
       # extrajet_set = particle_group_extrajet.create_dataset('extrajet', data=np.array(extrajet_list, dtype=np.int64)[indices])
       #empty_set = particle_group_extrajet.create_dataset('empty', data=np.array(empty_list, dtype=np.int64)[indices])

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
def assignIndicesljetsttbar( nominal ):

       # Here is where you need to do the truth matching
       #This code is for the spin correlation ntuples
       
       t1q1, t1q2, t1b, t2q1, t2q2, t2b = -1, -1, -1, -1, -1, -1
       t3q1, t3q2, t3b, t4q1, t4q2, t4b = -1, -1, -1, -1, -1, -1
       
       t1q1, t1q2, t1b = nominal.t1_w0_Jeti, nominal.t1_w1_Jeti, nominal.t1_b_Jeti
       t2q1, t2q2, t2b = nominal.t2_w0_Jeti, nominal.t2_w1_Jeti, nominal.t2_b_Jeti
       t3q1, t3q2, t3b = nominal.t3_w0_Jeti, nominal.t3_w1_Jeti, nominal.t3_b_Jeti
       t4q1, t4q2, t4b = nominal.t4_w1_Jeti, nominal.t4_w0_Jeti, nominal.t4_b_Jeti

       
       result = [t1q1, t1q2, t1b, t2q1, t2q2, t2b, t3q1, t3q2, t3b, t4q1, t4q2, t4b]
       # print(targetsAreUnique(result))

       # if not targetsAreUnique(result):
       #      print("ERROR: similar jets are used for multiple partons")


       return (t1q1, t1q2, t1b, t2q1, t2q2, t2b, t3q1, t3q2, t3b, t4q1, t4q2, t4b), targetsAreUnique(result)

        
def displayFoundFiles(fPaths : list):
       numFile = len(fPaths)
       print(f'Found {numFile} files:')
       for i in range(numFile):
              name = fPaths[i][fPaths[i].find(args.inloc)+len(args.inloc):]
              if name[0] == '.':
                     name = name[1:]
              print(name)

if __name__ == "__main__":
       
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
       parser.add_argument('-p','--prefix',default='year18_', help='Prefix to separate files')
       parser.add_argument('-s','--suffix',default='_year18', help='Prefix to separate files')
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
       
       btag_alg = args.btag
       if not btag_alg:
              btag_alg = 'DL1r'
       

       # For now we choose only two files
       selectionSubstring = [ 
                                   'user.nhidic.412043.aMcAtNloPythia8EvtGen.DAOD_PHYS.e7101_a907_r14859_p5855.4thad26_240130_v06.3_output_root.nominal.root',
                                   'user.nhidic.412043.aMcAtNloPythia8EvtGen.DAOD_PHYS.e7101_a907_r14860_p5855.4thad26_240130_v06.3_output_root.nominal.root',
                                   'user.nhidic.412043.aMcAtNloPythia8EvtGen.DAOD_PHYS.e7101_a907_r14861_p5855.4thad26_240130_v06.3_output_root.nominal.root',
                                   ]

       if not len(selectionSubstring) ==0 :
              file_paths = [x  for x in file_paths  if any(substring in x for substring in selectionSubstring)]
       print(file_paths)
       displayFoundFiles(file_paths)
       # Run production:
       source(file_paths, args)


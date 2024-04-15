import pickle
import ROOT


def plotHistos(histoDict : dict, presentNames : dict, missingNames : list)->None:
    c = ROOT.TCanvas('c', 'c', 800, 600)
    colors = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen, ROOT.kYellow, ROOT.kOrange, ROOT.kMagenta, ROOT.kCyan, ROOT.kPink, ROOT.kViolet, ROOT.kAzure, ROOT.kTeal, ROOT.kSpring, ROOT.kGray, ROOT.kWhite, ROOT.kBlack]
    nameOutput = ''
    legend = ROOT.TLegend(0.6, 0.7, 0.9, 0.9)
    for histoName in presentNames:
        if histoName in missingNames:
            continue
        nameOutput += histoName+'-'
        histoDict[histoName].SetLineColor(colors[ list(presentNames.keys()).index(histoName) %  len(colors) ])
        histoDict[histoName].SetStats(0)
        histoDict[histoName].Draw('SAME')
        legend.AddEntry(histoDict[histoName], presentNames[histoName], 'l')
    legend.Draw()
    c.SaveAs(f'plots/{nameOutput}.png')
def main(path: str)->None:
    with open(path, 'rb') as f:
        histos = pickle.load(f)

    c = ROOT.TCanvas('c', 'c', 800, 600)
    for histoName in histos:
        histos[histoName].Draw()
        c.SaveAs(f'plots/{histoName}.png')

    massDict = {
        'TruthMass' : 'Invariant Mass From Partons',
        'MatchedMass' : 'Invariant Mass From Partons with Matches',
        'MatchedJetsMass' : 'Invariant Mass From Matched Jets',
    }
    deltaRDict = {
        'DeltaR' : 'DeltaR between top all partons and top partons with matches',
        'topTruthRecoDeltaR' : 'DeltaR between top all partons and top jets with matches',
        'partonRecoDeltaR' : 'DeltaR between top partons with matches and top jets with matches',
        # 'kek':'lmao',
    }
    ptRatioDict = {
        'PtRatio' : 'PtRatio between top all partons and top partons with matches',
        'topTruthRecoPtRatio' : 'PtRatio between top all partons and top jets with matches',
        'partonRecoPtRatio' : 'PtRatio between top partons with matches and top jets with matches',
    }

    for dictNames in [massDict, deltaRDict, ptRatioDict]:
        allKeyPresent = True
        missingKeys = []
        for histoName in dictNames:
            if not histoName in histos:
                allKeyPresent = False
                missingKeys.append(histoName)
        if allKeyPresent:
            print('All keys are present')
        else:
            print('Not All Keys Are present!\n  Missing keys:', missingKeys)
        plotHistos(histos, dictNames, missingKeys)

if __name__ == '__main__':
    main(path='pickles/histograms_year18__tttt_truth_03-04-2024_test.pkl')
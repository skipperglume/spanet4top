import pickle
import ROOT

def main(path: str):
    with open(path, 'rb') as f:
        histos = pickle.load(f)

    c = ROOT.TCanvas('c', 'c', 800, 600)
    for histoName in histos:
        histos[histoName].Draw()
        c.SaveAs(f'plots/{histoName}_truth.png')

if __name__ == '__main__':
    main(path='plots/histograms_tttt_truth_28-03-2024.pkl')
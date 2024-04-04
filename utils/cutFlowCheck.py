import pickle
import numpy as np
import matplotlib.pyplot as plt

def main(path: str, typeDict : dict)->None:
    '''
    Main Body of the code.
    '''
    with open(path, 'rb') as f:
        cutFlow = pickle.load(f)
    for cut in cutFlow:
        nEvents = len(cutFlow['numberOfEvents'])
        if cut not in typeDict:
            print(f'Cut {cut} not in typeDict')
            continue
        if typeDict[cut] == 'bool':
            array = np.array(cutFlow[cut])
            printString = ''
            printString += f'Cut {cut} : \n'
            printString += f'  Number of passed events: {len(np.where(array == 1)[0])} \n'
            printString += f'  Number of failed events: {len(np.where(array == 0)[0])} \n'
            printString += f'  Efficiency: {len(np.where(array == 1)[0])/nEvents} \n'
            print(printString )
        elif typeDict[cut] == 'uint':
            array = np.array(cutFlow[cut])
            array = array.astype(int)
            printString = ''
            printString += f'Cut {cut} : \n'
            # Add info on min and max value of the cut
            printString += f'  Min value: {np.min(array)} \n'
            printString += f'  Max value: {np.max(array)} \n'
            for i in range(np.min(array), np.max(array)+1):
                printString += f'    Number of events with value {i}: {len(np.where(array == i)[0])} \n'
                printString += f'      Efficiency: {len(np.where(array == i)[0])/nEvents} \n'
                

            print(printString )
            # print(cut, len(cutFlow[cut]),  np.max(cutFlow[cut]), )
            # len(np.where(np.array(cutFlow[cut]) == 1)[0])/nEvents)

if __name__ == '__main__':
    typeDict = {
        'numberOfEvents' : 'bool',
        'JetNumber' : 'bool',
        'bJetNumber' : 'bool',
        'DecayType' : 'uint',
        'isAllHad' : 'bool',
        'Uniqueness' : 'bool',
        'AssignemntCheck' : 'bool',
    }


    

    # Example list of True/False booleans representing cut results
    cuts = [True, False, True, True, False]

    # Calculate efficiencies
    total_events = len(cuts)
    passing_events = sum(cuts)
    efficiencies = [passing_events / total_events]

    for i in range(1, len(cuts)):
        passing_events -= int(cuts[i - 1])
        efficiencies.append(passing_events / total_events)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, len(cuts) + 1), efficiencies, color='skyblue')
    plt.xlabel('Cut Index')
    plt.ylabel('Efficiency')
    plt.title('Efficiencies of Cuts')
    plt.xticks(range(1, len(cuts) + 1))
    plt.grid(True)
    plt.savefig('plots/cutflow.png')

    main(path='pickles/cutflow_year18__tttt_04-04-2024_test.pkl', typeDict=typeDict)
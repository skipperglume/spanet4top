import os

def displayFoundFiles(fPaths : list):
    numFile = len(fPaths)
    if numFile > 0:
        print('==============================================================')
        print(f'Looking in {numFile} files:')
        print(f'Found {numFile} files:')
        print('==============================================================')
    else:
        print('==============================================================')
        print(f'No files found')
        print('==============================================================')

    for i in range(numFile):
        name = os.path.basename(fPaths[i])
        if name == '':
            print(f'ERROR: {fPaths[i]} is a directory')
            exit(1)
        else:
            print(name)
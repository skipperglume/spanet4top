import os

def displayFoundFiles(fPaths : list) -> None:
    numFile = len(fPaths)
    if numFile > 0:
        print('==============================================================')
        print(f'Looking in folder:\n{os.path.split(fPaths[0])[0]}')
        print(f'Found {numFile} files:')
        print('==============================================================')
    else:
        print('==============================================================')
        print(f'No files found')
        print('==============================================================')
        return

    for i in range(numFile):
        name = os.path.basename(fPaths[i])
        if name == '':
            print(f'ERROR: {fPaths[i]} is a directory')
            exit(1)
        else:
            print('  ',name)
## Goal
We `aim` to reconstruct all-hadronic tttt events. The code for input creation was feveloped for this.
### To process all file into inputs:

    python produceHDF5.py  --prefix 'year15+16_' && /
    python produceHDF5.py  --prefix 'year17_' && /
    python produceHDF5.py  --prefix 'year18_'

### To test all of them: 

    python produceHDF5.py  --prefix 'year15+16_' --test && /
    python produceHDF5.py  --prefix 'year17_' --test && /
    python produceHDF5.py  --prefix 'year18_' --test

### To `mv` multiple files selected with several condition (e.g. `grep`)

    ls -lh | grep [CONDITION] | grep [FILE NAME SELECTION] -o   |  xargs -I {} scp   {} backup/    

### To merge several files of diffrent modality:

    python mergeH5Files.py --mode odd|even

### To test the correctness of branches and file naming:

    python mergeH5Files.py --mode odd|even --test

### To display the basic information about inputs:

    python h5Test.py -f [FILE_PATH] -n [NUMBER_OF_DATA_POINTS] -v 

  python testModelEval.py  ./models/even/ 
  python testModelEval.py  ./models/even/  /home/timoshyd/spanet4Top/ntuples/four_top_SPANET_input/predictions 
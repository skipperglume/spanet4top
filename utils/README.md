## Goal
We `aim` to reconstruct all-hadronic tttt events. The code for input creation was feveloped for this.
### To process all file into inputs:

    python3 produceHDF5.py  --prefix 'year15+16_' && /
    python3 produceHDF5.py  --prefix 'year17_' && /
    python3 produceHDF5.py  --prefix 'year18_'

### To test `produceHDF5.py`: 

    python3 produceHDF5.py  --prefix 'year15+16_' --test 

### New python3 script to produce SPANet training\prediction datasets:

    python3 bigInputProduction.py --saveSize 20000

#### To produce all needed for training:

    python3 bigInputProduction.py  --tag _train   --prefix 'year15+16_' && /
    python3 bigInputProduction.py  --tag _train  --prefix 'year17_' && /
    python3 bigInputProduction.py  --tag _train  --prefix 'year18_'

#### To produce all needed for prediction:

    python3 bigInputProduction.py  --tag _pred  --ignoreCuts  --prefix 'year15+16_' && /
    python3 bigInputProduction.py  --tag _pred  --ignoreCuts  --prefix 'year17_' && /
    python3 bigInputProduction.py  --tag _pred  --ignoreCuts  --prefix 'year18_'

Main feature of this script is to save each `saveSize` to h5 file, which prevents from memory overloading in the shell.

### To `mv` multiple files selected with several condition (e.g. `grep`)

    ls -lh | grep [CONDITION] | grep [FILE NAME SELECTION] -o   |  xargs -I {} scp   {} backup/    

### To merge several files of diffrent modality:

    python3 mergeH5Files.py --mode odd|even

### To test the correctness of branches and file naming:

    python3 mergeH5Files.py --mode odd|even --test

### To display the basic information about inputs:

    python3 h5Test.py -f [FILE_PATH] -n [NUMBER_OF_DATA_POINTS] -v 

  python3 testModelEval.py  ./models/even/ 
  python3 testModelEval.py  ./models/even/  /home/timoshyd/spanet4Top/ntuples/four_top_SPANET_input/predictions 
import os, sys
import glob
import datetime
class JobMaker:
    def __init__(self, batchJobsDir):
        self.__batchJobsDir = batchJobsDir
        self.__batchJobString = ''
    def clearJob(self):
        self.__batchJobString = ''
        
    # With a shorthand notation for multiple arguments:
    def createBatchJob(self, jobName, outputLog, errorLog, partition, nodes, ntasks, time, command):
        self.__batchJobString += '#SBATCH --job-name=' + jobName + '\n'
        self.__batchJobString += '#SBATCH --output=' + outputLog + '\n'
        self.__batchJobString += '#SBATCH --error=' + errorLog + '\n'
        self.__batchJobString += '#SBATCH --partition=' + partition + '\n'
        self.__batchJobString += '#SBATCH --nodes=' + nodes + '\n'
        self.__batchJobString += '#SBATCH --ntasks=' + ntasks + '\n'
        self.__batchJobString += '#SBATCH --time=' + time + '\n'
        self.__batchJobString += '\n'
        self.__batchJobString += command + '\n'
        
    def writeBatchJob(self):
        date = datetime.datetime.now().strftime("%d-%m-%y")
        with open(self.__batchJobsDir + 'batchJob.sh', 'w') as f:
            f.write(self.batchJobString)
            f.close()

if __name__ == '__main__':
    batchJobsDir = './batchJobs/'
    
    jobMaker = JobMaker(batchJobsDir)

    for period in []:
        for type in ['train', 'pred']:
    batchJobString = ''
    batchJobString += '#!/bin/bash \n'
    
    #SBATCH --job-name=prod_train_15_16
    #SBATCH --output=output.log
    #SBATCH --error=error.log
    #SBATCH --partition=ucjf
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --time=8:00:00

    # . ../../setup.sh 
    # . ../../spaenv/bin/activate

    # python3 bigInputProduction.py  --tag 15j_train  --prefix 'year15+16_'
    
    print(batchJobString)
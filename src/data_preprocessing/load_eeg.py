"""
    A Script for loading EEG data in Directory of THINGS-EEG2 Dataset:

    QUICK SETUPS:

    1) Things- EEG2 Proposed Preprocessing

        python loadEEG.py <path> --subjIds 1 2 3  --sfreq 100 --sessionNumber 4

    2) ATM - proposed Preprocessing

        python loadEEG.py <path> --subjIds 1 2 3 --sfreq 250 --sessionNumber 4 --repMean

    3) My Preprocessing 

        python loadEEG.py< path> --subjIds 1 2 3 --repMean --sfreq 1000 --sessionNumber 4 --repMean


    Usage: (write down in a terminal: )
        + loadEEG path --subjIds LIST ---repMean --sfreq INT --mnvdim --check INT --sessionNumber INT 

    Arguments:
        + path - a path to a directory where the data are located: The directory should contain: raw_data folder in which raw_data are stored ()
            Warning! If you intendt to check with preprocessed data - yoou will have to also have a preprocessed_data folder! 
        + subjNum LIST - (from 1 to 10) Number of a subject separated with space. Default is all subjects 1-10
        + --repMean -  default False - n analysis mode whether to Average the Repeating Evoked EEG to Image (ATMs thought that it improved performance)
        + --sfreq INT - default (1000), Resampling Frequency. If We want analysis to follow strictly the ImageNet and ATM papers:
                100 - will make the same analysis as in Image Net
                1000 - will make my own analysis, adjusting cuts and performing NO downsampling 
        + --mnvdim - In Which dimension do we compute our Multivariat Noise Normalization (Whitening) - either "time" or "epochs"
        + --check INT - accepts 0,1,2 --> performs data Checks:  
                0 - No data Check; 
                1 - Plot ERPs across all Image Samples 
                2 - Check Allignment with Preprocessed Data + Plot ERPs in a PDF (Warning! Requires preprocessed_data Folder in your path!)
        + --sessionsNumber - how many sessions do you want to include (up to 4)



"""

from load_eeg_utils import epoching,mvnn,mergeData,saveData,checkData
import argparse
import numpy  as np
import os


if __name__ == "__main__":

    #### ===== ARGUMENTS ===== ####

    ### Parse Input Arguments

    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--subjIds', nargs='+', type=int, default=[i for i in range (1,11)])
    parser.add_argument('--sfreq',type=int,default=1000)
    parser.add_argument('--mnvdim',default="epochs")
    parser.add_argument('--repMean', action="store_true")
    parser.add_argument('--check',default=0,choices=[0,1,2],type=int)
    parser.add_argument('--sessionNumber', default = 1, choices=[1,2,3,4],type=int)
    args = parser.parse_args()

    for subjNum in args.subjIds:
        print("=== Getting EEG Folders ===\n")
        print(f"   Data from: {args.path}/RawEEG ")
        print(f"   Subj numb: {subjNum}")

        ## Create a new Save - Folder - PreprocessedEEG  
        savePath = os.path.join(args.path, 'PreprocessedEEG','sub-'+format(subjNum,'02'))
        if not os.path.isdir(savePath):
            os.makedirs(savePath)

        print(f"   Getting Preprocessed Data....")

        ### get Ref Channels: (For check Oz, Pz and OPz)

        ### Get Subject Raw Data
        print(f"   Getting Raw Data....")
        subjectPathRaw =  os.path.join(args.path,"RawEEG",'sub-'+format(subjNum,'02'))
        seednum = 20200220

        # Test
        dataPart = "raw_eeg_test.npy"
        print(args.sfreq)
        epoched_test, _, ch_names, times,_ = epoching(args.sessionNumber,subjectPathRaw,dataPart,args.sfreq,seednum)
        # Train:

        dataPart = "raw_eeg_training.npy"
        epoched_train, img_conditions_train, _, _,events = rawDict = epoching(args.sessionNumber,subjectPathRaw,dataPart,args.sfreq,seednum)

        ### Whiten Data (Good Practice!) - I need to check how to do it 
        whitened_test, whitened_train = mvnn(args.sessionNumber,args.mnvdim,epoched_test,epoched_train)


        ### Merge EEG Data

        ### Merge and save the test data ###

        test_dict, train_dict = mergeData(args.sessionNumber, whitened_test, whitened_train, img_conditions_train,
            ch_names, times, args.repMean, seednum)

        print(test_dict)


        ###  ======Check Data with Original EEG: ======

        if args.check == 2: #Data Alignment with original + ERPs 
            ## - Does not work as intended!!! --> Clearly Shuffeling totally changes where the Representations Lay 
            ### Get Preprocessed_Folder 
            subjectPathPreprocessedTest =os.path.join(savePath,"preprocessed_eeg_test.npy")
            subjectPathPreprocessedTrain =os.path.join(savePath,"preprocessed_eeg_training.npy")

            ### Get Preprocessed Data (For Checking Purposes!)
            dataPrepTest = np.load(subjectPathPreprocessedTest,allow_pickle=True).item()
            dataPrepTrain = np.load(subjectPathPreprocessedTrain,allow_pickle=True).item()

            prepShape = dataPrepTest['preprocessed_eeg_data'].shape # Images x Sessions x Chan (17) x Time (100)
            print(f"Test  Prep Shape:       Trials: {prepShape[0]};\n       Sessions: {prepShape[1]};\n       Channels: {prepShape[2]};\n       Timepoints: {prepShape[0]};\n" ,prepShape)
            prepShape = dataPrepTrain['preprocessed_eeg_data'].shape # Images x Sessions x Chan (17) x Time (100)
            print(f"Train Prep Shape:       Trials: {prepShape[0]};\n       Sessions: {prepShape[1]};\n       Channels: {prepShape[2]};\n       Timepoints: {prepShape[0]};\n" ,prepShape)
            checkData(test_dict,train_dict,dataPrepTest,dataPrepTrain)


        elif args.check == 1: # ERPs Only
            checkData(test_dict,train_dict)

        ### Save EEG Data
        saveData(savePath,test_dict,train_dict)










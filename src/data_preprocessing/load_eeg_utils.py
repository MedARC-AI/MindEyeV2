

def epoching(sessions,dataPath,dataPart,sfreq,seed):
    """
        Load Your data and performs Trial Segmentation. Data that are provided are taken directly from raw_data files and must be a dictionary with keys:
            + raw_eeg_data - contains a matrix of chan x time
            + sfreq - current sampling freq of the data (DO NOT CONFUSE WITH RESAMPLING FREQUENCY)
            + ch_types - a type of channel (last one has to be a stim type)
            + ch_names - names of channels (Elecrodes!)

        
        Peprocessing steps are as follows
            + Load Data
            + Transform into raw format usingmne.create_info  and mne.io.RawArray
            + find events with method find_events using Stim channel
            + Segment to trial - either 0.2 to 0.8 s as intendet (But WRONG in my opinion) or 0.1 to 0.1 (as it should be!)
            + Resample data when the latter is true to a specified freequency (Reduced Dimensionality) 
            + Sort the data - Required, for images are differently rpesented in repetitions - select UTMOST 2x the repeated images! in case of train

        Output:
            +Epoched Data - a matrix of data containing Images x Repetitions*Sessions x Channels x Time
                + Test dataset  - 200  x 20*4 x 63 x 100
                + Train dataset - 8750 x 2*4  x 63 x 100
    
    """
    import os 
    import mne
    import numpy as np
    from sklearn.utils import shuffle


    ### Loop across data collection sessions ###
    epoched_data = []
    img_conditions = []
    for s in range(sessions): # Iterate thorugh sessions


        ### ===  Load the EEG data and convert it to MNE raw format  === ###
        dataP = os.path.join(dataPath,'ses-'+format(s+1,'02'),dataPart)
        eeg_data = np.load(os.path.join(dataP),
        allow_pickle=True).item()

        # Get fields from dictionariry 
        ch_names = eeg_data['ch_names']
        sfreq2    = eeg_data['sfreq']
        ch_types = eeg_data['ch_types']
        eeg_data = eeg_data['raw_eeg_data']


        ### ===  Convert to MNE raw format - requires for smooth processing  === ###
        info = mne.create_info(ch_names, sfreq2, ch_types)
        raw = mne.io.RawArray(eeg_data, info)
        del eeg_data


        ### === Get events === ###
        events = mne.find_events(raw, stim_channel='stim')

        # Reject the target trials (event 99999)
        idx_target = np.where(events[:,2] == 99999)[0]
        events = np.delete(events, idx_target, 0)

        # Drop stim channel as it was used only in assessing the events:
        raw.drop_channels("stim") # It is done In-place


        ### === Epoching === ###
        if sfreq  == 1000: # Controlled by sfreq --> if we have lower than 1000 Sfreq we use 100ms before, 100ms of presenting 
            epochs = mne.Epochs(raw, events, tmin=-.1, tmax=.1, baseline=(None,0),
            preload=True) # Using baseline and SWAP 
        else: # 0.2 before 0.8 s after - as in Things-EEG2
            epochs = mne.Epochs(raw, events, tmin=-.2, tmax=.8, baseline=(None,0),
            preload=True) # Using baseline and SWAP 
        del raw 


        ### === Resampling === ###
        if sfreq < 1000: # if necessery resample to a give sfreq!
            epochs.resample(sfreq)
        ch_names = epochs.info['ch_names'] # save new Ch_name and TImes from Epochs
        times = epochs.times # 


        ### === Sort the data === ### 
        # #- Precautionary, and Important in Training Data (for Images were presented at random) and each session presents 2x half of Images
        data = epochs.get_data()
        events = epochs.events[:,2] # Gets ID of events
        img_cond = np.unique(events) # This is a list of All Image IDs in a given Session 
        del epochs

        # Select only a maximum number of EEG repetitions (For Test it was 20 times)
        if dataPart == "raw_eeg_test.npy":
            max_rep = 20
        else:
            max_rep = 2 #2x half of images were presented each session

        # Sorted data matrix of shape: Image conditions × EEG repetitions × EEG channels × EEG time points
        sorted_data = np.zeros((len(img_cond),max_rep,data.shape[1],data.shape[2]))
        for i in range(len(img_cond)):
            # Find the indices of the selected image condition
            idx = np.where(events == img_cond[i])[0]
            # Randomly select only the max number of EEG repetitions
            idx = shuffle(idx, random_state=seed, n_samples=max_rep)
            sorted_data[i] = data[idx]
        del data
        epoched_data.append(sorted_data) # Append for each Session
        img_conditions.append(img_cond) # append - resultin in 16750 image labels
        del sorted_data

    ### Output ###
    return epoched_data, img_conditions, ch_names, times, events


def mvnn(sessions, mvnn_dim,epoched_test, epoched_train):
    """
    It is almost as the original - I have my doubts - but it seems legit:
    Whiteniing data will make variance uniform across channels buuut i am unsure whether it will normalise artifacts
    
    Compute the covariance matrices of the EEG data (calculated for each
    time-point or epoch/repetitions of each image condition), and then average
    them across image conditions and data partitions. The inverse of the
    resulting averaged covariance matrix is used to whiten the EEG data
    (independently for each session).

    Parameters
    ----------
    sessions - number of sessions to include
    mvnn_dim - whether we perform whitening across times or across repetitions
    epoched_test, epoched_train - our data epoched by previous functioms 

    Returns
    -------
    whitened_test : list of float
    Whitened test EEG data.
    whitened_train : list of float
    Whitened training EEG data.

    """

    import numpy as np
    from tqdm import tqdm
    from sklearn.discriminant_analysis import _cov
    import scipy

    ### Loop across data collection sessions ###
    whitened_test = []
    whitened_train = []
    for s in range(sessions):
        session_data = [epoched_test[s], epoched_train[s]]

        ### Compute the covariance matrices ###
        # Data partitions covariance matrix of shape:
        # Data partitions × EEG channels × EEG channels
        sigma_part = np.empty((len(session_data),session_data[0].shape[2],
        session_data[0].shape[2]))
        for p in range(sigma_part.shape[0]): # 
            # Image conditions covariance matrix of shape:
            # Image conditions × EEG channels × EEG channels
            sigma_cond = np.empty((session_data[p].shape[0],
            session_data[0].shape[2],session_data[0].shape[2]))
            for i in tqdm(range(session_data[p].shape[0])): # For nice progress Bar
                cond_data = session_data[p][i]
                # Compute covariace matrices at each time point, and then
                # average across time points
                if mvnn_dim == "time": # Compute MEAN Covariance for Whitening across Repetitions (There are only 2...) 
                    sigma_cond[i] = np.mean([_cov(cond_data[:,:,t],
                        shrinkage='auto') for t in range(cond_data.shape[2])],
                        axis=0)
                # Compute covariace matrices at each epoch (EEG repetition),
                # and then average across epochs/repetitions
                elif mvnn_dim == "epochs":
                    sigma_cond[i] = np.mean([_cov(np.transpose(cond_data[e]), # 100 x 64 -- time x Channels  for every Repetitions of stimuli
                        shrinkage='auto') for e in range(cond_data.shape[0])],
                        axis=0)
            # Average the covariance matrices across image conditions
            sigma_part[p] = sigma_cond.mean(axis=0)
        # Average the covariance matrices across image partitions
        sigma_tot = sigma_part.mean(axis=0) 
        # Compute the inverse of the covariance matrix
        sigma_inv = scipy.linalg.fractional_matrix_power(sigma_tot, -0.5) # Compute Inverese Matrix as a final Whitening Step

        ### Whiten the data ###
        whitened_test.append(np.reshape((np.reshape(session_data[0], (-1,
            session_data[0].shape[2],session_data[0].shape[3])).swapaxes(1, 2)
            @ sigma_inv).swapaxes(1, 2), session_data[0].shape))
        whitened_train.append(np.reshape((np.reshape(session_data[1], (-1,
            session_data[1].shape[2],session_data[1].shape[3])).swapaxes(1, 2)
            @ sigma_inv).swapaxes(1, 2), session_data[1].shape))

    ### Output ###
    return whitened_test, whitened_train



def mergeData(session, whitened_test, whitened_train, img_conditions_train,
ch_names, times,repMean, seed):
    """
    Also almost unchange - i wanted it to be as close to the original as Possible!
    
    Merge the EEG data of all sessions together, shuffle the EEG repetitions
    across sessions and reshaping the data to the format:
    Image conditions × EEG repetitions × EEG channels × EEG time points.
    Then, the data of both test and training EEG partitions is saved.

    Parameters
    ----------
    session : how many sessions to inclue (although it IS required to be 4  because otherwise it will collapse)
    whitened_test : list of float  - Whitened test EEG data.
    whitened_train : list of float - Whitened training EEG data.
    img_conditions_train : list of int - Unique image conditions of the epoched and sorted train EEG data.
    ch_names : list of str -  EEG channel names.
    times : float EEG time points.
    repMean: Whethet to average the repetitions or not
    seed : int Random seed.

    Returns:
    test_dict  - a data dictionary that refleects preprocessed Train data
    train_dict - a data dictionary that refleects preprocessed Test data

    """

    import numpy as np
    from sklearn.utils import shuffle
    import os

    ### Merge and save the test data ###
    for s in range(session):
        if s == 0:
            merged_test = whitened_test[s]
        else:
            merged_test = np.append(merged_test, whitened_test[s], 1)
    del whitened_test

    # Shuffle the repetitions of different sessions
    idx = shuffle(np.arange(0, merged_test.shape[1]), random_state=seed)
    merged_test = merged_test[:,idx]
    # Insert the data into a dictionary
    test_dict = {
        'preprocessed_eeg_data': merged_test,
        'ch_names': ch_names,
        'times': times
    }
    del merged_test
  

    ### Merge and save the training data ###
    # Remember the data structure! It is not as straightforward as it seems: 4 Training sessions: Each with 8750 or so Images (Half Of All 165444) - Repeating 2 times
    # It means that ONLY after getting all 4 sessions we acheieve our 16540,4,64,100)

    for s in range(session):
        if s == 0:
            white_data = whitened_train[s]
            img_cond = img_conditions_train[s]
        else:
            white_data = np.append(white_data, whitened_train[s], 0)
            img_cond = np.append(img_cond, img_conditions_train[s], 0)
    del whitened_train, img_conditions_train
    # Data matrix of shape:
    # Image conditions × EEG repetitions × EEG channels × EEG time points
    merged_train = np.zeros((len(np.unique(img_cond)), white_data.shape[1]*2, # Bizzare way to structure you data but it makes sense!
        white_data.shape[2],white_data.shape[3]))
    for i in range(len(np.unique(img_cond))): # Aftrer gathering ALL images together we THEN try to order Out the data to reflect 4 different Image Viewing Condition
        # Find the indices of the selected category
        idx = np.where(img_cond == i+1)[0]
        for r in range(len(idx)):
            if r == 0:
                ordered_data = white_data[idx[r]]
            else:
                ordered_data = np.append(ordered_data, white_data[idx[r]], 0)
        merged_train[i] = ordered_data
    
    # Shuffle the repetitions of different sessions
    idx = shuffle(np.arange(0, merged_train.shape[1]), random_state=seed)
    merged_train = merged_train[:,idx]

   # Own Adjustmenet - across concatenating repetitions to a single Dataset:
    if repMean:
        merged_train = np.mean(merged_train,1)


    # Insert the data into a dictionary
    train_dict = {
    'preprocessed_eeg_data': merged_train,
    'ch_names': ch_names,
    'times': times
    }

    return test_dict, train_dict

def checkData(test_dict, train_dict, *args, **kwargs):

    """
        Performs data checks
         - If dataPreps are provided computes the mean of a difference between timepoints in 3 preselected Channels: Oz, Pz, POz and then ERP visualisation
         - If not, just performs ERP visualisation: 

         Average across Repetitions and IMAGES to get an ERP for a given individual on Ever channel:
         - Warning! Such ERPs are bound to be skewed in some cases! 

    """

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np

    ### === Check whether Loaded Preprocessed data and our preprocessing is Coinciding === 
    # (!!!) Work in progress - not as intended -  

    # ### Check Whether Prep data are provided
    dataPrepTest = args[0] if len(args) > 0 else None
    dataPrepTrain = args[1] if len(args) > 1 else None

    if dataPrepTest is not None and dataPrepTrain is not None:

        chanNames = ['Pz','Oz','POz']
        ### ger 3 reference Data for our own Things - for  PZ, Oz, POz
        refDataTest  =  [dataPrepTest['preprocessed_eeg_data'][:,:,0,:], dataPrepTest['preprocessed_eeg_data'][:,:,4,:], dataPrepTest['preprocessed_eeg_data'][:,:,12,:]]
        refDataTrain =  [dataPrepTrain['preprocessed_eeg_data'][:,:,0,:], dataPrepTrain['preprocessed_eeg_data'][:,:,4,:], dataPrepTrain['preprocessed_eeg_data'][:,:,12,:]]

        # True Data in Channels:
        trueDataTest = [test_dict['preprocessed_eeg_data'][:,:,test_dict['ch_names'].index('Pz'),:],
                        test_dict['preprocessed_eeg_data'][:,:,test_dict['ch_names'].index('Oz'),:],
                        test_dict['preprocessed_eeg_data'][:,:,test_dict['ch_names'].index('POz'),:]]

        trueDataTrain= [train_dict['preprocessed_eeg_data'][:,:,train_dict['ch_names'].index('Pz'),:],
                        train_dict['preprocessed_eeg_data'][:,:,train_dict['ch_names'].index('Oz'),:],
                        train_dict['preprocessed_eeg_data'][:,:,train_dict['ch_names'].index('POz'),:]]
        fig = plt.figure()
        for i in range(3):
            plt.plot( np.mean(np.mean(trueDataTest[i] - refDataTest[i],2),1),label=chanNames[i])
        plt.legend()
        plt.savefig("Test_Mean_SamplDiff")

        fig = plt.figure()
        for i in range(3):
            plt.plot( np.mean(np.mean(trueDataTrain[i] - refDataTrain[i],2),1),label=chanNames[i])
        plt.legend()
        plt.savefig("Train_Mean_SamplDiff")
        
    # create a PdfPages object
    pdf = PdfPages('ERPs.pdf')
    # define here the dimension of your figure

    # Retrieve the times array and calculate step for readable ticks
    times = train_dict["times"]
    step = len(times) // 10  # Adjust this as needed for readability
    ticks = np.arange(0, len(times), step)
    tick_labels = np.round(times[ticks], 2)  # Limit to 2 decimal places for clarity


    #### Plot ERPs - For test purposes, assuming enough repetitions
    for i in range(train_dict['preprocessed_eeg_data'].shape[2]):
        # Create a new figure
        fig = plt.figure()
        plt.title(train_dict['ch_names'][i])
        # Plot the data
        plt.plot(np.mean(np.mean(train_dict['preprocessed_eeg_data'][:, :,i, :], axis=1), axis=0))
        # Customize x-axis ticks and labels
        plt.xticks(ticks, tick_labels)
        plt.xlabel("Time (s)")
        # Save the current figure to the PDF
        pdf.savefig(fig)

        # Destroy the current figure to free up memory
        plt.close(fig)

    # Close the PdfPages object to ensure all plots are saved
    pdf.close()





def saveData(save_dir,test_dict,train_dict):
# Saving directories
    import os
    import numpy as np

    file_name_test = 'preprocessed_eeg_test.npy'
    file_name_train = 'preprocessed_eeg_training.npy'
    # Create the directory if not existing and save the data
    if os.path.isdir(save_dir) == False:
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, file_name_test), test_dict)

    # Create the directory if not existing and save the data
    if os.path.isdir(save_dir) == False:
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, file_name_train),train_dict)

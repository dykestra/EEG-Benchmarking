import json
import numpy as np
import pandas as pd

def common_average_reference(eeg):
    # Apply common average referencing to signal eeg: (N, C, T)
    return eeg - eeg.mean(axis=-2, keepdims=True)

def load_kuerp(root):
    print("Loading KU ERP...")
    kuerp_eeg = np.load(root +  '/KU_ERP/kuerp_data.npy', mmap_mode='r')
    kuerp_tf = pd.read_pickle(root +  '/KU_ERP/kuerp_trial_features.pd')

    trial_mask = (kuerp_tf['task'] == 'target') | (kuerp_tf['task'] == 'nontarget')
    kuerp_tf = kuerp_tf[trial_mask]
    kuerp_eeg = kuerp_eeg[trial_mask, :, :]

    kuerp_eeg = common_average_reference(kuerp_eeg)

    labels = kuerp_tf['task'].replace({'nontarget': 0, 'target': 1}).to_numpy()
    subject_ids = kuerp_tf['subject_id'].to_numpy()
    
    with open('splits/kuerp.txt') as f:
        splits = json.load(f)

    return kuerp_eeg, subject_ids, labels, splits

def load_physionetmi(root):
    print("Loading PhysionetMI...")
    physioeyes_eeg = np.load(root +  '/PhysionetMI/mmidb_data.npy', mmap_mode='r')
    physioeyes_tf = pd.read_pickle(root + '/PhysionetMI/mmidb_trial_features.pd')

    trial_mask = (physioeyes_tf['type'] == 'eye_open') | (physioeyes_tf['type'] == 'eye_closed')
    physioeyes_tf = physioeyes_tf[trial_mask]

    physioeyes_eeg = common_average_reference(physioeyes_eeg)

    labels = physioeyes_tf['type'].replace({'eye_closed': 0, 'eye_open': 1}).to_numpy()
    subject_ids = physioeyes_tf['subject_id'].to_numpy()
    
    with open('splits/eyes.txt') as f:
        splits = json.load(f)

    return physioeyes_eeg, subject_ids, labels, splits

def load_pavlov22(root):
    print("Loading Pavlov22...")
    pavlov_eeg = np.load(root + '/Pavlov22/pavlov2022_data.npy', mmap_mode='r')
    pavlov_tf = pd.read_pickle(root + '/Pavlov22/pavlov2022_trial_features.pd')

    trial_mask = (pavlov_tf['task'] == 'memory') | (pavlov_tf['task'] == 'control')
    trial_mask = trial_mask & (pavlov_tf['type'] == '13_digits')
    pavlov_tf = pavlov_tf[trial_mask]

    pavlov_eeg = common_average_reference(pavlov_eeg)

    labels = pavlov_tf['task'].replace({'control': 0, 'memory': 1}).to_numpy()
    subject_ids = pavlov_tf['subject_id'].to_numpy()

    with open('splits/pavlov.txt') as f:
        splits = json.load(f)

    return pavlov_eeg, subject_ids, labels, splits

def load_sleepedf(root):
    print("Loading Sleep EDF...")
    sleep_eeg = np.load(root + '/SleepEDF/SleepEDF_eeg_trials.npy', mmap_mode='r')
    sleep_tf = pd.read_pickle(root + '/SleepEDF/SleepEDF_trial_features.pd')

    sleep_eeg = common_average_reference(sleep_eeg)

    labels = sleep_tf['task'].to_numpy()
    subject_ids = sleep_tf['subject_id'].to_numpy()

    with open('splits/sleepedf.txt') as f:
        splits = json.load(f)

    return sleep_eeg, subject_ids, labels, splits

def load_highgamma(root):
    print("Loading High Gamma...")
    hgd_eeg = np.load(root +  '/HighGamma/highgamma_data.npy', mmap_mode='r')
    hgd_tf = pd.read_pickle(root +  '/HighGamma/highgamma_trial_features.pd')

    # mask out channels
    hgd_chnames = hgd_tf.attrs['channel_names']
    non_EEG = ['EOGh', 'EOGv', 'EMG_RH', 'EMG_LH', 'EMG_RF']
    other_EEG = ['AFF1', 'AFF2', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h', 'FCC5h', 'FCC3h', 'FCC4h', 'FCC6h',
        'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h', 'CPP5h', 'CPP3h', 'CPP4h', 'CPP6h', 'PPO1', 'PPO2', 'I1', 'I2', 
        'AFp3h', 'AFp4h', 'AFF5h', 'AFF6h', 'FFT7h', 'FFC1h', 'FFC2h', 'FFT8h', 'FTT7h', 'FCC1h', 'FCC2h', 'FTT8h', 
        'CCP1h', 'CCP2h', 'TTP8h', 'TPP7h', 'CPP1h', 'CPP2h', 'PPO9h', 'PPO5h', 'PPO6h', 'PPO10h', 'POO9h',
        'POO3h', 'POO4h', 'POO10h', 'OI1h', 'OI2h'] # i.e. electrodes not in standard 10-20
    hgd_chmask = np.invert(np.isin(hgd_chnames, non_EEG + other_EEG))
    hgd_eeg = hgd_eeg[:, hgd_chmask, :]

    hgd_eeg = common_average_reference(hgd_eeg)

    labels = hgd_tf['task'].replace({'no_action': 0, 'left_fist': 1, 'right_fist': 2, 'both_feet': 3}).to_numpy()
    subject_ids = hgd_tf['subject_id'].to_numpy()

    with open('splits/hgd.txt') as f:
        splits = json.load(f)

    return hgd_eeg, subject_ids, labels, splits

BENCHMARK_LOADERS = {
    "High Gamma": load_highgamma,
    "KU ERP": load_kuerp,
    "Pavlov memory": load_pavlov22,
    "Sleep EDF": load_sleepedf,
    "Physionet MI": load_physionetmi
}
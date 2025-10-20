# from neural_feature_identification.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
import h5py
import numpy as np
import random
from collections import defaultdict
from loguru import logger
from pathlib import Path

logger.remove()
# Configure logger. add() configures logger destinations (a.k.a. sinks).
# 'level' sets the min level for this sink; messages with lower levels (e.g., DEBUG) will be ignored
logger.add(lambda msg: print(msg, end=""), colorize=True, level="INFO")

def _create_trial_info(trial_list):
    """
    Internal helper to create trial info with RELATIVE start/stop indices
    :param trial_list:
    :return:
    """
    info_list = []
    current_offset = 0
    # sort trials by absolute idx to ensure chronological order
    sorted_trials = sorted(trial_list, key=lambda t: t['start'])
    for trial in sorted_trials:
        trial_length = trial['stop'] - trial['start']
        info_list.append({
            'gesture_id': trial['gesture_id'],
            'start_idx': trial['start'],
            'stop_idx': trial['stop'],
            'relative_start_idx': current_offset,
            'relative_stop_idx': current_offset + trial_length,
        })
        current_offset += trial_length
    return info_list

def load_session_kinematics_and_events(kinematics_path: Path, events_path: Path) -> dict:
    """
    Loads the kinematics and events data for a single session. Assumes HDF5 files
    :param kinematics_path: path to kinematics.h5
    :param events_path: path to events.h5
    :return: dictionary of session kinematics and events as numpy arrays
    """
    session_data = {}
    with h5py.File(kinematics_path, "r") as f:
        session_data['kinematics'] = f['kinematics'][:]
        session_data['timestamps'] = f['nip_time'][:]
    with h5py.File(events_path, "r") as f:
        session_data['trial_start_timestamps'] = f['trial_start_idxs'][:].astype(np.int64)
        session_data['trial_stop_timestamps'] = f['trial_stop_idxs'][:].astype(np.int64)
    return session_data

def generate_train_test_split(
        kinematics: np.ndarray,
        kinematics_timestamps: np.ndarray,
        trial_start_timestamps: np.ndarray,
        trial_stop_timestamps: np.ndarray,
        train_ratio: float = 0.7,
        kinematics_activation_threshold: float = 0.1,
        training_type: str = 'train_first',
        include_combined: bool=False,
) -> tuple[np.ndarray, np.ndarray, list[dict], list[dict]]:
    """
    Splits time series into training and test sets.
    The split ensures that all movements are represented proportionally in both sets.
    Default behavior ignores combined movements, and assigns older trials to the training set while assigning newer trials to the test set.
    :param kinematics: (NxD) array of kinematic data
    :param kinematics_timestamps: (N, ) array of NIP timestamps for the kinematic data
    :param trial_start_timestamps: Start timestamps for each trial
    :param trial_stop_timestamps: Stop timestamps for each trial
    :param train_ratio: proportion of trials for each gesture to use for training
    :param kinematics_activation_threshold: threshold to consider a kinematic channel "active"
    :param training_type: method for assigning training trials
           - 'train_first': (default) use the first trials for training
           - 'train_last': use the last trials for training
           - 'train_random': use the random trials for training
    :param include_combined: if True, include trials with multiple active DOFs. Defaults to False
    :return: tuple containing training and test set timestamps
    """
    logger.info('---Generating training and test sets---')
    if training_type not in ['train_first', 'train_last', 'train_random']:
        raise ValueError('Invalid training type')
    random.seed(2025)

    # Convert timestamps to array indices
    logger.info("Converting trial timestamps to array indices...")
    trial_start_idxs = np.searchsorted(kinematics_timestamps.flatten(), trial_start_timestamps.flatten(), side='left')
    trial_stop_idxs = np.searchsorted(kinematics_timestamps.flatten(), trial_stop_timestamps.flatten(), side='left')

    # Validate trial markers
    valid_trials = []
    num_starts, num_stops = len(trial_start_idxs), len(trial_stop_idxs)
    if num_starts > num_stops:
        logger.warning(f'Too many starts:\n\t{trial_start_idxs}\n\t{trial_stop_idxs}')
        trial_start_idxs = trial_start_idxs[:num_stops]
    elif num_starts < num_stops:
        logger.warning(f'Too many stops:\n\t{trial_start_idxs}\n\t{trial_stop_idxs}')
        trial_stop_idxs = trial_stop_idxs[:num_starts]
    for i in range(len(trial_start_idxs)):
        start_idx, stop_idx = trial_start_idxs[i], trial_stop_idxs[i]
        if start_idx < stop_idx < len(kinematics):
            valid_trials.append({'start': start_idx, 'stop': stop_idx, 'original_idx': i})
    logger.info(f"Found {len(valid_trials)} valid trials")

    # Classify trials by DOF/COMBO_ID and directionality
    classified_trials = []
    for trial in valid_trials:
        trial_kinematics = kinematics[trial['start']:trial['stop'], :]

        max_abs_activations = np.max(np.abs(trial_kinematics), axis=0) # find the max for each col
        active_dofs = np.where(max_abs_activations > kinematics_activation_threshold)[0] # index by zero to handle the tuple returned by np.where

        if len(active_dofs) == 1:
            trial['is_combined'] = False
            dominant_dof = active_dofs[0]
            peak_sample_idx = np.argmax(np.abs(trial_kinematics[:, dominant_dof]))
            peak_value = trial_kinematics[peak_sample_idx, dominant_dof]
            direction = "pos" if peak_value > 0 else "neg"
            trial['gesture_id'] = f"dof_{dominant_dof + 1}_{direction}"
            classified_trials.append(trial)
        elif len(active_dofs) > 1:
            trial['is_combined'] = True
            dof_details = []
            for dof in active_dofs:
                peak_sample_idx = np.argmax(np.abs(trial_kinematics[:, dof]))
                peak_value = trial_kinematics[peak_sample_idx, dof]
                direction = "pos" if peak_value > 0 else "neg"
                dof_details.append(f"dof_{dof + 1}_{direction}")
            # Sort the details to ensure a canonical gesture ID (e.g., 1_pos-2_neg is same as 2_neg-1_pos)
            dof_ids_str = sorted(dof_details)
            trial['gesture_id'] = f"combo_{'-'.join(dof_ids_str)}"
            classified_trials.append(trial)

    # Filter trials
    if not include_combined:
        processed_trials = [t for t in classified_trials if not t['is_combined']]
        logger.info(f"Ignoring combined movements. Processing {len(processed_trials)} trials...")
    else:
        processed_trials = classified_trials
        logger.info(f"Including combined movements. Processing {len(processed_trials)} trials...")

    # Group trials by gesture
    gesture_groups = defaultdict(list) # Elegant way to create an empty list to append to if the key doesn't exist.
    for trial in processed_trials:
        gesture_groups[trial['gesture_id']].append(trial)
    logger.info("Gesture representations:")
    for gesture, trials in gesture_groups.items():
        logger.info(f"- {gesture}: {len(trials)} trials")

    # Split trials for each group (Stratified split)
    train_trials_global_ref, test_trials_global_ref = [], [] # Store trials based on their absolute position in the file
    logger.info(f"Splitting with '{training_type}' config and {train_ratio} train ratio...")
    for gesture, trials in gesture_groups.items():
        if len(trials) < 2: continue
        if training_type == 'train_random':
            random.shuffle(trials)
        split_point = int(np.round(len(trials) * train_ratio))
        if training_type in ['train_first', 'train_random']:
            train_trials, test_trials = trials[:split_point], trials[split_point:]
        elif training_type == 'train_last':
            train_trials, test_trials =  trials[len(trials) - split_point:], trials[:len(trials) - split_point]
        train_trials_global_ref.extend(train_trials)
        test_trials_global_ref.extend(test_trials)
        logger.info(f"- {gesture}: {len(train_trials)} train, {len(test_trials)} test")

    train_trials_info = _create_trial_info(train_trials_global_ref)
    test_trials_info = _create_trial_info(test_trials_global_ref)

    # Stitch together the trials
    if not train_trials_info and not test_trials_info:
        logger.warning("No training trials found. Returning empty arrays")
        return np.array([]), np.array([]), [], []

    stitched_train_idxs = np.concatenate([
        np.arange(t['start_idx'], t['stop_idx']) for t in train_trials_info
    ])
    stitched_test_idxs = np.concatenate([
        np.arange(t['start_idx'], t['stop_idx']) for t in test_trials_info
    ])
    logger.info("---Split Complete---")
    return stitched_train_idxs, stitched_test_idxs, train_trials_info, test_trials_info
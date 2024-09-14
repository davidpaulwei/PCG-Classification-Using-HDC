import wave
import numpy as np
import glob

# This python file is designed to read and preprocess signals stored in wav files,

def from_wav_to_array(file: str):
    """read audio data from wav. file;
    return time-domain signal (ndarray) and sampling rate (int)."""
    with wave.open(file, "rb") as f:
        frame_rate, num_frames = f.getparams()[2:4]
        str_data = f.readframes(num_frames) # obtain signal in strings.
    signal = np.frombuffer(str_data, dtype=np.short) # convert string signal into ndarray type.
    return signal, frame_rate

def load_data(num_trains: int = 800, num_tests: int = 200):
    """read labeled audio datas from wav. files, extract learnable features from the datas, 
    randomly shuffle them and sort them into training and testing sets for machine learning processes.
    return training features list, training labels list, testing features list, testing labels list."""
    class_labels = ["N", "AS", "MR", "MS", "MVP"] # labels of different classes. ('N' stands for normal heart sounds, while others are the name of diseases the source of the heart suffers)
    features, labels = [], []
    for label_idx, class_label in enumerate(class_labels):
        files = glob.glob(class_label + "_HSS/*.wav")
        for file in files:
            signal, _ = from_wav_to_array(file) # read time-domain signal and sampling rate from file.
            # feature = mfcc(signal, sample_rate) # transfer time-domain signal into learnable features.
            if len(signal) <= 20000:
                features.append(np.concatenate((signal, np.zeros(20000 - len(signal)))))
            else:
                features.append(signal[:20000])
            labels.append(label_idx)
    order = np.arange(num_trains + num_tests)
    np.random.shuffle(order) # generate random order.
    train_features = [features[idx] for idx in order[:num_trains]]
    train_labels = [labels[idx] for idx in order[:num_trains]]
    test_features = [features[idx] for idx in order[num_trains:num_trains+num_tests]]
    test_labels = [labels[idx] for idx in order[num_trains:num_trains+num_tests]]
    return train_features, train_labels, test_features, test_labels

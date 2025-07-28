import numpy as np
import pandas as pd

from torchic import Dataset
from torchic.physics.ITS import unpack_cluster_sizes, average_cluster_size

from sklearn.preprocessing import LabelEncoder

import sys
sys.path.append('..')
from core.nn_common import downsample_class, torch_data_preparation, NNRoutine
from core.pid_fcnn import PidFCNN

# constants

BATCH_SIZE = 128
NUMBER_EPOCHS = 20
LEARNING_RATE = 1e-3
# -------------


def data_preparation(df: pd.DataFrame):
    """
    Prepares the data for training by separating features and target labels.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.

    Returns:
    X (np.ndarray): Feature data.
    y (np.ndarray): Target labels.
    """
    
    feature_columns = ['fItsClusterSizeL0', 'fItsClusterSizeL1', 'fItsClusterSizeL2', 
                       'fItsClusterSizeL3', 'fItsClusterSizeL4', 'fItsClusterSizeL5',
                        'fItsClusterSizeL6', 'fEta', 'fPhi', 'fCosL', 'fPAbs',
                        'fMeanItsClSize', 'fClSizeCosL']
    X = df[feature_columns].values.astype('float32')

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['fPartID'].values)

    return X, y, label_encoder


def load_data() -> Dataset:

    dataset = Dataset.from_root(['/data/galucia/its_pid/pass7/LHC22o_pass7_minBias_small.root', 
                                 '/data/galucia/its_pid/pass7/LHC22o_pass7_minBias_longK.root'],
                                tree_name='O2clsttable',
                                folder_name='DF*',
                                columns=['fP', 'fEta', 'fPhi', 'fItsClusterSize', 'fPartID'])
    dataset.concat(Dataset.from_root(['/data/galucia/its_pid/LHC23_pass4_skimmed/LHC23_pass4_skimmed.root'],
                                     tree_name='O2clsttableextra',
                                     folder_name='DF*',
                                     columns=['fP', 'fEta', 'fPhi', 'fItsClusterSize', 'fPartID']) )
    
    np_unpack_cluster_sizes = np.vectorize(unpack_cluster_sizes)
    for layer in range(7):
        dataset[f'fItsClusterSizeL{layer}'] = np_unpack_cluster_sizes(dataset['fItsClusterSize'], layer)
    dataset['fCosL'] = 1 / np.cosh(dataset['fEta'])
    dataset['fPAbs'] = np.abs(dataset['fP'])
    dataset['fClSizeCosL'], dataset['fNHitsIts'] = average_cluster_size(dataset['fItsClusterSize'])
    dataset['fMeanItsClSize'] = dataset['fClSizeCosL'] / dataset['fCosL']
    dataset.query('fNHitsIts == 7', inplace=True)

    return dataset

if __name__ == "__main__":

    dataset = load_data()
    df = dataset.data    
    #df = downsample_class(df, target_class=2)  # Downsample pions
    df.sample(frac=0.1, random_state=42)
    print(f"Number of samples: {len(df)}")

    X, y, label_encoder = data_preparation(df)
    train_loader, validation_loader, test_loader = torch_data_preparation(X, y,
                                                                          test_val_size=0.5,
                                                                          batch_size=BATCH_SIZE)

    input_dim = X.shape[1]
    num_classes = len(label_encoder.classes_)
    model = PidFCNN(input_dim, num_classes)

    nn_routine = NNRoutine(model)
    nn_routine.run_training_loop(train_loader, LEARNING_RATE, NUMBER_EPOCHS)
    nn_routine.plot_loss(outfile='../output/nn/training_loss.pdf')
    accuracy = nn_routine.run_model_eval(validation_loader)

    print(f"Accuracy: {accuracy:.2f}%")

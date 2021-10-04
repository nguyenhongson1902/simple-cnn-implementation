from tensorflow.keras import datasets
import tensorflow_datasets as tfds
tfds.disable_progress_bar()


def load_and_get_data():
    '''
        Output: All are arrays
            train_images: Training data
            train_labels: Labels of training data
            test_images: Test data
            test_labels: Labels of test data
    '''
    #  LOAD AND SPLIT DATASET
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    return train_images, train_labels, test_images, test_labels

def load_and_split_data():
    # split the data manually into 80% training, 10% testing, 10% validation
    (raw_train, raw_validation, raw_test), metadata = tfds.load(
                            'cats_vs_dogs',
                            split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
                            with_info=True,
                            as_supervised=True,)

    return raw_train, raw_validation, raw_test, metadata
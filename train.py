from contextlib import redirect_stdout
import tensorflow as tf
from models import my_model
from datasets import load_and_get_data, load_and_split_data
import argparse


IMG_SIZE = 160 # All images will be resized to 160x160

def format_example(image, label):
    """
    returns an image that is reshaped to IMG_SIZE
    """
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1 # I don't know this method :(
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

def evaluate_model(model, test_images, test_labels):
    '''
        Input:
            model: The model to be evaluated
            test_images: Test data. An array
            test_labels: Labels of test data. An array
        
    '''
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('Test accuracy: ', test_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Choose a new model or pretrained model to use')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-p','--pretrained', action='store_true', help='Choose a pretrained mode or not')
    args = parser.parse_args()

    if args.pretrained:
        raw_train, raw_validation, raw_test, metadata = load_and_split_data()
        get_label_name = metadata.features['label'].int2str  # creates a function object that we can use to get labels

        # Apply format_example function to all images using .map()
        train = raw_train.map(format_example)
        validation = raw_validation.map(format_example)
        test = raw_test.map(format_example)

        # Shuffle and batch the images
        BATCH_SIZE = 32
        SHUFFLE_BUFFER_SIZE = 1000

        train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
        validation_batches = validation.batch(BATCH_SIZE)
        test_batches = test.batch(BATCH_SIZE)

        # Picking a pretrained model: MobileNetV2
        IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

        # Create the base model from the pre-trained model MobileNet V2
        base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, 
                                            include_top=False, # to not include classification part of the original model
                                            weights='imagenet') # pretrained weights from imagenet
        with open('./summary/base_model_summary_before_freezing.txt', 'w') as f:
            with redirect_stdout(f):
                base_model.summary()

        # Freezing the base model
        base_model.trainable = False

        with open('./summary/base_model_summary_after_freezing.txt', 'w') as f:
            with redirect_stdout(f):
                base_model.summary()
        

        # Adding a classifier
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = tf.keras.layers.Dense(1) # only for binary classification
        

        # Combine the base model and the classifier
        model = tf.keras.Sequential([base_model, global_average_layer, prediction_layer])

        with open('./summary/base_model_summary_after_adding_a_classifier.txt', 'w') as f:
            with redirect_stdout(f):
                model.summary()
        
        # Training the Model
        base_learning_rate = 0.0001
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate),
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        
        # We can evaluate the model right now to see how it does before training it on our new images
        validation_steps=20
        loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)
        print('Before training: loss={}, accuracy={}'.format(loss0, accuracy0))

        # Now we can train it on our images
        initial_epochs = 3
        history = model.fit(train_batches,
                            epochs=initial_epochs,
                            validation_data=validation_batches)
        acc = history.history['accuracy']
        print('After training: accuracy=', acc)
        
        # Saving model
        print('Saving model... ', end='')
        model.save("./models/dogs_vs_cats.h5")  # we can save the model and reload it at anytime in the future
        print('Succeed!')

        # Loading model
        print('Loading model... ', end='')
        new_model = tf.keras.models.load_model('./models/dogs_vs_cats.h5')
        print('Succeed!')


    else:
        # Preparing data
        train_images, train_labels, test_images, test_labels = load_and_get_data()

        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck']

        model = my_model(input_shape=(32, 32, 3)) # CIFAR 10
        with open('./summary/my_model_summary.txt', 'w') as f:
            with redirect_stdout(f):
                model.summary()

        # Training model
        model.compile(optimizer='adam', 
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                    metrics=['accuracy'])
        
        history = model.fit(train_images, train_labels, epochs=4, validation_data=(test_images, test_labels))
        print('History: ', history)

        print('Evaluating the Model')
        evaluate_model(model=model, test_images=test_images, test_labels=test_labels) # print test accuracy

        # Saving model
        print('Saving model... ', end='')
        model.save("./models/my_model.h5")  # we can save the model and reload it at anytime in the future
        print('Succeed!')

        # Loading model
        print('Loading model... ', end='')
        new_model = tf.keras.models.load_model('./models/my_model.h5')
        print('Succeed!')





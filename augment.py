import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow_datasets.core.load import load
from datasets import load_and_get_data

# Get data
train_images, train_labels, test_images, test_labels = load_and_get_data()

# creates a data generator object that transforms images
datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                            shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

# pick an image to transform
test_img = train_images[20] # change this
img = image.img_to_array(test_img)  # convert image to numpy array
img = img.reshape((1,) + img.shape)  # reshape image

i = 0

for batch in datagen.flow(img, save_to_dir='./data/train_augmented', save_prefix='test', save_format='jpeg'):  # this loops runs forever until we break, saving images to data/train_augmented with specified prefix
    plt.figure(i)
    plot = plt.imshow(image.img_to_array(batch[0]))
    i += 1
    if i > 4:  # show 4 images
        break

plt.show()


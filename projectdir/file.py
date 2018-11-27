import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from PIL import *
from PIL import Image
import numpy as np
import keras
from keras import backend as K
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy



from keras.models import Model
from keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

from keras import backend as K
K.set_image_dim_ordering('th')



# In[2]:


datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


# In[3]:


img = load_img('C:/Users/Indira/dataset/dataset/TRAIN_DIR/dress/dress35.jpg.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='C:/Users/Indira/dataset/dataset/preview', save_prefix='dress', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely


# In[4]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 150, 150)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# the model so far outputs 3D feature maps (height, width, features)


# In[5]:


model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# In[6]:


#path='C:/Users/Indira/dataset/dataset/'


batch_size = 128

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'C:/Users/Indira/dataset/dataset/TRAIN_DIR',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'C:/Users/Indira/dataset/dataset/TEST_DIR',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')


# In[7]:


model.fit_generator(
        train_generator,
        steps_per_epoch=500 // batch_size,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=100 // batch_size)
model.save_weights('weights.h5')  # always save your weights after training or during training


# In[8]:


model.save('C:/Users/Indira/dataset/dataset/weights.h5')


# In[10]:


from keras.models import load_model



# In[11]:


model=load_model('C:/Users/Indira/dataset/dataset/weights.h5')


# In[12]:


#path='C:/Users/Indira/dataset/dataset/'


import numpy as np
from keras.preprocessing import image
test_image=image.load_img('C:/Users/Indira/dataset/dataset/TEST_DIR/coat/coat73.jpg.jpg', target_size=(150,150))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image, axis=0)
result=model.predict(test_image)
train_generator.class_indices
if result[0][0] >= 0.5:
        prediction='dress'
else:
        prediction='coat'
print(prediction)


# In[13]:


print(model.summary())


# In[36]:


# path to the model weights files.
weights_path = '../keras/examples/vgg16_weights.h5'
top_model_weights_path = 'fc_model.h5'
# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'C:/Users/Indira/dataset/dataset/TRAIN_DIR'
validation_data_dir = 'C:/Users/Indira/dataset/dataset/TEST_DIR'
nb_train_samples = 140
nb_validation_samples = 60
epochs = 20
batch_size = 20


# In[30]:


from keras import applications


# In[39]:


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
model = applications.VGG16(include_top=False, weights='imagenet')

generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
np.save(open('bottleneck_features_train.npy', 'wb'),
            bottleneck_features_train)


# In[40]:


generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
np.save(open('bottleneck_features_validation.npy', 'wb'),
            bottleneck_features_validation)


# In[ ]:


#np.save(open(file_name, 'wb'), saved)
#loaded = np.load(open(file_name,'rb'))


# In[43]:


train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
train_labels = np.array(
        [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
validation_labels = np.array(
        [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))


model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy', metrics=['accuracy'])


# In[44]:


model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
model.save_weights(top_model_weights_path)


# In[46]:


save_bottlebeck_features()


# In[47]:


# path to the model weights files.
weights_path = '../keras/examples/vgg16_weights.h5'
top_model_weights_path = 'fc_model.h5'
# dimensions of our images.
img_width, img_height = 150, 150


# In[96]:


# build the VGG16 network
model = applications.VGG16(weights='imagenet', include_top=False, input_shape = (3,150,150))
print('Model loaded.')


# In[98]:


input_shape = input_shape=(3, 150, 150)


# In[86]:


from keras import backend as K
K.set_image_dim_ordering('th')


# In[95]:


print(keras.__version__)



# In[104]:


from keras.models import Model



# In[109]:


# build a classifier model to put on top of the convolutional model

top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))



# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
#model.add(top_model)


model = Model(input= top_model.input, output= top_model(top_model.output))

#model = Model(input=model.input, output=top_model(model.output))


# In[64]:


from keras import optimizers


# In[92]:


# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:25]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])


# In[93]:


# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')


# In[94]:


# fine-tune the model
#model.fit_generator(
 #   train_generator,
  #  samples_per_epoch=nb_train_samples,
   # epochs=epochs,
    #validation_data=validation_generator,
    #nb_val_samples=nb_validation_samples)

#new_model.fit_generator(
#train_generator,
#steps_per_epoch=nb_train_samples//batch_size,
#epochs=epochs,
#validation_data=validation_generator,
#nb_val_samples=nb_validation_samples//batch_size)`


model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)


# In[ ]:

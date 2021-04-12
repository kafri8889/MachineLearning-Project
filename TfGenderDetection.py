import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset
# https://drive.google.com/file/d/11jFQkkGiHtTvHF4sepNml-IVvtIM4MPq/view?usp=sharing

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Training Pake CPU

def plot_image(mdl, imejch):
    plt.imshow(imejch)

    x = image.img_to_array(imejch)
    x = np.expand_dims(x, axis=0)
    imegs = np.vstack([x])
    val = mdl.predict(imegs)
    if val == 0:
        print('Cewe')
        plt.xlabel("Cewe")
    else:
        print("Cowo"),
        plt.xlabel("Cowo")

    plt.show()


data_gen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    fill_mode='nearest'
)

train_generator = data_gen.flow_from_directory(
    # Ubah Path Ke Directory Dataset Train
    os.path.abspath("C:/Users/kk/PycharmProjects/mlproject/zip_extracted/gender_detection/train"),
    target_size=(150, 150),
    batch_size=8,
    class_mode='binary'
)

validation_generator = data_gen.flow_from_directory(
    # Ubah Path Ke Directory Dataset Validation
    os.path.abspath("C:/Users/kk/PycharmProjects/mlproject/zip_extracted/gender_detection/validation"),
    target_size=(150, 150),
    batch_size=8,
    class_mode='binary'
)

print(validation_generator.class_indices)

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), strides=(1, 1), input_shape=(150, 150, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

''' Ubah Pathnya '''
''' Kalo Make Tensorboard '''
# log_dir = os.path.abspath("/all_logs/tf_genderdetection")
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=22,
    steps_per_epoch=len(train_generator),
    validation_steps=len(validation_generator),
    # callbacks=[tensorboard_callback],  # Tensorboard Callback
    verbose=1
)

''' Ubah Pathnya '''
# model.save(os.path.abspath('C:/Users/kk/PycharmProjects/mlproject'
#                            '/saved_tf_model/tf_detectgender_model.h5'))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

''' Ubah Pathnya '''
# model = tf.keras.models.load_model(os.path.abspath('C:/Users/kk/PycharmProjects/mlproject'
#                                                    '/saved_tf_model/tf_detectgender_model.h5'))

# img = image.load_img(os.path.abspath("C:/Users/kk/PycharmProjects/mlproject/img/jeni.png"), target_size=(150, 150))

# plot_image(model, img)  # Prediksi Foto

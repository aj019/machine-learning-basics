import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

#x_train: A NumPy array of 50,000 color training images, each with a size of 32x32 pixels and 3 color channels (RGB).
#y_train: A NumPy array of corresponding labels for the training images. The labels are integers from 0 to 9, representing the class of the image.
#x_test: A NumPy array of 10,000 color test images (32x32 pixels, 3 channels).
#y_test: A NumPy array of corresponding labels for the test images. 
#The pixel values are initially in the range of unsigned integers [0, 255]. A common preprocessing step is to normalize these values to a range like [0, 1] or [-1, 1] before training a model. 

# first pixel value of x_train
#print(x_train[0][0][0])

x_train, x_test = x_train / 255.0, x_test / 255.0
num_classes = 10
# Convert the labels to one-hot encoded vectors which means
# Converting integer into a vector of zeros except the index of the class which is 1.
# 3 becomes [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test  = tf.keras.utils.to_categorical(y_test , num_classes)

model = models.Sequential([
    
    layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.Flatten(),

    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    epochs=15,
                    batch_size=64,
                    validation_split=0.2,
                    verbose=2)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

print(f"Test accuracy = {test_acc:.3f}")

plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend()
plt.title('Accuracy')
plt.show()
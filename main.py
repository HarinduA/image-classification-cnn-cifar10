# main.py
import tensorflow as tf
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalize the images to values between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Class names corresponding to CIFAR-10 labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Display an example image
plt.imshow(train_images[0])
plt.title(class_names[train_labels[0][0]])  # Show class name
plt.show()

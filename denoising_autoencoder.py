import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers
from matplotlib.animation import FuncAnimation

# Load MNIST data
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), -1))
x_test = x_test.reshape((len(x_test), -1))

# Add noise
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Autoencoder architecture
input_dim = x_train.shape[1]
encoding_dim = 64

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(1e-5))(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train
autoencoder.fit(x_train_noisy, x_train,
                epochs=20,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test_noisy, x_test),
                verbose=1)

# Reconstruct test images
decoded_imgs = autoencoder.predict(x_test_noisy)

# Create GIF comparing noisy vs reconstructed
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
plt.tight_layout()


def update(frame):
    for i in range(5):
        axes[0, i].clear()
        axes[0, i].imshow(x_test_noisy[frame + i].reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title("Noisy")

        axes[1, i].clear()
        axes[1, i].imshow(decoded_imgs[frame + i].reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title("Reconstructed")
    return axes


ani = FuncAnimation(fig, update, frames=20, interval=400)
ani.save("denoising_autoencoder.gif", writer="pillow", fps=2)
plt.close()

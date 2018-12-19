import keras
import numpy as np
from tqdm import tqdm
import cv2




# convenience
def set_trainable_flag(model: keras.Model, flag: bool):
    model.trainable = flag
    for layer in model.layers:
        layer.trainable = flag



##############
### models ###
##############

# discriminator
D = keras.models.Sequential()
D.add(keras.layers.Conv2D(64, (3, 3), padding="same", input_shape=(32, 32, 3)))
D.add(keras.layers.LeakyReLU())

D.add(keras.layers.Conv2D(64, (3, 3), padding="same"))
D.add(keras.layers.MaxPool2D(pool_size=(2, 2), padding="same"))
D.add(keras.layers.LeakyReLU())

D.add(keras.layers.Conv2D(128, (3, 3), padding="same"))
D.add(keras.layers.LeakyReLU())

D.add(keras.layers.Conv2D(128, (3, 3), padding="same"))
D.add(keras.layers.MaxPool2D(pool_size=(2, 2), padding="same"))
D.add(keras.layers.LeakyReLU())

D.add(keras.layers.Conv2D(256, (3, 3), padding="same"))
D.add(keras.layers.LeakyReLU())

D.add(keras.layers.Flatten())

D.add(keras.layers.Dense(100))
D.add(keras.layers.LeakyReLU())

D.add(keras.layers.Dense(1))
D.add(keras.layers.Activation("sigmoid"))



# generator
G = keras.models.Sequential()
G.add(keras.layers.Dense(128*8*8, input_shape=(100,)))
G.add(keras.layers.BatchNormalization())
G.add(keras.layers.Reshape([8, 8, 128]))

G.add(keras.layers.Conv2D(128, (3, 3), padding="same"))
G.add(keras.layers.BatchNormalization())
G.add(keras.layers.LeakyReLU())

G.add(keras.layers.Conv2D(128, (3, 3), padding="same"))
G.add(keras.layers.BatchNormalization())
G.add(keras.layers.LeakyReLU())

G.add(keras.layers.UpSampling2D(size=(2, 2)))

G.add(keras.layers.Conv2D(64, (3, 3), padding="same"))
G.add(keras.layers.BatchNormalization())
G.add(keras.layers.LeakyReLU())

G.add(keras.layers.Conv2D(64, (3, 3), padding="same"))
G.add(keras.layers.BatchNormalization())
G.add(keras.layers.LeakyReLU())

G.add(keras.layers.UpSampling2D(size=(2, 2)))

G.add(keras.layers.Conv2D(32, (3, 3), padding="same"))
G.add(keras.layers.BatchNormalization())
G.add(keras.layers.LeakyReLU())

G.add(keras.layers.Conv2D(32, (3, 3), padding="same"))
G.add(keras.layers.BatchNormalization())
G.add(keras.layers.LeakyReLU())

G.add(keras.layers.Conv2D(3, (3, 3), padding="same"))
G.add(keras.layers.BatchNormalization())
G.add(keras.layers.Activation("sigmoid"))



# GAN
GAN = keras.models.Sequential()
GAN.add(G)
GAN.add(D)
G.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(lr=1e-4))
D.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(lr=1e-3))
GAN.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(lr=1e-4))


# data
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()


# training
num_samples = x_train.shape[0]
batch_size = 64
epochs = 10


for e in range(1, epochs+1):
    print(f"epoch {e}")
    X = x_train.copy()
    np.random.shuffle(X)

    idx_iter = (i for i in range(0, num_samples, batch_size))

    for run in tqdm(range(int(num_samples/batch_size))):
        idx = next(idx_iter)
        batch_real = X[idx:idx+batch_size]

        # D trainable
        set_trainable_flag(D, True)

        # train D - real
        D.train_on_batch(batch_real, np.ones(shape=(batch_real.shape[0], 1)))

        # train D - fake
        z = np.random.uniform(0.0, 1.0, size=(batch_real.shape[0], 100))
        generated = G.predict(z)
        D.train_on_batch(generated, np.zeros(shape=(batch_real.shape[0], 1)))

        # train G
        set_trainable_flag(D, False)
        z = np.random.uniform(0.0, 1.0, size=(batch_real.shape[0], 100))
        GAN.train_on_batch(z, np.ones(shape=(batch_real.shape[0], 1)))

    # log
    z = np.random.uniform(0.0, 1.0, size=(5, 100))
    sample = G.predict(z)
    sample = np.hstack([*sample])
    cv2.imwrite(f"{e}.png", sample)



















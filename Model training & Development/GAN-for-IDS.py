# ========================================
# GAN-Based Data Augmentation for CICIDS2017
# ========================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models

# Load preprocessed dataset (cleaned and encoded)
df = pd.read_csv("cicids2017_preprocessed.csv")  # Replace with your actual file
attack_data = df[df['Label'] != 'BENIGN'].drop(columns=['Label'])

# Normalize the data
scaler = MinMaxScaler()
attack_scaled = scaler.fit_transform(attack_data)

latent_dim = 100
data_dim = attack_scaled.shape[1]

# Generator
def build_generator():
    model = models.Sequential([
        layers.Dense(128, input_dim=latent_dim, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(data_dim, activation='tanh')
    ])
    return model

# Discriminator
def build_discriminator():
    model = models.Sequential([
        layers.Dense(256, input_dim=data_dim, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Build and compile models
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# GAN setup
discriminator.trainable = False
gan_input = layers.Input(shape=(latent_dim,))
generated_data = generator(gan_input)
gan_output = discriminator(generated_data)
gan = models.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Training loop
epochs = 10000
batch_size = 64
real_labels = np.ones((batch_size, 1))
fake_labels = np.zeros((batch_size, 1))

for epoch in range(epochs):
    # Train discriminator
    idx = np.random.randint(0, attack_scaled.shape[0], batch_size)
    real_data = attack_scaled[idx]
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake_data = generator.predict(noise)

    d_loss_real = discriminator.train_on_batch(real_data, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_data, fake_labels)

    # Train generator via GAN
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = gan.train_on_batch(noise, real_labels)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch} | D Loss: {0.5 * np.add(d_loss_real, d_loss_fake)} | G Loss: {g_loss}")

# Generate synthetic attack data
synthetic_samples = generator.predict(np.random.normal(0, 1, (10000, latent_dim)))
synthetic_samples = scaler.inverse_transform(synthetic_samples)
synthetic_df = pd.DataFrame(synthetic_samples, columns=attack_data.columns)
synthetic_df["Label"] = "SYNTHETIC_ATTACK"

# Save for later use in training your IDS
synthetic_df.to_csv("synthetic_attack_data.csv", index=False)
print("Synthetic data saved as synthetic_attack_data.csv")

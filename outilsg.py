import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras import Model
import matplotlib.pyplot as plt

def discriminator():
    model = models.Sequential([
        # Bloc 1
        layers.Conv2D(32, (3,3), padding='same', input_shape=(225,225,3), 
                      kernel_regularizer=regularizers.l2(0.0005)),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.3),
        
        # Bloc 2
        layers.Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.0005)),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        #layers.Dropout(0.3),
        
        # Bloc 3
        layers.Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.0005)),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.4),
        
        # Bloc 4
        layers.Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.0005)),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.4),

        
        layers.Flatten(),
        # layers.Dense(512, kernel_regularizer=regularizers.l2(0.0005)),
        # layers.LeakyReLU(alpha=0.2),
        # layers.Dropout(0.5),

        # layers.Dense(256, kernel_regularizer=regularizers.l2(0.0005)),
        # layers.LeakyReLU(alpha=0.2),
        # layers.Dropout(0.5),

        layers.Dense(1, activation='sigmoid')  # Sortie binaire (vrai/faux)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model





def generator():
    model = models.Sequential([
        # Dense -> Reshape initial (Base de 15x15 pour ajuster naturellement à 225x225)
        layers.Dense(256 * 15 * 15, activation='relu', input_shape=(128,)),
        layers.Reshape((15, 15, 256)),
        layers.BatchNormalization(),

        # Upsampling 30x30
        layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        # Upsampling 60x60
        layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        # Upsampling 120x120
        layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        # Upsampling 240x240 (trop grand, on doit ajuster !)
        layers.Conv2DTranspose(32, (4,4), strides=(2,2), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        # Dernière couche pour atteindre 225x225
        layers.Conv2DTranspose(3, (3,3), activation='tanh', padding='same'),
        layers.Cropping2D(((7, 8), (7, 8)))  # Ajustement précis à 225x225
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5),
                  loss='hinge')
    return model



class GAN(Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, disc_opt, gen_opt, loss_function):
        super().compile()
        self.disc_opt = disc_opt
        self.gen_opt = gen_opt
        self.loss_function = loss_function
        self.disc_loss_metric = tf.keras.metrics.Mean(name="disc_loss")
        self.gen_loss_metric = tf.keras.metrics.Mean(name="gen_loss")

    @property
    def metrics(self):
        return [self.disc_loss_metric, self.gen_loss_metric]

    def call(self, inputs, training=False):
        # Forward pass for GAN: use the generator for generating images
        random_latent_vectors = inputs  # Assume inputs are latent vectors
        generated_images = self.generator(random_latent_vectors, training=training)
        return generated_images



    def train_step(self, data):
        real_images, _ = data  # Ignore labels if present
        batch_size = tf.shape(real_images)[0]
    
        # ======== Ajout de bruit aléatoire aux images réelles ========
        noise_factor = 0.1  # Ajuste l'intensité du bruit
        noisy_real_images = real_images + noise_factor * tf.random.normal(shape=tf.shape(real_images))
    
        # S'assurer que les valeurs restent entre -1 et 1 (si normalisation [-1,1])
        noisy_real_images = tf.clip_by_value(noisy_real_images, -1.0, 1.0)
    
        # ======== Génération d'images fictives ========
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))  # Latent vectors as input
        generated_images = self.generator(random_latent_vectors)  # Generator takes latent vectors
    
        # Concaténation des images réelles bruitées et générées
        combined_images = tf.concat([generated_images, noisy_real_images], axis=0)
    
        # Création des labels (1 pour les vrais, 0 pour les faux)
        labels = tf.concat([tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))], axis=0)
    
        # Ajout d'un léger bruit aux labels (lissage)
        labels += 0.05 * tf.random.uniform(tf.shape(labels))
    
        # ======== Mise à jour du discriminateur ========
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            disc_loss = self.loss_function(labels, predictions)
    
        grads = tape.gradient(disc_loss, self.discriminator.trainable_weights)
        self.disc_opt.apply_gradients(zip(grads, self.discriminator.trainable_weights))
    
        # ======== Mise à jour du générateur ========
        misleading_labels = tf.ones((batch_size, 1))  # On veut tromper le discriminateur
    
        with tf.GradientTape() as tape:
            fake_predictions = self.discriminator(self.generator(random_latent_vectors))  # Pass latent vectors
            gen_loss = self.loss_function(misleading_labels, fake_predictions)
    
        grads = tape.gradient(gen_loss, self.generator.trainable_weights)
        self.gen_opt.apply_gradients(zip(grads, self.generator.trainable_weights))
    
        # Mise à jour des métriques
        self.disc_loss_metric.update_state(disc_loss)
        self.gen_loss_metric.update_state(gen_loss)
    
        return {
            "disc_loss": self.disc_loss_metric.result(),
            "gen_loss": self.gen_loss_metric.result(),
        }

    
   



def gen_images(generator, latent_dim, current_epoch):
    noise = tf.random.normal([2, latent_dim])  # Génération de bruit aléatoire
    generated_images = generator(noise, training=False)

    # Remettre les images dans l'intervalle [0,1] pour un affichage correct
    generated_images = (generated_images * 0.5) + 0.5  

    num_of_sample = generated_images.shape[0]
    
    plt.figure(figsize=(10, 5))
    for i in range(num_of_sample):
        plt.subplot(1, num_of_sample, i + 1)  # Affichage sur une ligne
        plt.imshow(generated_images[i]*225)  # Suppression de `[:,:,0]`
        plt.title(f"After epoch {current_epoch}")        
        plt.axis('off')

    plt.savefig(f'After_epoch_{current_epoch:04d}.png')
    plt.show()

class GAN_Callback(tf.keras.callbacks.Callback):
    def __init__(self, num_images=2, latent_dim=128):
        self.num_images = num_images
        self.latent_dim = latent_dim       
    
    def on_epoch_end(self, epoch, logs=None):
        latent_vectors = tf.random.normal(shape=(self.num_images, self.latent_dim))
        generated_images = self.model.generator(latent_vectors, training=False)

        # Remettre les images dans l'intervalle [0,225] pour un affichage correct
        generated_images = (generated_images) 
        generated_images = generated_images.numpy()

        # Affichage des images générées
        plt.figure(figsize=(16, 5))
        for i in range(self.num_images):
            plt.subplot(1, self.num_images, i + 1)
            plt.imshow((generated_images[i] + 1)/ 2)  # Suppression de `[:, :, 0]`
            plt.title(f"After epoch {epoch+1}")        
            plt.axis('off')

        plt.savefig(f'After_epoch_{epoch+1:04d}.png')
        plt.show()

        # Sauvegarde du modèle toutes les 10 époques avec un nom unique
        if (epoch + 1) % 10 == 0:
            self.model.generator.save(f'gen_epoch_{epoch+1:04d}.keras')
            self.model.discriminator.save(f'disc_epoch_{epoch+1:04d}.keras')



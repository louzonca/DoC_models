from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
     Dense, Reshape, Conv2DTranspose, Activation, Flatten
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import Reduction
from tensorflow.keras.optimizers import SGD, Adadelta, Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import RandomNormal
import os
import pickle
import numpy as np


class AutoEncoderDense:
    #  Class AutoEncoderDens builds the encoder & decoder with only fully connected layers
    # and provides methods to train it and save the trained results
    # it also provides access to the latent space results

    def __init__(self,
                 input_size,
                 layers_dim,
                 latent_space_dim):
        self.input_size = input_size  # size of the input data for one patient i.e. [num_ROIs, num_time_steps, num_channels=1]
        self.layers_dim = layers_dim  # dim of the output for each layer e.g. [128, 64, 32]
        self.latent_space_dim = latent_space_dim  # latent space dimension that we wish to obtain (e.g. loop between 1 and 15)

        self.encoder = None
        self.decoder = None
        self.auto_encoder = None

        self._num_layers = len(layers_dim)  # Gives the number of conv layers e.g. 3 layers
        self._encoder_output_shape = None
        self._input_model = None

        self._build()  # Method that will build the full autoencoder

    # Here we define all the PUBLIC methods of the AE class (i.e. that can be called from outside):
    # public methods include summary, train and save of the AE AND predict AND get_latent_space_output

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.auto_encoder.summary()

    def evaluate(self, data_output, data_target):
        self.auto_encoder.evaluate(
            data_output,
            data_target
        )

    def train(self, train_data, valid_data, learning_rate=0.00001, batch_siz=32, num_epochs=20):
        # First compile the model for training
        opt = Adam(learning_rate=learning_rate)
            #SGD(learning_rate=learning_rate, momentum=0.0, nesterov=False, name="SGD")
            #Adadelta(learning_rate=learning_rate, rho=0.95, epsilon=1e-07)#
        loss_fun = MeanSquaredError()
        metrics = ['MeanSquaredError', 'accuracy']
        self.auto_encoder.compile(opt, loss_fun, metrics)
        # Then fit the model to the training set
        self.auto_encoder.fit(
            train_data,  # input for training
            train_data,  # target for training
            epochs=num_epochs,
            shuffle=True,
            validation_data=(valid_data, valid_data)
        )

    def save(self, save_folder=".", save_name="model_saved.h5"):
        # Create folder if it does not exist
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        # Save the entire model
        save_path = os.path.join(save_folder, save_name)
        self.auto_encoder.save(
            save_path,
            include_optimizer=True,
            save_format='h5'
        )
        # Save only model weights in a separate h5 file
        self.auto_encoder.save_weights(os.path.join(save_folder, save_name + "_weights.h5"))
        # Save the input parameters in a pickle file (useful for the loading part)
        input_param = [
            self.input_size,
            self.layers_dim,
            self.latent_space_dim
        ]
        with open(os.path.join(save_folder, save_name + "_parameters.pkl"), "wb") as f:
            pickle.dump(input_param, f)

    def save_encoder(self, save_folder=".", save_name="encoder_saved.h5"):
        # Create folder if it does not exist
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        # Save the encoder
        # """
        save_path = os.path.join(save_folder, save_name)
        self.encoder.save(
            save_path,
            include_optimizer=True,
            save_format='h5'
        )
        # """
        # Save encoder weights in a separate h5 file
        self.encoder.save_weights(os.path.join(save_folder, save_name + "_weights.h5"))

    def save_decoder(self, save_folder=".", save_name="decoder_saved.h5"):
        # Create folder if it does not exist
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        # Save the decoder
        # """
        save_path = os.path.join(save_folder, save_name)
        self.decoder.save(
            save_path,
            include_optimizer=True,
            save_format='h5'
        )
        # """
        # Save decoder weights in a separate h5 file
        self.decoder.save_weights(os.path.join(save_folder, save_name + "_weights.h5"))
    
    def load_weights(self, model):
        self.auto_encoder.load_weights(model)

    def get_weights(self):
        self.auto_encoder.get_weights()

    @classmethod
    def load(cls, save_folder=".", save_name="model_saved.h5"):
        """
        # Get the parameters
        with open(os.path.join(save_folder, "parameters.pkl"), "rb") as f:
            input_param = pickle.load(f)
        loaded_autoenc = AutoEncoder(*input_param)
        loaded_autoenc.load_weights(os.path.join(save_folder, "weights.h5"))
        return loaded_autoenc
        """
        loaded_autoenc = load_model(os.path.join(save_folder, save_name))
        return loaded_autoenc

    def predict(self, test_data):
        self.auto_encoder.predict(
            test_data,
            batch_size=32,
            verbose=0,
            steps=None,
            callbacks=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
        )

    @classmethod
    def load_encoder(cls, save_folder=".", save_name="encoder_saved.h5"):
        loaded_enc = load_model(os.path.join(save_folder, save_name), compile=False)
        return loaded_enc

    @classmethod
    def load_decoder(cls, save_folder=".", save_name="decoder_saved.h5"):
        loaded_dec = load_model(os.path.join(save_folder, save_name), compile=False)
        return loaded_dec

    # Here we define all the PRIVATE methods of the AE class:
    # building the layers of the encoder and decoder and combining them into one AE

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_autoencoder(self):
        input_model = self._input_model
        model_output = self.decoder(self.encoder(input_model))
        self.auto_encoder = Model(input_model, model_output, name="autoencoder")

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        self._input_model = encoder_input
        encoder_layers = encoder_input
        for layer_idx in range(0, self._num_layers):
            # -------------- Encoder dense layers ---------------
            new_layer = Dense(
                self.layers_dim[layer_idx],
                activation=None,
                use_bias=True,
                kernel_initializer='random_normal',
                name=f"encoder_layer_{layer_idx}"
            )
            encoder_layers = new_layer(encoder_layers)
            # ---------------------- Batch normalization + activation -------------------------
           # encoder_layers = ReLU(name=f"encoder_relu_{layer_idx}")(encoder_layers)
            encoder_layers = BatchNormalization(name=f"encoder_bn_{layer_idx}")(encoder_layers)
        # Finish by flattening then making dense layer at latent space dim
        flat_layer = Flatten()(encoder_layers)
        dense_layer = Dense(
            self.latent_space_dim,
            activation=None, #'tanh', ###'relu'
            use_bias=True,
            kernel_initializer='random_normal',
            name='latent_space_layer'
        )
        encoder_layers = dense_layer(flat_layer)
        self.encoder = Model(encoder_input, encoder_layers, name="encoder")
        # Keep the shape before flatten for the decoder input
        self._encoder_output_shape = self.encoder.get_layer(f"encoder_bn_{self._num_layers-1}").output_shape
        return encoder_layers

    def _add_encoder_input(self):
        return Input(shape=self.input_size, name="encoder_input")

    def _build_decoder(self):
        decoder_input = Input(shape=self.latent_space_dim, name="decoder_input")
        decoder_layers = decoder_input
        # Add reshape layer starting on a Dense layer of the good size
        decoder_layers = Dense(np.prod(self._encoder_output_shape[1:]), name="dec_dense_4_reshape")(decoder_layers)
        decoder_layers = Reshape(self._encoder_output_shape[1:])(decoder_layers)
        # Decoder layers
        for layer_idx in reversed(range(0, self._num_layers)):
            # -------------- Decoder dense layers ---------------
            up_layer = Dense(
                self.layers_dim[layer_idx],
                activation=None,
                use_bias=True,
                kernel_initializer='random_normal',
                name=f"decoder_layer_{layer_idx}"
            )
            decoder_layers = up_layer(decoder_layers)
            # ---------------------- Batch normalization + activation -------------------------
            decoder_layers = BatchNormalization(name=f"decoder_bn_{layer_idx}")(decoder_layers)
            decoder_layers = ReLU(name=f"decoder_relu_{layer_idx}")(decoder_layers)
        # Final reconstruction layer --> get one single output
        final_layer = Dense(
            self.input_size[-1],
            activation=None,
            use_bias=True,
            kernel_initializer='random_normal',
            name=f"decoder_last_layer"
        )
        decoder_layers = final_layer(decoder_layers)
        # Finish with activation layer (XXX Keep sigmoid activation ?? XXX)
        decoder_layers = Activation(input_shape=self.input_size, activation='sigmoid')(decoder_layers)
        # final_layer = Activation(input_shape=self.input_size,activation='sigmoid')
        # decoder_layers = final_layer
        self.decoder = Model(decoder_input, decoder_layers, name="decoder")
        return decoder_layers


if __name__ == "__main__":
    autoenc = AutoEncoderDense(
        input_size=[214, 51, 1],
        layers_dim=[128, 64, 32],
        latent_space_dim=5
    )
    autoenc.summary()

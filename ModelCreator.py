# from keras_applications.imagenet_utils import _obtain_input_shape
from CoordConv.CoordConv import AddCoords
# from CoordConv.tf_plus import *
import tensorflow.keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Multiply, Concatenate, Conv2DTranspose, BatchNormalization, Reshape, GlobalAveragePooling2D, MaxPooling2D, SeparableConv2D, UpSampling2D
from tensorflow.keras.layers import ReLU, ELU
from tensorflow.keras.activations import selu, elu, relu
from tensorflow.keras.regularizers import l2
from resnet_blocks import *
from tensorflow.keras.utils import plot_model
from support_defs import *
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2
from tensorflow.keras.applications.vgg19 import VGG19



class ModelCreator(object):
    def __init__(self, img_size, curr_run_name = 'devel', bottleneck_dims = 128, variational = True, debug = False):
        self.debug                      = debug
        self.image_size                 = img_size
        self.bottleneck_dims            = bottleneck_dims
        self.curr_run_name              = curr_run_name
        self.variational                = variational


    def convolutional_encoder_hgt_vgglike(self, input_shape, tag = 'hgt_'):
        input_layer = Input(shape=input_shape)

        curr_shape = input_layer.shape[1:-1]
        # Block 1
        x = AddCoords(x_dim=curr_shape[1], y_dim=curr_shape[0], with_r=False)(input_layer)
        # x = input_layer
        x = Conv2D(64, (7, 7), padding='same', name='block1_conv1')(x)
        x = BatchNormalization(name='block1_conv1_bn')(x)
        x = selu(x)

        curr_shape = x.shape[1:-1]
        x = AddCoords(x_dim=curr_shape[1], y_dim=curr_shape[0], with_r=False)(x)
        x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), name='block1_conv2_strides')(x)
        x = BatchNormalization(name='block1_conv2_bn')(x)
        x = selu(x)

        # Block 2
        curr_shape = x.shape[1:-1]
        x = AddCoords(x_dim=curr_shape[1], y_dim=curr_shape[0], with_r=False)(x)
        x = Conv2D(128, (5, 5), padding='same', name='block2_conv1')(x)
        x = BatchNormalization(name='block2_conv1_bn')(x)
        x = selu(x)

        curr_shape = x.shape[1:-1]
        x = AddCoords(x_dim=curr_shape[1], y_dim=curr_shape[0], with_r=False)(x)
        x = Conv2D(128, (5, 5), padding='same', strides=(2, 2), name='block2_conv2_strides')(x)
        x = BatchNormalization(name='block2_conv2_bn')(x)
        x = selu(x)

        # Block 3
        curr_shape = x.shape[1:-1]
        x = AddCoords(x_dim=curr_shape[1], y_dim=curr_shape[0], with_r=False)(x)
        x = Conv2D(256, (5, 5), padding='same', name='block3_conv1')(x)
        x = BatchNormalization(name='block3_conv1_bn')(x)
        x = selu(x)

        curr_shape = x.shape[1:-1]
        x = AddCoords(x_dim=curr_shape[1], y_dim=curr_shape[0], with_r=False)(x)
        x = Conv2D(256, (5, 5), padding='same', strides=(2, 2), name='block3_conv2_strides')(x)
        x = BatchNormalization(name='block3_conv2_bn')(x)
        x = selu(x)

        # pixelwise depth-pooling
        curr_shape = x.shape[1:-1]
        x = AddCoords(x_dim=curr_shape[1], y_dim=curr_shape[0], with_r=False)(x)
        x = Conv2D(1, (1, 1), padding='same', name='block3_conv3')(x)
        x = BatchNormalization(name='block3_conv3_bn')(x)
        x = selu(x)                                                                                                     # [None, 32, 32, 1]

        encoder_model = Model(inputs=[input_layer], outputs=[x], name='encoder_model')

        return encoder_model


    def convolutional_decoder_ght_vgglike(self, input_shape, tag='ght_'):
        decoder_input = Input(input_shape)

        # Block 3-reversed
        curr_shape = decoder_input.shape[1:-1]                                                                          # [None, 32, 32, 1]
        x = AddCoords(x_dim=curr_shape[1], y_dim=curr_shape[0], with_r=False)(decoder_input)
        x = Conv2D(256, (5, 5), padding='same')(x)
        x = BatchNormalization()(x)
        x = selu(x)

        curr_shape = x.shape[1:-1]                                                                                      # [None, 32, 32, 256]
        x = AddCoords(x_dim=curr_shape[1], y_dim=curr_shape[0], with_r=False)(x)
        x = Conv2D(256, (5, 5), padding='same')(x)
        x = BatchNormalization()(x)
        x = selu(x)

        x = UpSampling2D((2, 2), interpolation='bilinear')(x)                                                           # [None, 64, 64, 256]

        # Block 2-reversed
        curr_shape = x.shape[1:-1]                                                                                      # [None, 64, 64, 256]
        x = AddCoords(x_dim=curr_shape[1], y_dim=curr_shape[0], with_r=False)(x)
        x = Conv2D(128, (5, 5), padding='same')(x)                                                                      # [None, 64, 64, 128]
        x = BatchNormalization()(x)
        x = selu(x)

        curr_shape = x.shape[1:-1]                                                                                      # [None, 64, 64, 128]
        x = AddCoords(x_dim=curr_shape[1], y_dim=curr_shape[0], with_r=False)(x)
        x = Conv2D(128, (5, 5), padding='same')(x)                                                                      # [None, 64, 64, 128]
        x = BatchNormalization()(x)
        x = selu(x)

        x = UpSampling2D((2, 2), interpolation='bilinear')(x)                                                           # [None, 128, 128, 128]

        # Block 1-reversed
        curr_shape = x.shape[1:-1]
        x = AddCoords(x_dim=curr_shape[1], y_dim=curr_shape[0], with_r=False)(x)
        x = Conv2D(64, (7, 7), padding='same')(x)                                                                       # [None, 128, 128, 64]
        x = BatchNormalization()(x)
        x = selu(x)

        curr_shape = x.shape[1:-1]                                                                                      # [None, 128, 128, 128]
        x = AddCoords(x_dim=curr_shape[1], y_dim=curr_shape[0], with_r=False)(x)
        x = Conv2D(64, (7, 7), padding='same')(x)                                                                       # [None, 128, 128, 64]
        x = BatchNormalization()(x)
        x = selu(x)

        x = UpSampling2D((2, 2), interpolation='bilinear')(x)                                                           # [None, 256, 256, 64]

        # Block 0-reversed
        curr_shape = x.shape[1:-1]                                                                                      # [None, 256, 256, 64]
        x = AddCoords(x_dim=curr_shape[1], y_dim=curr_shape[0], with_r=False)(x)                                        # [None, 256, 256, 66]
        x = Conv2D(64, (7, 7), padding='same')(x)                                                                       # [None, 256, 256, 64]
        x = BatchNormalization()(x)                                                                                     # [None, 256, 256, 64]
        x = selu(x)                                                                                                     # [None, 256, 256, 64]

        curr_shape = x.shape[1:-1]      # [None, 256, 256, 64]
        x = AddCoords(x_dim=curr_shape[1], y_dim=curr_shape[0], with_r=False)(x)                                        # [None, 256, 256, 66]
        x = Conv2D(128, (5, 5), padding='same')(x)                                                                      # [None, 256, 256, 128]
        x = BatchNormalization()(x)                                                                                     # [None, 256, 256, 128]
        x = selu(x)                                                                                                     # [None, 256, 256, 128]

        curr_shape = x.shape[1:-1]  # [None, 256, 256, 64]
        x = AddCoords(x_dim=curr_shape[1], y_dim=curr_shape[0], with_r=False)(x)                                        # [None, 256, 256, 130]
        x = Conv2D(256, (3, 3), padding='same')(x)                                                                      # [None, 256, 256, 256]
        x = BatchNormalization()(x)                                                                                     # [None, 256, 256, 256]
        x = selu(x)                                                                                                     # [None, 256, 256, 256]

        curr_shape = x.shape[1:-1]                                                                                      # [None, 256, 256, 256]
        x = AddCoords(x_dim=curr_shape[1], y_dim=curr_shape[0], with_r=False)(x)                                        # [None, 256, 256, 258]
        x = Conv2D(1, (1, 1), padding='same')(x)                                                                        # [None, 256, 256, 1]
        x = BatchNormalization()(x)
        decoder_output = selu(x)

        decoder_model = Model(inputs=[decoder_input], outputs=[decoder_output], name='decoder_model')

        return decoder_model




    def convolutional_encoder_hgt(self, input_shape, tag='hgt_'):
        input_layer = Input(shape=input_shape)

        x = conv_block_coordconv(input_layer, 7, (16, 16, 32), stage=1, block=tag+'a', strides=(1, 1))
        # x2 = conv_block_coordconv(input_layer, 5, (32, 32, 64), stage=1, block=tag+'a2', strides=(1, 1))
        # x3 = conv_block_coordconv(input_layer, 3, (16, 16, 32), stage=1, block=tag+'a3', strides=(1, 1))
        # x = Concatenate()([x1, x2, x3])

        x = conv_block_coordconv(x, 5, (8, 8, 16), stage=2, block=tag+'a', strides=(1, 1))
        # x2 = conv_block_coordconv(x, 3, (16, 16, 32), stage=2, block=tag+'a2', strides=(1, 1))
        # x = Concatenate()([x1, x2])

        x = conv_block_coordconv(x, 3, (16, 16, 32), stage=3, block=tag+'a')
        x = identity_block_ccordconv(x, 3, (16, 16, 32), stage=3, block=tag+'b')
        # x = identity_block_ccordconv(x, 3, (32, 32, 64), stage=3, block=tag+'c')

        x = conv_block_coordconv(x, 3, (32, 32, 64), stage=4, block=tag+'a')
        x = identity_block_ccordconv(x, 3, (32, 32, 64), stage=4, block=tag+'b')
        # x = identity_block_ccordconv(x, 3, (64, 64, 128), stage=4, block=tag+'c')

        x = conv_block_coordconv(x, 3, (64, 64, 128), stage=5, block=tag+'a')
        x = identity_block_ccordconv(x, 3, (64, 64, 128), stage=5, block=tag+'b')
        # x = identity_block_ccordconv(x, 3, (128, 128, 256), stage=5, block=tag+'c')

        x = conv_block_coordconv(x, 3, (128, 128, 256), stage=6, block=tag+'a')
        x = identity_block_ccordconv(x, 3, (128, 128, 256), stage=6, block=tag+'b')
        # x = identity_block_ccordconv(x, 3, (256, 256, 512), stage=6, block=tag+'c')

        # x = conv_block_coordconv(x, 3, (512, 512, 1024), stage=7, block=tag+'a')
        # x = identity_block_ccordconv(x, 3, (512, 512, 1024), stage=7, block=tag+'b')
        # x = identity_block_ccordconv(x, 3, (512, 512, 1024), stage=7, block=tag+'c')

        gap = GlobalAveragePooling2D()(x)

        encoder_model = Model(inputs = [input_layer], outputs = [gap], name=tag+'encoder')

        return encoder_model



    def convolutional_encoder_pv(self, input_shape, tag='pv_'):
        input_layer = Input(shape=input_shape)

        # x1 = conv_block_coordconv(input_layer, 7, (16, 16, 32), stage=1, block=tag+'a1', strides=(1, 1))
        # x2 = conv_block_coordconv(input_layer, 5, (32, 32, 64), stage=1, block=tag+'a2', strides=(1, 1))
        x = conv_block_coordconv(input_layer, 3, (16, 16, 32), stage=1, block=tag+'a3', strides=(1, 1))
        # x = Concatenate()([x1, x2, x3])

        # x1 = conv_block_coordconv(x, 5, (8, 8, 16), stage=2, block=tag+'a1', strides=(1, 1))
        x = conv_block_coordconv(x, 3, (16, 16, 32), stage=2, block=tag+'a2', strides=(1, 1))
        # x = Concatenate()([x1, x2])

        x = conv_block_coordconv(x, 3, (16, 16, 32), stage=3, block=tag+'a')
        x = identity_block_ccordconv(x, 3, (16, 16, 32), stage=3, block=tag+'b')
        # x = identity_block_ccordconv(x, 3, (32, 32, 64), stage=3, block=tag+'c')

        x = conv_block_coordconv(x, 3, (32, 32, 64), stage=4, block=tag+'a')
        x = identity_block_ccordconv(x, 3, (32, 32, 64), stage=4, block=tag+'b')
        # x = identity_block_ccordconv(x, 3, (64, 64, 128), stage=4, block=tag+'c')

        x = conv_block_coordconv(x, 3, (64, 64, 128), stage=5, block=tag+'a')
        x = identity_block_ccordconv(x, 3, (64, 64, 128), stage=5, block=tag+'b')
        # x = identity_block_ccordconv(x, 3, (128, 128, 256), stage=5, block=tag+'c')

        x = conv_block_coordconv(x, 3, (128, 128, 256), stage=6, block=tag+'a')
        x = identity_block_ccordconv(x, 3, (128, 128, 256), stage=6, block=tag+'b')
        # x = identity_block_ccordconv(x, 3, (256, 256, 512), stage=6, block=tag+'c')

        # x = conv_block_coordconv(x, 3, (512, 512, 1024), stage=7, block=tag+'a')
        # x = identity_block_ccordconv(x, 3, (512, 512, 1024), stage=7, block=tag+'b')
        # x = identity_block_ccordconv(x, 3, (512, 512, 1024), stage=7, block=tag+'c')

        gap = GlobalAveragePooling2D()(x)

        encoder_model = Model(inputs = [input_layer], outputs = [gap], name=tag+'encoder')

        return encoder_model


    def convolutional_decoder_hgt(self, input_length, tag='hgt_'):
        decoder_input = Input(input_length)

        size2D = (16,16,1)
        size2D_flattened_length = np.product(size2D)

        x = Dense(size2D_flattened_length)(decoder_input)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Reshape(size2D)(x)

        # x = conv_block_coordconv(x, 3, [64, 64, 128], stage=8, block=tag + 'a', strides=(1, 1))
        # x = identity_block_ccordconv(x, 3, [64, 64, 128], stage=8, block=tag + 'b')
        x = conv_block(x, 3, [64, 64, 128], stage=8, block=tag + 'a', strides=(1, 1))
        x = identity_block(x, 3, [64, 64, 128], stage=8, block=tag + 'b')
        # x = identity_block_ccordconv(x, 3, [64, 64, 128], stage=8, block=tag + 'c')

        x = Conv2DTranspose(128, 3, strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # x = conv_block_coordconv(x, 3, [32, 32, 64], stage=9, block=tag + 'a', strides=(1, 1))
        # x = identity_block_ccordconv(x, 3, [32, 32, 64], stage=9, block=tag + 'b')
        x = conv_block(x, 3, [32, 32, 64], stage=9, block=tag + 'a', strides=(1, 1))
        x = identity_block(x, 3, [32, 32, 64], stage=9, block=tag + 'b')
        # x = identity_block_ccordconv(x, 3, [32, 32, 64], stage=9, block=tag + 'c')

        x = Conv2DTranspose(64, 3, strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # x = conv_block_coordconv(x, 3, [16, 16, 32], stage=10, block=tag + 'a', strides=(1, 1))
        # x = identity_block_ccordconv(x, 3, [16, 16, 32], stage=10, block=tag + 'b')
        x = conv_block(x, 3, [16, 16, 32], stage=10, block=tag + 'a', strides=(1, 1))
        x = identity_block(x, 3, [16, 16, 32], stage=10, block=tag + 'b')
        # x = identity_block_ccordconv(x, 3, [16, 16, 32], stage=10, block=tag + 'c')

        x = Conv2DTranspose(32, 3, strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # x = conv_block_coordconv(x, 5, [8, 8, 16], stage=11, block=tag + 'a', strides=(1, 1))
        # x = identity_block_ccordconv(x, 5, [8, 8, 16], stage=11, block=tag + 'b')
        x = conv_block(x, 5, [8, 8, 16], stage=11, block=tag + 'a', strides=(1, 1))
        x = identity_block(x, 5, [8, 8, 16], stage=11, block=tag + 'b')
        # x = identity_block_ccordconv(x, 3, [8, 8, 16], stage=11, block=tag + 'c')

        x = Conv2DTranspose(16, 7, strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        decoder_output = Conv2D(1, (1, 1), activation=ELU(), name='decoder_output')(x)

        m = Model(inputs=[decoder_input], outputs=[decoder_output], name=tag+'decoder')

        return m



    def convolutional_decoder_pv(self, input_length, output_shape = (256,256,2), tag='pv_'):
        decoder_input = Input(input_length)

        size2D = (16,16,1)
        size2D_flattened_length = np.product(size2D)

        x = Dense(size2D_flattened_length)(decoder_input)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Reshape(size2D)(x)

        # x = conv_block_coordconv(x, 3, [64, 64, 128], stage=8, block=tag + 'a', strides=(1, 1))
        # x = identity_block_ccordconv(x, 3, [64, 64, 128], stage=8, block=tag + 'b')
        x = conv_block(x, 3, [64, 64, 128], stage=8, block=tag + 'a', strides=(1, 1))
        x = identity_block(x, 3, [64, 64, 128], stage=8, block=tag + 'b')
        # x = identity_block_ccordconv(x, 3, [64, 64, 128], stage=8, block=tag + 'c')

        x = Conv2DTranspose(128, 3, strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # x = conv_block_coordconv(x, 3, [32, 32, 64], stage=9, block=tag + 'a', strides=(1, 1))
        # x = identity_block_ccordconv(x, 3, [32, 32, 64], stage=9, block=tag + 'b')
        x = conv_block(x, 3, [32, 32, 64], stage=9, block=tag + 'a', strides=(1, 1))
        x = identity_block(x, 3, [32, 32, 64], stage=9, block=tag + 'b')
        # x = identity_block_ccordconv(x, 3, [32, 32, 64], stage=9, block=tag + 'c')

        x = Conv2DTranspose(64, 3, strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # x = conv_block_coordconv(x, 3, [16, 16, 32], stage=10, block=tag + 'a', strides=(1, 1))
        # x = identity_block_ccordconv(x, 3, [16, 16, 32], stage=10, block=tag + 'b')
        x = conv_block(x, 3, [16, 16, 32], stage=10, block=tag + 'a', strides=(1, 1))
        x = identity_block(x, 3, [16, 16, 32], stage=10, block=tag + 'b')
        # x = identity_block_ccordconv(x, 3, [16, 16, 32], stage=10, block=tag + 'c')

        x = Conv2DTranspose(32, 3, strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # x = conv_block_coordconv(x, 3, [8, 8, 16], stage=11, block=tag + 'a', strides=(1, 1))
        # x = identity_block_ccordconv(x, 3, [8, 8, 16], stage=11, block=tag + 'b')
        x = conv_block(x, 3, [8, 8, 16], stage=11, block=tag + 'a', strides=(1, 1))
        x = identity_block(x, 3, [8, 8, 16], stage=11, block=tag + 'b')
        # x = identity_block_ccordconv(x, 3, [8, 8, 16], stage=11, block=tag + 'c')

        x = Conv2DTranspose(16, 3, strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        decoder_output = Conv2D(1, (1, 1), activation=ELU(), name='decoder_output')(x)

        m = Model(inputs=[decoder_input], outputs=[decoder_output], name=tag+'decoder')

        return m



    def ComposeModel_VGGlike(self):
        hgt_input = Input(shape=tuple(list(self.image_size) + [1]), name='HGT_encoder_Input')
        # hgt_input_3ch = Lambda(K.tile, arguments={'n':(1, 1, 1, 3)})(hgt_input)
        # pv_input = Input(shape=tuple(list(self.image_size) + [1]), name='PV_encoder_Input')
        # pv_input_3ch = Lambda(K.tile, arguments={'n':(1, 1, 1, 3)})(pv_input)
        maskbatch_input = Input(shape=tuple(list(self.image_size) + [1]), name='mask_batch_input')

        encoder_model = self.convolutional_encoder_hgt_vgglike(hgt_input.shape[1:])
        encoder_output = encoder_model(hgt_input)
        x = Reshape((np.product(np.array(encoder_output.shape[1:])),))(encoder_output)

        if self.variational:
            z_mean = Dense(512, activation='sigmoid', name='z_mean')(x)
            z_log_var = Dense(512, activation=selu, name='z_log_var')(x)
            z_combined = Concatenate(name='z_combined')([z_mean, z_log_var])
            z = Lambda(sampling, name='z')([z_mean, z_log_var])

            embeddings = z
        else:
            embeddings = x

        x = Dense(np.product(np.array(encoder_output.shape[1:])), selu)(embeddings)
        sampling_output = Reshape(encoder_output.shape[1:])(x)

        decoder_model = self.convolutional_decoder_ght_vgglike(sampling_output.shape[1:])
        decoder_output = decoder_model(sampling_output)

        Decoder_MaskedOutput = Multiply(name='Decoder_MaskedOutput')([decoder_output, maskbatch_input])

        if self.variational:
            model = Model([hgt_input, maskbatch_input], [Decoder_MaskedOutput, z_combined], name='pv_CVAE')
        else:
            model = Model([hgt_input,  maskbatch_input], [Decoder_MaskedOutput], name='pv_CVAE')

        plot_model(model, to_file='./logs/%s/model_structure.png' % self.curr_run_name, show_shapes=True, expand_nested=True)
        with open('./logs/%s/model_structure.txt' % self.curr_run_name, 'w') as fh:
            model.summary(print_fn=lambda x: fh.write(x + '\n'))

        return model



    def ComposeModel(self):
        hgt_input = Input(shape=tuple(list(self.image_size) + [1]), name='HGT_encoder_Input')
        hgt_input_3ch = Lambda(K.tile, arguments={'n':(1, 1, 1, 3)})(hgt_input)
        pv_input = Input(shape=tuple(list(self.image_size) + [1]), name='PV_encoder_Input')
        pv_input_3ch = Lambda(K.tile, arguments={'n':(1, 1, 1, 3)})(pv_input)
        maskbatch_input = Input(shape=tuple(list(self.image_size) + [1]), name='mask_batch_input')

        # input_shape = input.shape[1:]

        # encoder_model_hgt = self.convolutional_encoder_hgt(hgt_input.shape[1:])
        # encoder_model_pv = self.convolutional_encoder_pv(pv_input.shape[1:])
        # encoder_output = Concatenate()([encoder_model_hgt(hgt_input), encoder_model_pv(pv_input)])

        encoder_model_hgt = ResNet50V2(include_top=False,
                                       weights='imagenet',
                                       input_shape=tuple(hgt_input_3ch.shape)[1:],
                                       pooling='avg')
        encoder_model_hgt._name = 'encoder_model_hgt'
        encoder_model_hgt.trainable = False
        hgt_encoder_output = encoder_model_hgt(hgt_input_3ch)

        encoder_model_pv = ResNet50V2(include_top=False,
                                      weights='imagenet',
                                      input_shape=tuple(pv_input_3ch.shape)[1:],
                                      pooling='avg')
        encoder_model_pv._name = 'encoder_model_pv'
        encoder_model_pv.trainable = False
        pv_encoder_output = encoder_model_pv(pv_input_3ch)
        encoder_output = Concatenate()([hgt_encoder_output, pv_encoder_output])

        if self.variational:

            z_mean = Dense(self.bottleneck_dims, activation='sigmoid', name='z_mean', kernel_regularizer = l2(0.02))(encoder_output)
            z_log_var = Dense(self.bottleneck_dims, activation='relu', name='z_log_var', kernel_regularizer = l2(0.02))(encoder_output)

            z_combined = Concatenate(name='z_combined')([z_mean, z_log_var])

            z = Lambda(sampling, name='z')([z_mean, z_log_var])

            # Decoder_hgt = self.convolutional_decoder(z, dim_flattened, block7_leastconv_shape, tag='hgt_')
            # Decoder_pv = self.convolutional_decoder(z, dim_flattened, block7_leastconv_shape, tag='pv_')
            decoder_model_hgt = self.convolutional_decoder_hgt(z.shape[1])
            decoder_model_pv = self.convolutional_decoder_pv(z.shape[1])
            decoder_hgt_output = decoder_model_hgt(z)
            decoder_pv_output = decoder_model_pv(z)
        else:
            # Decoder_hgt = self.convolutional_decoder(encoder_output, dim_flattened, block7_leastconv_shape, tag='hgt_')
            # Decoder_pv = self.convolutional_decoder(encoder_output, dim_flattened, block7_leastconv_shape, tag='pv_')
            # decoder_model = self.convolutional_decoder(encoder_output.shape[1])
            decoder_model_hgt = self.convolutional_decoder_hgt(encoder_output.shape[1])
            decoder_model_pv = self.convolutional_decoder_pv(encoder_output.shape[1])
            decoder_hgt_output = decoder_model_hgt(encoder_output)
            decoder_pv_output = decoder_model_pv(encoder_output)
            # decoder_output = decoder_model(encoder_output)

        Decoder_hgt_MaskedOutput = Multiply(name='Decoder_hgt_MaskedOutput')([decoder_hgt_output, maskbatch_input])
        Decoder_pv_MaskedOutput = Multiply(name='Decoder_pv_MaskedOutput')([decoder_pv_output, maskbatch_input])

        if self.variational:
            model = Model([hgt_input, pv_input, maskbatch_input], [Decoder_hgt_MaskedOutput, Decoder_pv_MaskedOutput, z_combined], name='pv_SpCVAE')
        else:
            model = Model([hgt_input, pv_input, maskbatch_input], [Decoder_hgt_MaskedOutput, Decoder_pv_MaskedOutput], name='pv_SpCVAE')

        plot_model(model, to_file='./logs/%s/model_structure.png' % self.curr_run_name, show_shapes=True, expand_nested=True)
        with open('./logs/%s/model_structure.txt' % self.curr_run_name, 'w') as fh:
            model.summary(print_fn=lambda x: fh.write(x + '\n'))

        return model


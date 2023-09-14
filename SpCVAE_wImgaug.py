import time, getopt, pdb, os, gc, sys
import numpy as np
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Dense, Flatten, Reshape, Lambda, Multiply, Concatenate, Conv2DTranspose
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from tensorflow.python.keras.callbacks import LambdaCallback
from CustomEarlyStoppingCallback import *
from CustomTensorBoardCallback import CustomTensorBoard
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras.metrics import mean_squared_error
from tensorflow.python.keras.losses import binary_crossentropy
from ssim_metric import *
from msssim_metric import *
from support_defs import *
from DataGeneratorImgaug import *
from CustomModelCheckpoint import *
from IntermediateVisualizations_Callback import *
from break_callback import BreakLearningCallback
from KFoldCycleCallback import *
from vgg16_local import *


K.set_floatx('float32')

log_device_placement = False


#MKPC
tries_each_IntReprVec = 3

images_size = (256,256)
# original_dim = images_size[0] * images_size[0]
maskfname = './dataset_ge40North/mask_256.npy'
# internal_representations = [2,4,8,16,32,64,96,128]

import socket
hostname = socket.gethostname()


class mkSpCVAE(object):
    def __init__(self, img_size, bottleneck_dims = 128, start_num = 0):

        self.start_num                  = start_num
        # self.gpu_num                    = gpu_num

        self.debug                      = True

        # MKPC
        self.GPUs_count                 = 1
        self.epochs                     = 70
        self.current_batch_size         = 20
        # self.current_batch_size         = 16 * self.GPUs_count

        self.epochs_to_cycle            = 5
        self.image_size                 = img_size
        self.bottleneck_dims            = bottleneck_dims
        self.fnames_prefix              = '%s_SpCVAE_hiddim%04d_startnum%02d' % (hostname, self.bottleneck_dims, start_num)
        self.mask                       = np.load('./dataset_ge40North/mask_256.npy').astype(np.float32)


    def compose_convolutional_encoder(self, input_layer, layernames_postfix = ''):
        input_3channel = Concatenate(axis=-1, name='Encoder_Gray2RGB_vgg16' + layernames_postfix)([input_layer, input_layer, input_layer])
        conv_base = VGG16_local(weights='imagenet', include_top=False,
                                input_shape=(self.image_size[0], self.image_size[1], 3),
                                weights_path='./vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                layernames_postfix=layernames_postfix)
        vggoutput = conv_base(input_3channel)
        add_block6_conv01 = Conv2D(256, (3, 3), activation='relu', padding='same', name='add_block6_conv01' + layernames_postfix)(vggoutput)
        add_block6_conv02 = Conv2D(256, (3, 3), activation='relu', padding='same', name='add_block6_conv02' + layernames_postfix)(add_block6_conv01)
        add_block6_conv03 = Conv2D(128, (3, 3), activation='relu', padding='same', name='add_block6_conv03' + layernames_postfix)(add_block6_conv02)
        add_block6_conv04 = Conv2D(128, (3, 3), activation='relu', padding='same', name='add_block6_conv04' + layernames_postfix)(add_block6_conv03)
        add_block6_conv05 = Conv2D(64, (3, 3), activation='relu', padding='same', name='add_block6_conv05' + layernames_postfix)(add_block6_conv04)
        add_block6_conv06 = Conv2D(64, (3, 3), activation='relu', padding='same', name='add_block6_conv06' + layernames_postfix)(add_block6_conv05)
        add_block6_conv07 = Conv2D(32, (3, 3), activation='relu', padding='same', name='add_block6_conv07' + layernames_postfix)(add_block6_conv06)
        add_block6_conv08 = Conv2D(32, (3, 3), activation='relu', padding='same', name='add_block6_conv08' + layernames_postfix)(add_block6_conv07)
        # add_block6_conv09 = Conv2D(16, (3, 3), activation='relu', padding='same', name='add_block6_conv09' + layernames_postfix)(add_block6_conv08)
        # add_block6_conv10 = Conv2D(16, (3, 3), activation='relu', padding='same', name='add_block6_conv10' + layernames_postfix)(add_block6_conv09)
        VGGout = Flatten()(add_block6_conv08)
        return VGGout, K.int_shape(add_block6_conv08)[-1]


    def compose_convolutional_decoder(self, input_layer, k_size, channels, tag=''):
        decoder_reshape = Reshape((k_size, k_size, channels), name='%s_Decoder_Reshape_01' % tag)(input_layer)
        # Decoder_Conv2D_0001 = Conv2D(32, (3, 3), activation='relu', padding='same', name='%s_Decoder_Conv2D_0001' % tag)(decoder_reshape)
        # Decoder_Conv2D_0002 = Conv2D(32, (3, 3), activation='relu', padding='same', name='%s_Decoder_Conv2D_0002' % tag)(Decoder_Conv2D_0001)
        Decoder_Conv2D_0003 = Conv2D(64, (3, 3), activation='relu', padding='same', name='%s_Decoder_Conv2D_0003' % tag)(decoder_reshape)
        Decoder_Conv2D_0004 = Conv2D(64, (3, 3), activation='relu', padding='same', name='%s_Decoder_Conv2D_0004' % tag)(Decoder_Conv2D_0003)
        Decoder_Conv2D_0005 = Conv2D(128, (3, 3), activation='relu', padding='same', name='%s_Decoder_Conv2D_0005' % tag)(Decoder_Conv2D_0004)
        Decoder_Conv2D_0006 = Conv2D(128, (3, 3), activation='relu', padding='same', name='%s_Decoder_Conv2D_0006' % tag)(Decoder_Conv2D_0005)
        Decoder_Conv2D_0007 = Conv2D(256, (3, 3), activation='relu', padding='same', name='%s_Decoder_Conv2D_0007' % tag)(Decoder_Conv2D_0006)
        Decoder_Conv2D_0008 = Conv2D(256, (3, 3), activation='relu', padding='same', name='%s_Decoder_Conv2D_0008' % tag)(Decoder_Conv2D_0007)
        Decoder_Conv2D_0009 = Conv2D(512, (3, 3), activation='relu', padding='same', name='%s_Decoder_Conv2D_0009' % tag)(Decoder_Conv2D_0008)
        Decoder_Conv2D_0010 = Conv2D(512, (3, 3), activation='relu', padding='same', name='%s_Decoder_Conv2D_0010' % tag)(Decoder_Conv2D_0009)

        Decoder_UpSampling2D_01 = UpSampling2D(size=(2, 2), name='%s_Decoder_UpSampling2D_01' % tag)(Decoder_Conv2D_0010)
        Decoder_Conv2D_0101 = Conv2D(512, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='%s_Decoder_Conv2D_0101' % tag)(Decoder_UpSampling2D_01)
        Decoder_Conv2D_0102 = Conv2D(512, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='%s_Decoder_Conv2D_0102' % tag)(Decoder_Conv2D_0101)
        Decoder_UpSampling2D_02 = UpSampling2D(size=(2, 2), name='%s_Decoder_UpSampling2D_02' % tag)(Decoder_Conv2D_0102)
        Decoder_Conv2D_0201 = Conv2D(256, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='%s_Decoder_Conv2D_0201' % tag)(Decoder_UpSampling2D_02)
        Decoder_Conv2D_0202 = Conv2D(256, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='%s_Decoder_Conv2D_02' % tag)(Decoder_Conv2D_0201)
        Decoder_UpSampling2D_03 = UpSampling2D(size=(2, 2), name='%s_Decoder_UpSampling2D_03' % tag)(Decoder_Conv2D_0202)
        Decoder_Conv2D_0301 = Conv2D(128, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='%s_Decoder_Conv2D_0301' % tag)(Decoder_UpSampling2D_03)
        Decoder_Conv2D_0302 = Conv2D(128, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='%s_Decoder_Conv2D_0302' % tag)(Decoder_Conv2D_0301)
        Decoder_UpSampling2D_04 = UpSampling2D(size=(2, 2), name='%s_Decoder_UpSampling2D_04' % tag)(Decoder_Conv2D_0302)
        Decoder_Conv2D_0401 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='%s_Decoder_Conv2D_0401' % tag)(Decoder_UpSampling2D_04)
        Decoder_Conv2D_0402 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='%s_Decoder_Conv2D_0402' % tag)(Decoder_Conv2D_0401)
        Decoder_UpSampling2D_05 = UpSampling2D(size=(2, 2), name='%s_Decoder_UpSampling2D_05' % tag)(Decoder_Conv2D_0402)
        Decoder_Conv2D_0501 = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='%s_Decoder_Conv2D_0501' % tag)(Decoder_UpSampling2D_05)
        Decoder_Conv2D_0502 = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='%s_Decoder_Conv2D_0502' % tag)(Decoder_Conv2D_0501)
        Decoder_Conv2D_06_out = Conv2D(1, (3, 3), strides=1, activation='sigmoid', padding='same', name='%s_Decoder_Conv2D_06_out' % tag)(Decoder_Conv2D_0502)
        return Decoder_Conv2D_06_out


    def ComposeModel_singleGPU(self, metrics = None):

        hgt_input = Input(shape=tuple(list(self.image_size) + [1]), name='HGT_Encoder_Input_vgg16')
        pv_input = Input(shape=tuple(list(self.image_size) + [1]), name='PV_Encoder_Input_vgg16')
        # pvth_input = Input(shape=tuple(list(self.image_size) + [1]), name='PVth_Encoder_Input_vgg16')
        maskbatch_input = Input(shape=tuple(list(self.image_size) + [1]), name='mask_batch_input')

        hgtVGGout,outVGGchannels = self.compose_convolutional_encoder(hgt_input, layernames_postfix='_hgt')
        pvVGGout,_ = self.compose_convolutional_encoder(pv_input, layernames_postfix='_pv')
        # pvthVGGout,_ = self.compose_convolutional_encoder(pvth_input, layernames_postfix='_pvth')

        singleVGGout_length = K.int_shape(hgtVGGout)[1]
        # VGGout = Concatenate(axis=-1)([hgtVGGout, pvVGGout, pvthVGGout])
        VGGout = Concatenate(axis=-1)([hgtVGGout, pvVGGout])

        Flatten_length = K.int_shape(VGGout)[1]
        k_size = np.int32(np.sqrt(singleVGGout_length / outVGGchannels))
        # Encoder_Dropout03 = Dropout(0.2, name='Encoder_Dropout03')(VGGout)
        # Encoder_Dense_01 = Dense(4096, activation='relu', name='Encoder_Dense_01', kernel_regularizer = l2(0.02))(Encoder_Dropout03)
        # Encoder_Dropout04 = Dropout(0.2, name='Encoder_Dropout04')(Encoder_Dense_01)
        # Encoder_Dense_02 = Dense(1024, activation='relu', name='Encoder_Dense_02', kernel_regularizer = l2(0.02))(Encoder_Dropout04)
        Encoder_Dropout05 = Dropout(0.2, name='Encoder_Dropout05')(VGGout)
        Encoder_Dense_03 = Dense(512, activation='relu', name='Encoder_Dense_03', kernel_regularizer=l2(0.02))(Encoder_Dropout05)
        Encoder_Dropout06 = Dropout(0.2, name='Encoder_Dropout06')(Encoder_Dense_03)
        Encoder_Dense_04 = Dense(256, activation='relu', name='Encoder_Dense_04', kernel_regularizer=l2(0.02))(Encoder_Dropout06)


        z_mean = Dense(self.bottleneck_dims, activation='sigmoid', name='z_mean', kernel_regularizer = l2(0.02))(Encoder_Dense_04)
        z_log_var = Dense(self.bottleneck_dims, activation='relu', name='z_log_var', kernel_regularizer = l2(0.02))(Encoder_Dense_04)

        z_combined = Concatenate(name='z_combined')([z_mean, z_log_var])

        z = Lambda(sampling, output_shape=(self.bottleneck_dims,), name='z')([z_mean, z_log_var])

        # wind speed regression part
        reg_Dropout01 = Dropout(0.1, name='reg_Dropout01')(z)
        reg_Dense01 = Dense(128, activation='relu', name='reg_Dense01', kernel_regularizer=l2(0.02))(reg_Dropout01)
        reg_Dropout02 = Dropout(0.1, name='reg_Dropout02')(reg_Dense01)
        reg_Dense03 = Dense(64, activation='relu', name='reg_Dense03', kernel_regularizer=l2(0.02))(reg_Dropout02)
        reg_Dropout04 = Dropout(0.1, name='reg_Dropout04')(reg_Dense03)
        reg_Dense05 = Dense(32, activation='relu', name='reg_Dense05', kernel_regularizer=l2(0.02))(reg_Dropout04)
        reg_output = Dense(2, activation=None, name='reg_output', kernel_regularizer=l2(0.02))(reg_Dense05)


        # decoder
        Decoder_Dense_01 = Dense(256, activation='relu', name='Decoder_Dense_01', kernel_regularizer = l2(0.02))(z)
        Decoder_drop_01 = Dropout(0.2, name='Decoder_drop_01')(Decoder_Dense_01)
        Decoder_Dense_02 = Dense(512, activation='relu', name='Decoder_Dense_02', kernel_regularizer = l2(0.02))(Decoder_drop_01)
        Decoder_Dropout02 = Dropout(0.2, name='Decoder_Dropout02')(Decoder_Dense_02)
        # Decoder_Dense_03 = Dense(1024, activation='relu', name='Decoder_Dense_03', kernel_regularizer=l2(0.02))(Decoder_Dropout02)
        # Decoder_Dropout03 = Dropout(0.2, name='Decoder_Dropout03')(Decoder_Dense_03)
        # Decoder_Dense_04 = Dense(4096, activation='relu', name='Decoder_Dense_04', kernel_regularizer=l2(0.02))(Decoder_Dropout03)
        # Decoder_Dropout04 = Dropout(0.2, name='Decoder_Dropout04')(Decoder_Dense_04)
        Decoder_Dense_05 = Dense(Flatten_length, activation='relu', name='Decoder_Dense_05', kernel_regularizer = l2(0.02))(Decoder_Dropout02)

        # hgt_flattened, pv_flattened, phth_flattened = Lambda(lambda x: (x[:,:singleVGGout_length], x[:,singleVGGout_length:2*singleVGGout_length], x[:,2*singleVGGout_length:]))(Decoder_Dense_03)
        hgt_flattened, pv_flattened = Lambda(lambda x: (x[:, :singleVGGout_length], x[:, singleVGGout_length:]))(Decoder_Dense_05)
        hgt_cnn_decoder = self.compose_convolutional_decoder(hgt_flattened, k_size, outVGGchannels, tag='HGT')
        pv_cnn_decoder = self.compose_convolutional_decoder(pv_flattened, k_size, outVGGchannels, tag='PV')
        # pvth_cnn_decoder = self.compose_convolutional_decoder(phth_flattened, k_size, outVGGchannels, tag='PVth')

        Decoder_hgt_MaskedOutput = Multiply(name='Decoder_hgt_MaskedOutput')([hgt_cnn_decoder, maskbatch_input])
        Decoder_pv_MaskedOutput = Multiply(name='Decoder_pv_MaskedOutput')([pv_cnn_decoder, maskbatch_input])
        # Decoder_pvth_MaskedOutput = Multiply(name='Decoder_pvth_MaskedOutput')([pvth_cnn_decoder, maskbatch_input])



        # model = Model([hgt_input, pv_input, pvth_input, maskbatch_input], [Decoder_hgt_MaskedOutput, Decoder_pv_MaskedOutput, Decoder_pvth_MaskedOutput, wsp_reg_output, z_combined], name='pv_SpCVAE')
        model = Model([hgt_input, pv_input, maskbatch_input], [Decoder_hgt_MaskedOutput, Decoder_pv_MaskedOutput, reg_output, z_combined], name='pv_SpCVAE')
        for l in model.layers:
            if 'vgg16' in l.name:
                l.trainable = False
            else:
                l.trainable = True

        plot_model(model, to_file='./output/' + self.fnames_prefix + '_model_structure.png', show_shapes=True)
        with open('./output/' + self.fnames_prefix + '_model_structure.txt', 'w') as fh:
            model.summary(print_fn=lambda x: fh.write(x + '\n'))

        # self.template_model = model
        # self.encoder = encoder


        return model




    def get_cae(self, metrics = None):
        if self.GPUs_count <= 1:
            return self.ComposeModel_singleGPU(metrics=metrics)



    def train(self):
        EnsureDirectoryExists('./output/')
        EnsureDirectoryExists('./logs/')



        # config = tf.ConfigProto(allow_soft_placement=True, log_device_placement = log_device_placement, device_count={'CPU': 1, 'GPU': self.GPUs_count})
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=log_device_placement, device_count={'CPU': 1, 'GPU': 1})
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        K.set_session(session)


        metrics = {'Decoder_hgt_MaskedOutput': [SSIMMetric(tag = 'HGT'), keras.metrics.MSE, MSSSIMMetric(average=False,
                                                                                                         debug=True,
                                                                                                         log_directory = './logs/',
                                                                                                         tag = 'HGT')],
                   'Decoder_pv_MaskedOutput': [SSIMMetric(tag = 'PV'), keras.metrics.MSE, MSSSIMMetric(average=False,
                                                                                                       debug=True,
                                                                                                       log_directory = './logs/',
                                                                                                       tag = 'PV')]}
                   # 'Decoder_pvth_MaskedOutput': [SSIMMetric(tag = 'PVth'), custom_MSE, MSSSIMMetric(average=False,
                   #                                                                                  debug=True,
                   #                                                                                  log_directory = './logs/',
                   #                                                                                  tag = 'PVth')]}

        model = self.get_cae(metrics)
        print("got SpCVAE")


        def beta_value_vs_epoch(ep):
            return 1.
            # return (1. - 0.9 * np.exp(-0.1 * ep))
        def hgt_reconstruction_weight_vs_epoch(ep):
            # return (0.8 + 0.4*np.exp(-0.2*ep))
            return (self.image_size[0]*self.image_size[1])
        def pv_reconstruction_weight_vs_epoch(ep):
            # return (0.8 + 0.4*np.exp(-0.2*ep))
            return (self.image_size[0] * self.image_size[1])
        # def pvth_reconstruction_weight_vs_epoch(ep):
        #     return (30 + 20.*np.exp(-0.1*ep))
        def z_combined_weight_vs_epoch(ep):
            # return (0.8 - 1.0 * np.exp(-0.3 * ep) + 0.5 * np.exp(-0.1 * ep))
            return self.current_batch_size
        def wspregroutput_combined_weight_vs_epoch(ep):
            # return (0.5 - 0.5 * np.exp(-0.05 * ep))
            return self.current_batch_size

        beta                            = K.variable(value=beta_value_vs_epoch(0))
        hgt_reconstruction_weight       = K.variable(value=hgt_reconstruction_weight_vs_epoch(0))
        pv_reconstruction_weight        = K.variable(value=pv_reconstruction_weight_vs_epoch(0))
        # pvth_reconstruction_weight      = K.variable(value=pvth_reconstruction_weight_vs_epoch(0))
        z_combined_weight               = K.variable(value=z_combined_weight_vs_epoch(0))
        wspregroutput_combined_weight   = K.variable(value=wspregroutput_combined_weight_vs_epoch(0))
        def WeightsWarmup_schedule(epoch):
            beta_value = beta_value_vs_epoch(epoch)
            print("VAE KL loss beta: %f" % beta_value)
            K.set_value(beta, beta_value)
            K.set_value(hgt_reconstruction_weight, hgt_reconstruction_weight_vs_epoch(epoch))
            K.set_value(pv_reconstruction_weight, pv_reconstruction_weight_vs_epoch(epoch))
            # K.set_value(pvth_reconstruction_weight, pvth_reconstruction_weight_vs_epoch(epoch))
            K.set_value(z_combined_weight, z_combined_weight_vs_epoch(epoch))
            K.set_value(wspregroutput_combined_weight, wspregroutput_combined_weight_vs_epoch(epoch))


        WeightsSchedulingCallback = LambdaCallback(on_epoch_end=lambda epoch, log: WeightsWarmup_schedule(epoch))

        def lr_vs_epoch(epoch, lr_initial=1.e-4, exp_rate=0.1, lower_bound=1.e-8, epochs_drop=3):
            return lower_bound + (lr_initial - lower_bound) * np.exp(-exp_rate * (epoch // epochs_drop))

        model.compile(optimizer=Adam(lr=lr_vs_epoch(0)),
                      loss={'Decoder_hgt_MaskedOutput': keras.losses.binary_crossentropy,
                            'Decoder_pv_MaskedOutput': keras.losses.binary_crossentropy,
                            # 'Decoder_pvth_MaskedOutput': 'binary_crossentropy',
                            'z_combined': SpVAE_loss(rho=0.1,
                                                     beta=beta,
                                                     logfile='./logs/' + self.fnames_prefix + '_SpVAE_loss_debug.log'),
                            'reg_output': keras.losses.mse},
                      loss_weights={'Decoder_hgt_MaskedOutput': hgt_reconstruction_weight,
                                    'Decoder_pv_MaskedOutput': pv_reconstruction_weight,
                                    # 'Decoder_pvth_MaskedOutput': pvth_reconstruction_weight,
                                    'z_combined': z_combined_weight,
                                    'reg_output': wspregroutput_combined_weight},
                      metrics=metrics)




        print("loading data")
        # datagen = DataManager(srcdatafiles={'hgt': './dataset_ge40North/hgt/res256x256/hgt_data_projected.npy',
        #                                     'pv': './dataset_ge40North/pv/res256x256/pv_data_projected.npy',
        #                                     'pv_th': './dataset_ge40North/pv_thresholded/res256x256/pv_thresholded_data_projected.npy',
        #                                     'mwsp': './dataset_ge40North/mean_wind_speed_along_jet.npy'},
        #                       normdata_files={'mwsp': './dataset_ge40North/wsp_normalization_vales.pkl'},
        #                       bottleneck_dims=self.bottleneck_dims,
        #                       input_vars=['hgt', 'pv', 'pv_th'],
        #                       output_vars=['hgt', 'pv', 'pv_th', 'mwsp'],
        #                       mask_file='./dataset_ge40North/mask_256.npy',
        #                       mmap_vars=['hgt', 'pv', 'pv_th'],
        #                       val_folds=5,
        #                       test_split_ratio = None,
        #                       image_size=self.image_size)
        datagen = DataManager(srcdatafiles={'hgt': './dataset_ge40North/hgt/hgt_data_projected_all.normed01.npy',
                                            'pv': './dataset_ge40North/pv/pv_data_projected_all.normed01.npy',
                                            'reg_output': './dataset_ge40North/regression_MeanZonalWindSpeed60N_TPolarCape.npy'},
                              normdata_files = {},
                              bottleneck_dims = self.bottleneck_dims,
                              input_vars=['hgt', 'pv'],
                              expand_dims_for_channel = ['hgt', 'pv'],
                              imgaugment_vars=['hgt', 'pv'],
                              output_vars=['hgt', 'pv', 'reg_output'],
                              mask_file='./dataset_ge40North/mask_256.npy',
                              mmap_vars=['hgt', 'pv'],
                              val_folds=5,
                              test_split_ratio=None,
                              image_size=self.image_size)

        train_generator = datagen.flow(None, batch_size=self.current_batch_size, category='train')
        val_generator = datagen.flow(None, batch_size=self.current_batch_size, category='val')

        def msssim_monitor(logs):
            # return np.sum([logs['val_%s_MSSSIMMetric_%s' % (outputname,tag)] for (tag,outputname) in zip(['HGT', 'PV', 'PVth'], ['Decoder_hgt_MaskedOutput', 'Decoder_pv_MaskedOutput', 'Decoder_pvth_MaskedOutput'])])
            return np.sum([logs['val_%s_MSSSIMMetric_%s' % (outputname, tag)] for (tag, outputname) in zip(['HGT', 'PV'], ['Decoder_hgt_MaskedOutput', 'Decoder_pv_MaskedOutput'])])
        # def mse_monitor(logs):
        #     return np.sum([logs['val_%s_custom_MSE' % outputname] for outputname in ['Decoder_hgt_MaskedOutput', 'Decoder_pv_MaskedOutput', 'Decoder_pvth_MaskedOutput']])
        # def ssim_monitor(logs):
        #     return np.sum([logs['val_%s_SSIMMetric_%s' % (outputname, tag)] for (tag, outputname) in zip(['HGT', 'PV', 'PVth'], ['Decoder_hgt_MaskedOutput', 'Decoder_pv_MaskedOutput', 'Decoder_pvth_MaskedOutput'])])

#region callbacks
        filepathMSSSIMMetric = './logs/' + self.fnames_prefix + '_checkpoint_ep{epoch:04d}_valMSSSIM_{monitorvalue:.6f}.hdf5'
        checkpointingMSSSIMMetric = CustomModelCheckpoint(model, filepathMSSSIMMetric, monitor=msssim_monitor, verbose=1, save_best_only=True, mode='min')
        filepathLossMetric = './logs/' + self.fnames_prefix + '_epoch{epoch:04d}-valLOSS_{monitorvalue:.6f}.hdf5'
        checkpointingLoss = CustomModelCheckpoint(model, filepathLossMetric, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        cycle_train_val_sets = KFoldCycleCallback(datagen, epochs_to_cycle = self.epochs_to_cycle)
        csv_logger = CSVLogger('./logs/' + self.fnames_prefix + 'train_progress.csv', separator=';', append=True, )
        tb_callback = CustomTensorBoard(log_dir='./logs/TBoard/' + self.fnames_prefix + '_warmup/', write_graph=True, update_freq=self.current_batch_size*32)
        lrCallback = LearningRateScheduler(lambda ep: lr_vs_epoch(ep), verbose=1)
        # EarlyStoppingCallback = CustomEarlyStopping(monitor=msssim_monitor, min_delta=0., patience=10, verbose=1, mode='min')
        InterVizCallback = IntermediateVisualizations(datagen, output_path='./logs/', model_prefix = self.fnames_prefix+'_warmup', model = model, mask = self.mask, picture_vars=['hgt', 'pv'], corr_vars = ['reg_output'])
        stopping_callback = BreakLearningCallback(model, control_file='./break/triggerfile')

#endregion

        print('\n\n\n\n\n\n')
        print('========================================')
        print('Fitting model - WARMUP (VGG is fixed)...')
        print('========================================')
        print('\n\n\n\n\n\n')

        start = time.time()

        # warmup for 10 epochs
        warmup_epochs = 10
        history = model.fit_generator(train_generator,
                                      steps_per_epoch = datagen.dataset_length_batches(category='train', batch_size = self.current_batch_size),
                                      epochs = warmup_epochs,
                                      validation_data = val_generator,
                                      validation_steps = datagen.dataset_length_batches(category='val', batch_size=self.current_batch_size),
                                      callbacks=[checkpointingMSSSIMMetric, checkpointingLoss, csv_logger, tb_callback, cycle_train_val_sets, lrCallback, WeightsSchedulingCallback, InterVizCallback, stopping_callback])

        model.save('./logs/' + self.fnames_prefix + '_checkpoint_after_warmup.h5', include_optimizer=False)

        print('\n\n\n\n\n\n')
        print('====================================================')
        print('Fitting the model - all the weights are optimized...')
        print('====================================================')
        print('\n\n\n\n\n\n')

        # change all weights to trainable and continue training
        for l in model.layers:
            l.trainable = True

        beta = K.variable(value=beta_value_vs_epoch(warmup_epochs))
        hgt_reconstruction_weight = K.variable(value=hgt_reconstruction_weight_vs_epoch(warmup_epochs))
        pv_reconstruction_weight = K.variable(value=pv_reconstruction_weight_vs_epoch(warmup_epochs))
        # pvth_reconstruction_weight = K.variable(value=pvth_reconstruction_weight_vs_epoch(warmup_epochs))
        z_combined_weight = K.variable(value=z_combined_weight_vs_epoch(warmup_epochs))
        wspregroutput_combined_weight = K.variable(value=wspregroutput_combined_weight_vs_epoch(warmup_epochs))

        def WeightsWarmup_schedule_2stage(epoch):
            beta_value = beta_value_vs_epoch(epoch+warmup_epochs)
            print("VAE KL loss beta: %f" % beta_value)
            K.set_value(beta, beta_value)
            K.set_value(hgt_reconstruction_weight, hgt_reconstruction_weight_vs_epoch(epoch+warmup_epochs))
            K.set_value(pv_reconstruction_weight, pv_reconstruction_weight_vs_epoch(epoch+warmup_epochs))
            # K.set_value(pvth_reconstruction_weight, pvth_reconstruction_weight_vs_epoch(epoch+warmup_epochs))
            K.set_value(z_combined_weight, z_combined_weight_vs_epoch(epoch+warmup_epochs))
            K.set_value(wspregroutput_combined_weight, wspregroutput_combined_weight_vs_epoch(epoch+warmup_epochs))

        WeightsSchedulingCallback_2stage = LambdaCallback(on_epoch_end=lambda epoch, log: WeightsWarmup_schedule_2stage(epoch))

        model.compile(optimizer=Adam(lr=lr_vs_epoch(warmup_epochs)),
                      loss={'Decoder_hgt_MaskedOutput': keras.losses.binary_crossentropy,
                            'Decoder_pv_MaskedOutput': keras.losses.binary_crossentropy,
                            # 'Decoder_pvth_MaskedOutput': 'binary_crossentropy',
                            'z_combined': SpVAE_loss(rho=0.1,
                                                     beta=beta,
                                                     logfile='./logs/' + self.fnames_prefix + '_SpVAE_loss_debug.log'),
                            'reg_output': 'mse'},
                      loss_weights={'Decoder_hgt_MaskedOutput': hgt_reconstruction_weight,
                                    'Decoder_pv_MaskedOutput': pv_reconstruction_weight,
                                    # 'Decoder_pvth_MaskedOutput': pvth_reconstruction_weight,
                                    'z_combined': z_combined_weight,
                                    'reg_output': wspregroutput_combined_weight},
                      metrics=metrics)


#region new callbacks
        filepathMSSSIMMetric = './logs/' + self.fnames_prefix + '_checkpoint_ep{epoch:04d}_valMSSSIM_{monitorvalue:.6f}.hdf5'
        checkpointingMSSSIMMetric = CustomModelCheckpoint(model, filepathMSSSIMMetric, monitor=msssim_monitor, verbose=1, save_best_only=True, mode='min', start_epoch=warmup_epochs)
        filepathLossMetric = './logs/' + self.fnames_prefix + '_epoch{epoch:04d}-valLOSS_{monitorvalue:.6f}.hdf5'
        checkpointingLoss = CustomModelCheckpoint(model, filepathLossMetric, monitor='val_loss', verbose=1, save_best_only=True, mode='min', start_epoch=warmup_epochs)
        cycle_train_val_sets = KFoldCycleCallback(datagen, epochs_to_cycle=self.epochs_to_cycle)
        csv_logger = CSVLogger('./logs/' + self.fnames_prefix + 'train_progress.csv', separator=';', append=True, )
        tb_callback = CustomTensorBoard(log_dir='./logs/TBoard/' + self.fnames_prefix + '/', write_graph=True, update_freq=self.current_batch_size * 32)
        lrCallback = LearningRateScheduler(lambda ep: lr_vs_epoch(ep+warmup_epochs), verbose=1)
        InterVizCallback = IntermediateVisualizations(datagen, output_path='./logs/', model_prefix=self.fnames_prefix, model=model, mask=self.mask, picture_vars=['hgt', 'pv'], corr_vars=['reg_output'])
#endregion

        history2 = model.fit_generator(train_generator,
                                       steps_per_epoch=datagen.dataset_length_batches(category='train', batch_size=self.current_batch_size),
                                       epochs = self.epochs,
                                       validation_data=val_generator,
                                       validation_steps=datagen.dataset_length_batches(category='val', batch_size=self.current_batch_size),
                                       callbacks=[checkpointingMSSSIMMetric, checkpointingLoss, csv_logger, tb_callback, cycle_train_val_sets, lrCallback, WeightsSchedulingCallback_2stage, InterVizCallback])


        end = time.time()

        # with open('./output/' + self.fnames_prefix + 'model.json', "w") as json_file:
        #     json_file.write(self.template_model.to_json())
        model.save('./logs/' + self.fnames_prefix + '_checkpoint_final_weights.h5', include_optimizer=False)

        # print('loading test data')
        # test_generator = datagen.flow(None, batch_size=self.current_batch_size, category='test')

        # print('evaluating final model...')
        # final_eval_result = model.evaluate_generator(test_generator, steps=datagen.dataset_length_batches(category='test', batch_size=self.current_batch_size))
        # print(final_eval_result)

        K.clear_session()
        gc.collect()

        print("training time for %d epochs: %.2fs = %.2fHrs" % (len(list(history.history.values())[0]) + len(list(history2.history.values())[0]), (end - start), (end - start) / 3600.0))
        with open('./logs/' + self.fnames_prefix + 'train_summary.txt', 'w') as f:
            f.writelines("training time for %d epochs: %.2fs = %.2fHrs" % (len(list(history.history.values())[0]) + (len(list(history2.history.values())[0])), (end - start), (end - start) / 3600.0))
            # f.writelines('\n\nevaluating final model:')
            # f.writelines(str(final_eval_result))







if __name__ == '__main__':
    input_args = sys.argv[1:]
    opts, args = getopt.getopt(input_args, "", ["hiddim=", "startidx=", "runs="])

    if ('--hiddim' in [opt[0] for opt in opts]):
        bottleneck_dim = int([opt for opt in opts if opt[0] == '--hiddim'][0][1])
    else:
        bottleneck_dim = 128

    if ('--startidx' in [opt[0] for opt in opts]):
        start_idx = int([opt for opt in opts if opt[0] == '--startidx'][0][1])
    else:
        start_idx = 0

    if ('--runs' in [opt[0] for opt in opts]):
        runs = int([opt for opt in opts if opt[0] == '--runs'][0][1])
    else:
        runs = 4


    for tryIdx in range(start_idx, start_idx+runs):
        myCVAE = mkSpCVAE(img_size = images_size,
                          bottleneck_dims = bottleneck_dim,
                          start_num = tryIdx)
        try:
            myCVAE.train()
        except Exception as ex1:
            err_fname = './logs/' + myCVAE.fnames_prefix + 'errors.log'
            exc_type, exc_value, exc_traceback = sys.exc_info()
            with open(err_fname, 'a') as errf:
                traceback.print_tb(exc_traceback, limit=None, file=errf)
                traceback.print_exception(exc_type, exc_value, exc_traceback, limit=None, file=errf)
            print(str(ex1))
        del myCVAE
        K.clear_session()
        gc.collect()

    print('\n\nFINISHED')
    quit()
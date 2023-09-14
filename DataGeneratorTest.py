import time, getopt, pdb, os, gc, sys
import numpy as np
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Dense, Flatten, Reshape, Lambda, Multiply, Concatenate, Conv2DTranspose
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger, TensorBoard, EarlyStopping
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras.metrics import mean_squared_error
from tensorflow.python.keras.losses import binary_crossentropy
from ssim_metric import *
from msssim_metric import *
from support_defs import *
from DataGenerator import *
from CustomModelCheckpoint import *
from IntermediateVisualizations_Callback import *
from KFoldCycleCallback import *
from vgg16_local import *



log_device_placement = False


#MKPC
tries_each_IntReprVec = 1

images_size = (256,256)
internal_representations = [2,3,8,16,32,64,128]


class mkSpCVAE(object):
    def __init__(self, img_size, internal_representation = 128, start_num = 0):

        self.start_num                  = start_num
        # self.gpu_num                    = gpu_num

        self.debug                      = True

        # MKPC
        self.GPUs_count                 = 1
        self.epochs                     = 150
        # self.current_batch_size         = 2
        self.current_batch_size         = 16 * self.GPUs_count

        self.epochs_to_cycle            = 10
        self.image_size                 = img_size
        self.internal_representation    = internal_representation
        self.fnames_prefix              = 'SpCVAE_4pv_clustering_hiddim%04d_startnum%04d' % (self.internal_representation, start_num)


    def ComposeModel_singleGPU(self, metrics = []):
        img_input = Input(shape=tuple(list(self.image_size) + [3]), name='Encoder_Input_vgg16')

        conv_base2 = VGG16_local(weights='imagenet', include_top=False,
                                 input_shape=(self.image_size[0], self.image_size[1], 3),
                                 weights_path='./vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                 layernames_postfix='_vgg16')
        conv_base2.trainable = False
        vgg_model = conv_base2(img_input)

        Encoder_Flatten = Flatten()(vgg_model)
        Flatten_length = Encoder_Flatten._keras_shape[1]
        Encoder_Dropout03 = Dropout(0.2, name='Encoder_Dropout03')(Encoder_Flatten)
        Encoder_Dense_01 = Dense(4096, activation='relu', name='Encoder_Dense_01')(Encoder_Dropout03)
        Encoder_Dropout04 = Dropout(0.2, name='Encoder_Dropout04')(Encoder_Dense_01)
        Encoder_Dense_02 = Dense(512, activation='relu', name='Encoder_Dense_02')(Encoder_Dropout04)

        z_mean = Dense(self.internal_representation, activation='sigmoid', name='z_mean')(Encoder_Dense_02)
        z_log_var = Dense(self.internal_representation, activation='relu', name='z_log_var')(Encoder_Dense_02)

        z_combined = Concatenate(name='z_combined')([z_mean, z_log_var])

        z = Lambda(sampling, output_shape=(self.internal_representation,), name='z')([z_mean, z_log_var])

        encoder = Model(img_input, [z_mean, z_log_var, z, z_combined])
        for l in encoder.layers:
            if 'vgg16' in l.name:
                l.trainable = False
            else:
                l.trainable = True

        plot_model(encoder, to_file='./output/' + self.fnames_prefix + '_Encoder_model_structure.png', show_shapes=True)
        with open('./output/' + self.fnames_prefix + '_Encoder_model_structure.txt', 'w') as fh:
            encoder.summary(print_fn=lambda x: fh.write(x + '\n'))

        # Decoder_Zsampling_input = Input(shape=(self.internal_representation,), name='Decoder_Zsampling_input')
        Decoder_mask_input = Input(shape=tuple(list(self.image_size) + [1]), name='Decoder_mask_input')

        Decoder_Dense_01 = Dense(512, activation='relu', name='Decoder_Dense_01')(z)
        Decoder_drop_01 = Dropout(0.2, name='Decoder_drop_01')(Decoder_Dense_01)
        Decoder_Dense_02 = Dense(4096, activation='relu', name='Decoder_Dense_02')(Decoder_drop_01)
        Decoder_Dropout02 = Dropout(0.2, name='Decoder_Dropout02')(Decoder_Dense_02)
        Decoder_Dense_03 = Dense(Flatten_length, activation='relu', name='Decoder_Dense_03')(Decoder_Dropout02)
        k_size = np.int32(np.sqrt(Flatten_length / 512))
        Decoder_Reshape_01 = Reshape((k_size, k_size, 512), name='Decoder_Reshape_01')(Decoder_Dense_03)

        Decoder_UpSampling2D_01 = UpSampling2D(size=(2, 2), name='Decoder_UpSampling2D_01')(Decoder_Reshape_01)
        Decoder_Conv2DTranspose_0101 = Conv2D(512, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='Decoder_Conv2DTranspose_0101')(Decoder_UpSampling2D_01)
        Decoder_Conv2DTranspose_0102 = Conv2D(512, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='Decoder_Conv2DTranspose_0102')(Decoder_Conv2DTranspose_0101)
        Decoder_UpSampling2D_02 = UpSampling2D(size=(2, 2), name='Decoder_UpSampling2D_02')(Decoder_Conv2DTranspose_0102)
        Decoder_Conv2DTranspose_0201 = Conv2D(256, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='Decoder_Conv2DTranspose_0201')(Decoder_UpSampling2D_02)
        Decoder_Conv2DTranspose_0202 = Conv2D(256, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='Decoder_Conv2DTranspose_02')(Decoder_Conv2DTranspose_0201)
        Decoder_UpSampling2D_03 = UpSampling2D(size=(2, 2), name='Decoder_UpSampling2D_03')(Decoder_Conv2DTranspose_0202)
        Decoder_Conv2DTranspose_0301 = Conv2D(128, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='Decoder_Conv2DTranspose_0301')(Decoder_UpSampling2D_03)
        Decoder_Conv2DTranspose_0302 = Conv2D(128, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='Decoder_Conv2DTranspose_0302')(Decoder_Conv2DTranspose_0301)
        Decoder_UpSampling2D_04 = UpSampling2D(size=(2, 2), name='Decoder_UpSampling2D_04')(Decoder_Conv2DTranspose_0302)
        Decoder_Conv2DTranspose_0401 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='Decoder_Conv2DTranspose_0401')(Decoder_UpSampling2D_04)
        Decoder_Conv2DTranspose_0402 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='Decoder_Conv2DTranspose_0402')(Decoder_Conv2DTranspose_0401)
        Decoder_UpSampling2D_05 = UpSampling2D(size=(2, 2), name='Decoder_UpSampling2D_05')(Decoder_Conv2DTranspose_0402)
        Decoder_Conv2DTranspose_0501 = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='Decoder_Conv2DTranspose_0501')(Decoder_UpSampling2D_05)
        Decoder_Conv2DTranspose_0502 = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='Decoder_Conv2DTranspose_0502')(Decoder_Conv2DTranspose_0501)
        Decoder_Conv2DTranspose_06_out = Conv2D(1, (3, 3), strides=1, activation='sigmoid', padding='same', name='Decoder_Conv2DTranspose_06_out')(Decoder_Conv2DTranspose_0502)
        DecoderMaskedOutput = Multiply(name='DecoderMaskedOutput')([Decoder_Conv2DTranspose_06_out, Decoder_mask_input])


        model = Model([img_input, Decoder_mask_input], [DecoderMaskedOutput, z_combined], name='pv_SpCVAE')
        for l in model.layers:
            if 'vgg16' in l.name:
                l.trainable = False
            else:
                l.trainable = True

        plot_model(model, to_file='./output/' + self.fnames_prefix + '_model_structure.png', show_shapes=True)
        with open('./output/' + self.fnames_prefix + '_model_structure.txt', 'w') as fh:
            model.summary(print_fn=lambda x: fh.write(x + '\n'))

        self.template_model = model
        self.encoder = encoder
        # self.decoder = decoder

        model.compile(optimizer=Adam(lr=1e-4),
                      loss= {'DecoderMaskedOutput': 'binary_crossentropy',
                             'z_combined': SpVAE_loss(rho=0.1,
                                                      beta=0.1,
                                                      debug=True,
                                                      logfile='./logs/' + self.fnames_prefix + '_SpVAE_loss_debug.log')},
                      loss_weights={'DecoderMaskedOutput': self.image_size[0] * self.image_size[1],
                                    'z_combined': 1.0},
                      metrics=metrics)
        return model




    def get_cae(self, metrics = []):
        if self.GPUs_count <= 1:
            return self.ComposeModel_singleGPU(metrics=metrics)



    def train(self):
        EnsureDirectoryExists('./output/')
        EnsureDirectoryExists('./logs/')



        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement = log_device_placement, device_count={'CPU': 1, 'GPU': self.GPUs_count})
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        K.set_session(session)


        print("loading data")
        datagen = DataManager(srcdatafiles={'hgt': './dataset_ge40North/hgt/res256x256/hgt_data_projected.npy',
                                            'pv': './dataset_ge40North/pv/res256x256/pv_data_projected.npy',
                                            'pv_th': './dataset_ge40North/pv_thresholded/res256x256/pv_thresholded_data_projected.npy',
                                            'mwsp': './dataset_ge40North/mean_wind_speed_along_jet.npy'},
                              normdata_files={'mwsp': './dataset_ge40North/wsp_normalization_vales.pkl'},
                              SpCVAE_internal_representation=self.internal_representation,
                              input_vars=['hgt', 'pv', 'pv_th'],
                              output_vars=['hgt', 'pv', 'pv_th', 'mwsp'],
                              mask_file='./dataset_ge40North/mask_256.npy',
                              val_folds=5,
                              test_split_ratio=0.2,
                              image_size=self.image_size)

        train_generator = datagen.flow(None, batch_size=self.current_batch_size, category='train')
        val_generator = datagen.flow(None, batch_size=self.current_batch_size, category='val')

        X,y = train_generator._get_batches_of_transformed_samples([next(train_generator.index_generator)])

        print('OK')







if __name__ == '__main__':
    input_args = sys.argv[1:]
    opts, args = getopt.getopt(input_args, "", ["hiddim=", "startidx="])

    if ('--hiddim' in [opt[0] for opt in opts]):
        internal_representation = int([opt for opt in opts if opt[0] == '--hiddim'][0][1])
    else:
        internal_representation = 128

    if ('--startidx' in [opt[0] for opt in opts]):
        start_idx = int([opt for opt in opts if opt[0] == '--startidx'][0][1])
    else:
        start_idx = 0


    for tryIdx in range(start_idx, tries_each_IntReprVec):
        mySpCVAE = mkSpCVAE(img_size = images_size,
                            internal_representation = internal_representation,
                            start_num = tryIdx)
        try:
            mySpCVAE.train()
        except Exception as ex1:
            err_fname = './logs/' + mySpCVAE.fnames_prefix + 'errors.log'
            exc_type, exc_value, exc_traceback = sys.exc_info()
            with open(err_fname, 'a') as errf:
                traceback.print_tb(exc_traceback, limit=None, file=errf)
                traceback.print_exception(exc_type, exc_value, exc_traceback, limit=None, file=errf)
            print(str(ex1))
        del mySpCVAE
        K.clear_session()
        gc.collect()

    print('\n\nFINISHED')
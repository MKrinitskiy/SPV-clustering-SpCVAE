import time, getopt, pdb, os, gc, sys
import numpy as np
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Dense, Flatten, Reshape, Lambda, Multiply, Concatenate, Conv2DTranspose
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
from DataGenerator import *
from CustomModelCheckpoint import *
from IntermediateVisualizations_Callback import *
from KFoldCycleCallback import *
from vgg16_local import *


K.set_floatx('float32')

log_device_placement = False


#MKPC
tries_each_IntReprVec = 1

images_size = (256,256)
# original_dim = images_size[0] * images_size[0]
maskfname = './dataset_ge40North/mask_256.npy'
internal_representations = [2,4,8,16,32,64,96,128]


class mkSpCVAE(object):
    def __init__(self, img_size, internal_representation = 128, start_num = 0):

        self.start_num                  = start_num
        self.debug                      = True
        self.GPUs_count                 = 1
        self.epochs                     = 70
        self.current_batch_size         = 2
        self.epochs_to_cycle            = 5
        self.image_size                 = img_size
        self.internal_representation    = internal_representation
        self.fnames_prefix              = 'testmodel'
        self.mask                       = np.load('./dataset_ge40North/mask_256.npy').astype(np.float32)


    def compose_convolutional_encoder(self, input_layer, layernames_postfix = ''):
        input_3channel = Concatenate(axis=-1, name='Encoder_Gray2RGB_vgg16' + layernames_postfix)([input_layer, input_layer, input_layer])
        conv_base = VGG16_local(weights='imagenet', include_top=False,
                                input_shape=(self.image_size[0], self.image_size[1], 3),
                                weights_path='./vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                layernames_postfix=layernames_postfix)
        vggoutput = conv_base(input_3channel)
        VGGout = Flatten()(vggoutput)
        return VGGout,K.int_shape(vggoutput)[-1]


    def compose_convolutional_decoder(self, input_layer, k_size, channels, tag=''):
        decoder_reshape = Reshape((k_size, k_size, channels), name='%s_Decoder_Reshape_01' % tag)(input_layer)
        Decoder_UpSampling2D_01 = UpSampling2D(size=(2, 2), name='%s_Decoder_UpSampling2D_01' % tag)(decoder_reshape)
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
        hgtVGGout,outVGGchannels = self.compose_convolutional_encoder(hgt_input, layernames_postfix='_hgt')
        singleVGGout_length = K.int_shape(hgtVGGout)[1]

        maskbatch_input = Input(shape=tuple(list(self.image_size) + [1]), name='mask_batch_input')

        Flatten_length = K.int_shape(hgtVGGout)[1]
        k_size = np.int32(np.sqrt(singleVGGout_length / outVGGchannels))
        Encoder_Dense_01 = Dense(1024, activation='relu', name='Encoder_Dense_01')(hgtVGGout)
        Encoder_Dense_02 = Dense(256, activation='relu', name='Encoder_Dense_02')(Encoder_Dense_01)

        z_mean = Dense(self.internal_representation, activation='sigmoid', name='z_mean')(Encoder_Dense_02)
        z_log_var = Dense(self.internal_representation, activation='relu', name='z_log_var')(Encoder_Dense_02)

        z_combined = Concatenate(name='z_combined')([z_mean, z_log_var])

        z = Lambda(sampling, output_shape=(self.internal_representation,), name='z')([z_mean, z_log_var])

        encoder = Model(inputs=[hgt_input], outputs=[z_mean, z_log_var, z, z_combined])
        for l in encoder.layers:
            if 'vgg16' in l.name:
                l.trainable = False
            else:
                l.trainable = True

        plot_model(encoder, to_file='./output/' + self.fnames_prefix + '_Encoder_model_structure.png', show_shapes=True)
        with open('./output/' + self.fnames_prefix + '_Encoder_model_structure.txt', 'w') as fh:
            encoder.summary(print_fn=lambda x: fh.write(x + '\n'))

        Decoder_Dense_01 = Dense(256, activation='relu', name='Decoder_Dense_01')(z)
        Decoder_Dense_02 = Dense(1024, activation='relu', name='Decoder_Dense_02')(Decoder_Dense_01)
        Decoder_Dense_03 = Dense(Flatten_length, activation='relu', name='Decoder_Dense_03')(Decoder_Dense_02)

        hgt_cnn_decoder = self.compose_convolutional_decoder(Decoder_Dense_03, k_size, outVGGchannels, tag='HGT')

        Decoder_hgt_MaskedOutput = Multiply(name='Decoder_hgt_MaskedOutput')([hgt_cnn_decoder, maskbatch_input])

        model = Model([hgt_input, maskbatch_input], [Decoder_hgt_MaskedOutput, z_combined], name='testmodel')
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


        metrics = {'Decoder_hgt_MaskedOutput': [SSIMMetric(tag = 'HGT'), custom_MSE, MSSSIMMetric(average=False,
                                                                                                  debug=True,
                                                                                                  log_directory = './logs/',
                                                                                                  tag = 'HGT')]}

        model = self.get_cae(metrics)
        print("got CVAE")

        beta = K.variable(value=0.1)
        hgt_reconstruction_weight       = K.variable(value=1.0)
        z_combined_weight = K.variable(value=200.)
        def WeightsWarmup_schedule(epoch):
            print('\n\nepoch_number = %d' % epoch)
            K.set_value(hgt_reconstruction_weight, (0.3 + np.exp(-epoch)))
            K.set_value(z_combined_weight, (1000. - 500. * np.exp(-0.1 * epoch)))


        WeightsSchedulingCallback = LambdaCallback(on_epoch_end=lambda epoch, log: WeightsWarmup_schedule(epoch))



        model.compile(optimizer=Adam(lr=1e-4),
                      loss={'Decoder_hgt_MaskedOutput': 'mse',
                            'z_combined': SpVAE_loss(rho=0.1,
                                                     beta=beta,
                                                     logfile='./logs/' + self.fnames_prefix + '_SpVAE_loss_debug.log')},
                      loss_weights={'Decoder_hgt_MaskedOutput': hgt_reconstruction_weight,
                                    'z_combined': z_combined_weight},
                      metrics=metrics)




        print("loading data")
        datagen = DataManager(srcdatafiles={'hgt': './dataset_ge40North/hgt/res256x256/hgt_data_projected.npy'},
                              normdata_files = {'mwsp': './dataset_ge40North/wsp_normalization_vales.pkl'},
                              SpCVAE_internal_representation = self.internal_representation,
                              input_vars=['hgt'],
                              output_vars=['hgt'],
                              mask_file='./dataset_ge40North/mask_256.npy',
                              val_folds=5,
                              test_split_ratio=0.2,
                              image_size=self.image_size)

        train_generator = datagen.flow(None, batch_size=self.current_batch_size, category='train')
        val_generator = datagen.flow(None, batch_size=self.current_batch_size, category='val')

        lrCallback = LearningRateScheduler(step_decay, verbose=1)

        print('\n\n\n\n\n\n')
        print('========================================')
        print('Fitting model - WARMUP (VGG is fixed)...')
        print('========================================')
        print('\n\n\n\n\n\n')

        start = time.time()

        # warmup for 20 epochs
        warmup_epochs = 3
        history = model.fit_generator(train_generator,
                                      steps_per_epoch = 7,
                                      epochs = warmup_epochs,
                                      validation_data = val_generator,
                                      validation_steps = 10,
                                      callbacks=[lrCallback, WeightsSchedulingCallback])

        model.save('./logs/' + self.fnames_prefix + '_checkpoint_after_warmup.h5', include_optimizer=False)

        print('\n\n\n\n\n\n')
        print('====================================================')
        print('Fitting the model - all the weights are optimized...')
        print('====================================================')
        print('\n\n\n\n\n\n')

        # change all weights to trainable and continue training
        for l in model.layers:
            l.trainable = True

        #re-compile the model
        beta = K.variable(value=0.1) # !!!!!! change the new starting value here !!!!!
        hgt_reconstruction_weight = K.variable(value=1.0)
        z_combined_weight = K.variable(value=200.)

        def WeightsWarmup_schedule_after_warmup(epoch):
            print('\n\nepoch_number = %d' % (epoch+warmup_epochs))
            K.set_value(hgt_reconstruction_weight, (0.3 + np.exp(-(epoch+warmup_epochs))))
            K.set_value(z_combined_weight, (1000. - 500. * np.exp(-0.1 * (epoch+warmup_epochs))))

        WeightsSchedulingCallback = LambdaCallback(on_epoch_end=lambda epoch, log: WeightsWarmup_schedule_after_warmup(epoch))

        model.compile(optimizer=Adam(lr=1e-4),
                      loss={'Decoder_hgt_MaskedOutput': 'mse',
                            'z_combined': SpVAE_loss(rho=0.1,
                                                     beta=beta,
                                                     logfile='./logs/' + self.fnames_prefix + '_SpVAE_loss_debug.log')},
                      loss_weights={'Decoder_hgt_MaskedOutput': hgt_reconstruction_weight,
                                    'z_combined': z_combined_weight},
                      metrics=metrics)

        with open('./output/' + self.fnames_prefix + '_model_structure_unfreezed.txt', 'w') as fh:
            model.summary(print_fn=lambda x: fh.write(x + '\n'))


        history2 = model.fit_generator(train_generator,
                                       steps_per_epoch = 7,
                                       epochs = 3,
                                       validation_data = val_generator,
                                       validation_steps = 10,
                                       callbacks=[lrCallback, WeightsSchedulingCallback])



        end = time.time()







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
        myCVAE = mkSpCVAE(img_size = images_size,
                          internal_representation = internal_representation,
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
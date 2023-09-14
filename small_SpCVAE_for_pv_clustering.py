import time, getopt, pdb, os, gc, sys
import numpy as np
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Dense, Flatten, Reshape, Lambda, Multiply, Concatenate, Conv2DTranspose
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger, EarlyStopping
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
srcdatafile = './dataset_ge40North/PV_hgt_data_DJF_256_normed.npy'
# internal_representations = [2,3,8,16,32,64,128]


class mkSpCVAE(object):
    def __init__(self, img_size, internal_representation = 128, start_num = 0):

        self.start_num                  = start_num
        # self.gpu_num                    = gpu_num

        self.debug                      = True

        # MKPC
        self.GPUs_count                 = 1
        self.epochs                     = 1
        self.current_batch_size         = 2
        # self.current_batch_size         = 16 * self.GPUs_count

        self.epochs_to_cycle            = 10
        self.image_size                 = img_size
        self.internal_representation    = internal_representation
        self.fnames_prefix              = 'smallSpCVAE_4pv_clustering_hiddim%04d_startnum%02d' % (self.internal_representation, start_num)
        self.mask                       = np.load('./dataset_ge40North/mask_256.npy').astype(np.float32)




    def compose_convolutional_encoder(self, input_layer, tag='', layernames_postfix = ''):
        x = Conv2D(8, (3, 3), activation='relu', padding='same', name='block1_conv1' + layernames_postfix)(input_layer)
        x = Conv2D(8, (3, 3), strides=(2, 2), activation='relu', padding='same', name='block1_conv2' + layernames_postfix)(input_layer)
        x = Conv2D(16, (3, 3), activation='relu', padding='same', name='block2_conv1' + layernames_postfix)(x)
        x = Conv2D(16, (3, 3), strides=(2, 2), activation='relu', padding='same', name='block2_conv2' + layernames_postfix)(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block3_conv1' + layernames_postfix)(x)
        x = Conv2D(32, (3, 3), strides=(2,2), activation='relu', padding='same', name='block3_conv2' + layernames_postfix)(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block4_conv1' + layernames_postfix)(x)
        x = Conv2D(64, (3, 3), strides=(2,2), activation='relu', padding='same', name='block4_conv2' + layernames_postfix)(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block5_conv1' + layernames_postfix)(x)
        x = Conv2D(64, (3, 3), strides=(2,2), activation='relu', padding='same', name='block5_conv2' + layernames_postfix)(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block6_conv1' + layernames_postfix)(x)
        x = Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same', name='block6_conv2' + layernames_postfix)(x)
        return x

    def compose_convolutional_decoder(self, input_layer, k_size, tag=''):
        Decoder_Reshape_01 = Reshape((k_size, k_size, 64), name='%s_Decoder_Reshape_01' % tag)(input_layer)
        Decoder_UpSampling2D_01 = UpSampling2D(size=(2, 2), name='%s_Decoder_UpSampling2D_01' % tag)(Decoder_Reshape_01)
        Decoder_Conv2DTranspose_0101 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='%s_Decoder_Conv2DTranspose_0101' % tag)(Decoder_UpSampling2D_01)
        Decoder_UpSampling2D_02 = UpSampling2D(size=(2, 2), name='%s_Decoder_UpSampling2D_02' % tag)(Decoder_Conv2DTranspose_0101)
        Decoder_Conv2DTranspose_0102 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='%s_Decoder_Conv2DTranspose_0102' % tag)(Decoder_UpSampling2D_02)
        Decoder_UpSampling2D_03 = UpSampling2D(size=(2, 2), name='%s_Decoder_UpSampling2D_03' % tag)(Decoder_Conv2DTranspose_0102)
        Decoder_Conv2DTranspose_0201 = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='%s_Decoder_Conv2DTranspose_0201' % tag)(Decoder_UpSampling2D_03)
        Decoder_UpSampling2D_04 = UpSampling2D(size=(2, 2), name='%s_Decoder_UpSampling2D_04' % tag)(Decoder_Conv2DTranspose_0201)
        Decoder_Conv2DTranspose_0301 = Conv2D(16, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='%s_Decoder_Conv2DTranspose_0301' % tag)(Decoder_UpSampling2D_04)
        Decoder_UpSampling2D_05 = UpSampling2D(size=(2, 2), name='%s_Decoder_UpSampling2D_05' % tag)(Decoder_Conv2DTranspose_0301)
        Decoder_Conv2DTranspose_0401 = Conv2D(8, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='%s_Decoder_Conv2DTranspose_0401' % tag)(Decoder_UpSampling2D_05)
        Decoder_UpSampling2D_06 = UpSampling2D(size=(2, 2), name='%s_Decoder_UpSampling2D_06' % tag)(Decoder_Conv2DTranspose_0401)
        Decoder_Conv2DTranspose_0501 = Conv2D(4, (3, 3), strides=1, activation='relu', padding='same', kernel_initializer='he_normal', name='%s_Decoder_Conv2DTranspose_0501' % tag)(Decoder_UpSampling2D_06)
        Decoder_Conv2DTranspose_07_out = Conv2D(1, (3, 3), strides=1, activation='sigmoid', padding='same', name='%s_Decoder_Conv2DTranspose_07_out' % tag)(Decoder_Conv2DTranspose_0501)
        return Decoder_Conv2DTranspose_07_out


    def ComposeModel_singleGPU(self, metrics = None):

        hgt_input = Input(shape=tuple(list(self.image_size) + [1]), name='HGT_Encoder_Input_vgg16')
        # hgt_3channel = Concatenate(axis=-1)([hgt_input,hgt_input,hgt_input])
        pv_input = Input(shape=tuple(list(self.image_size) + [1]), name='PV_Encoder_Input_vgg16')
        # pv_3channel = Concatenate(axis=-1)([pv_input, pv_input, pv_input])
        pvth_input = Input(shape=tuple(list(self.image_size) + [1]), name='PVth_Encoder_Input_vgg16')
        # pvth_3channel = Concatenate(axis=-1)([pvth_input, pvth_input, pvth_input])
        maskbatch_input = Input(shape=tuple(list(self.image_size) + [1]), name='mask_batch_input')
        # maskbatch_3channel = Concatenate(axis=-1)([maskbatch_input, maskbatch_input, maskbatch_input])



        vggbase_hgt = self.compose_convolutional_encoder(hgt_input, tag='HGT', layernames_postfix = '_hgt')
        vggbase_pv = self.compose_convolutional_encoder(pv_input, tag='PV', layernames_postfix='_pv')
        vggbase_pvth = self.compose_convolutional_encoder(pvth_input, tag='PVth', layernames_postfix='_pvth')

        vggbase_hgt.trainable = False
        vggbase_pv.trainable = False
        vggbase_pvth.trainable = False
        hgtVGGout = Flatten()(vggbase_hgt)
        pvVGGout = Flatten()(vggbase_pv)
        pvthVGGout = Flatten()(vggbase_pvth)
        singleVGGout_length = K.int_shape(hgtVGGout)[1]
        VGGout = Concatenate(axis=-1)([hgtVGGout, pvVGGout, pvthVGGout])

        Flatten_length = K.int_shape(VGGout)[1]
        Encoder_Dropout03 = Dropout(0.2, name='Encoder_Dropout03')(VGGout)
        Encoder_Dense_01 = Dense(1024, activation='relu', name='Encoder_Dense_01')(Encoder_Dropout03)
        Encoder_Dropout04 = Dropout(0.2, name='Encoder_Dropout04')(Encoder_Dense_01)
        Encoder_Dense_02 = Dense(128, activation='relu', name='Encoder_Dense_02')(Encoder_Dropout04)

        # wind speed regression part
        wsp_reg_Dropout01 = Dropout(0.2, name='wsp_reg_Dropout01')(Encoder_Dense_02)
        wsp_reg_Dense01 = Dense(96, activation='relu', name='wsp_reg_Dense01')(wsp_reg_Dropout01)
        wsp_reg_Dropout02 = Dropout(0.2, name='wsp_reg_Dropout02')(wsp_reg_Dense01)
        wsp_reg_Dense02 = Dense(24, activation='relu', name='wsp_reg_Dense02')(wsp_reg_Dropout02)
        wsp_reg_Dense03 = Dense(12, activation='relu', name='wsp_reg_Dense03')(wsp_reg_Dense02)
        wsp_reg_output = Dense(1, activation='relu', name='wsp_reg_output')(wsp_reg_Dense03)


        z_mean = Dense(self.internal_representation, activation='sigmoid', name='z_mean')(Encoder_Dense_02)
        z_log_var = Dense(self.internal_representation, activation='relu', name='z_log_var')(Encoder_Dense_02)

        z_combined = Concatenate(name='z_combined')([z_mean, z_log_var])

        z = Lambda(sampling, output_shape=(self.internal_representation,), name='z')([z_mean, z_log_var])

        encoder = Model(inputs=[hgt_input, pv_input, pvth_input], outputs=[z_mean, z_log_var, z, z_combined])
        # for l in encoder.layers:
        #     if 'vgg16' in l.name:
        #         l.trainable = False
        #     else:
        #         l.trainable = True

        plot_model(encoder, to_file='./output/' + self.fnames_prefix + '_Encoder_model_structure.png', show_shapes=True)
        with open('./output/' + self.fnames_prefix + '_Encoder_model_structure.txt', 'w') as fh:
            encoder.summary(print_fn=lambda x: fh.write(x + '\n'))

        Decoder_Dense_01 = Dense(128, activation='relu', name='Decoder_Dense_01')(z)
        Decoder_drop_01 = Dropout(0.2, name='Decoder_drop_01')(Decoder_Dense_01)
        Decoder_Dense_02 = Dense(1024, activation='relu', name='Decoder_Dense_02')(Decoder_drop_01)
        Decoder_Dropout02 = Dropout(0.2, name='Decoder_Dropout02')(Decoder_Dense_02)
        Decoder_Dense_03 = Dense(Flatten_length, activation='relu', name='Decoder_Dense_03')(Decoder_Dropout02)

        k_size = np.int32(np.sqrt(Flatten_length / (3 * 64)))

        hgt_flattened, pv_flattened, phth_flattened = Lambda(lambda x: (x[:,:singleVGGout_length], x[:,singleVGGout_length:2*singleVGGout_length], x[:,2*singleVGGout_length:]))(Decoder_Dense_03)
        hgt_cnn_decoder = self.compose_convolutional_decoder(hgt_flattened, k_size, tag='HGT')
        pv_cnn_decoder = self.compose_convolutional_decoder(pv_flattened, k_size, tag='PV')
        pvth_cnn_decoder = self.compose_convolutional_decoder(phth_flattened, k_size, tag='PVth')

        Decoder_hgt_MaskedOutput = Multiply(name='Decoder_hgt_MaskedOutput')([hgt_cnn_decoder, maskbatch_input])
        Decoder_pv_MaskedOutput = Multiply(name='Decoder_pv_MaskedOutput')([pv_cnn_decoder, maskbatch_input])
        Decoder_pvth_MaskedOutput = Multiply(name='Decoder_pvth_MaskedOutput')([pvth_cnn_decoder, maskbatch_input])



        model = Model([hgt_input, pv_input, pvth_input, maskbatch_input], [Decoder_hgt_MaskedOutput, Decoder_pv_MaskedOutput, Decoder_pvth_MaskedOutput, wsp_reg_output, z_combined], name='pv_SpCVAE')
        # for l in model.layers:
        #     if 'vgg16' in l.name:
        #         l.trainable = False
        #     else:
        #         l.trainable = True

        plot_model(model, to_file='./output/' + self.fnames_prefix + '_model_structure.png', show_shapes=True)
        with open('./output/' + self.fnames_prefix + '_model_structure.txt', 'w') as fh:
            model.summary(print_fn=lambda x: fh.write(x + '\n'))

        self.template_model = model
        self.encoder = encoder

        model.compile(optimizer=Adam(lr=1e-4),
                      loss={'Decoder_hgt_MaskedOutput': 'mse',
                            'Decoder_pv_MaskedOutput': 'mse',
                            'Decoder_pvth_MaskedOutput': 'binary_crossentropy',
                            'z_combined': SpVAE_loss(rho=0.1,
                                                     beta=0.1,
                                                     debug=True,
                                                     logfile='./logs/' + self.fnames_prefix + '_SpVAE_loss_debug.log'),
                            'wsp_reg_output': 'mse'},
                      loss_weights={'Decoder_hgt_MaskedOutput': 1.,
                                    'Decoder_pv_MaskedOutput': 1.,
                                    'Decoder_pvth_MaskedOutput': 1.,
                                    'z_combined': 1.,
                                    'wsp_reg_output': 1.},
                      metrics=metrics)
        return model




    def get_cae(self, metrics = None):
        if self.GPUs_count <= 1:
            return self.ComposeModel_singleGPU(metrics=metrics)



    def train(self):
        EnsureDirectoryExists('./output/')
        EnsureDirectoryExists('./logs/')



        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement = log_device_placement, device_count={'CPU': 1, 'GPU': self.GPUs_count})
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        K.set_session(session)


        metrics = {'Decoder_hgt_MaskedOutput': [SSIMMetric(tag = 'HGT'), custom_MSE, MSSSIMMetric(average=False,
                                                                                                  debug=True,
                                                                                                  log_directory = './logs/',
                                                                                                  tag = 'HGT')],
                   'Decoder_pv_MaskedOutput': [SSIMMetric(tag = 'PV'), custom_MSE, MSSSIMMetric(average=False,
                                                                                                debug=True,
                                                                                                log_directory = './logs/',
                                                                                                tag = 'PV')],
                   'Decoder_pvth_MaskedOutput': [SSIMMetric(tag = 'PVth'), custom_MSE, MSSSIMMetric(average=False,
                                                                                                    debug=True,
                                                                                                    log_directory = './logs/',
                                                                                                    tag = 'PVth')]}

        model = self.get_cae(metrics)
        print("got CVAE")

        print("loading data")
        datagen = DataManager(srcdatafiles={'hgt': './dataset_ge40North/hgt/res256x256/hgt_data_projected.npy',
                                            'pv': './dataset_ge40North/pv/res256x256/pv_data_projected.npy',
                                            'pv_th': './dataset_ge40North/pv_thresholded/res256x256/pv_thresholded_data_projected.npy',
                                            'mwsp': './dataset_ge40North/mean_wind_speed_along_jet.npy'},
                              normdata_files={'mwsp': './dataset_ge40North/wsp_normalization_vales.pkl'},
                              SpCVAE_internal_representation = self.internal_representation,
                              input_vars=['hgt', 'pv', 'pv_th'],
                              output_vars=['hgt', 'pv', 'pv_th', 'mwsp'],
                              mask_file='./dataset_ge40North/mask_256.npy',
                              val_folds=5,
                              test_split_ratio=0.2,
                              image_size=self.image_size)

        train_generator = datagen.flow(None, batch_size=self.current_batch_size, category='train')
        val_generator = datagen.flow(None, batch_size=self.current_batch_size, category='val')

        def msssim_monitor(logs):
            return np.sum([logs['val_%s_MSSSIMMetric_%s' % (outputname,tag)] for (tag,outputname) in zip(['HGT', 'PV', 'PVth'], ['Decoder_hgt_MaskedOutput', 'Decoder_pv_MaskedOutput', 'Decoder_pvth_MaskedOutput'])])
        def mse_monitor(logs):
            return np.sum([logs['val_%s_custom_MSE' % outputname] for outputname in ['Decoder_hgt_MaskedOutput', 'Decoder_pv_MaskedOutput', 'Decoder_pvth_MaskedOutput']])
        def ssim_monitor(logs):
            return np.sum([logs['val_%s_SSIMMetric_%s' % (outputname, tag)] for (tag, outputname) in zip(['HGT', 'PV', 'PVth'], ['Decoder_hgt_MaskedOutput', 'Decoder_pv_MaskedOutput', 'Decoder_pvth_MaskedOutput'])])

        filepathMSSSIMMetric = './logs/' + self.fnames_prefix + '_checkpoint_ep{epoch:04d}_valMSSSIM_{monitorvalue:.6f}.hdf5'
        checkpointingMSSSIMMetric = CustomModelCheckpoint(self.template_model, filepathMSSSIMMetric, monitor=msssim_monitor, verbose=1, save_best_only = False, mode='min')
        filepathLossMetric = './logs/' + self.fnames_prefix + '_epoch{epoch:04d}-valLOSS_{monitorvalue:.6f}.hdf5'
        checkpointingLoss = CustomModelCheckpoint(self.template_model, filepathLossMetric, monitor='val_loss', verbose=1, save_best_only = False, mode='min')
        cycle_train_val_sets = KFoldCycleCallback(datagen, epochs_to_cycle = self.epochs_to_cycle)
        csv_logger = CSVLogger('./logs/' + self.fnames_prefix + 'train_progress.csv', separator=';', append=True, )
        tb_callback = CustomTensorBoard(log_dir='./logs/TBoard/' + self.fnames_prefix + '/', write_graph=True, update_freq='epoch')
        lrCallback = LearningRateScheduler(step_decay, verbose=1)
        EarlyStoppingCallback = EarlyStopping(monitor='val_DecoderMaskedOutput_MSSSIMMetric', min_delta=0., patience=100, verbose=1, mode='max')
        # InterVis = IntermediateVisualizations(png_filepath_template         = './logs/' + self.fnames_prefix + '_encodings_epoch{epoch:04d}.png',
        #                                       decoded_filepath_template     ='./logs/' + self.fnames_prefix + '_decoded_epoch{epoch:04d}.npy',
        #                                       testdata_filename_template    ='./logs/' + self.fnames_prefix + '_testdata.npy',
        #                                       models_to_eval                = (self.encoder, self.template_model),
        #                                       mask                          = self.mask,
        #                                       testdata                      = np.expand_dims(datagen.srcdata[datagen.test_indices], -1),
        #                                       varname                       = self.varname,
        #                                       period                        = 1,
        #                                       batch_size                    = self.current_batch_size)

        print('Fitting model...')

        start = time.time()


        history = model.fit_generator(train_generator,
                                      # steps_per_epoch = datagen.dataset_length_batches(category='train', batch_size = self.current_batch_size),
                                      steps_per_epoch = 10,
                                      # epochs = self.epochs,
                                      epochs = 3,
                                      validation_data = val_generator,
                                      # validation_steps = datagen.dataset_length_batches(category='val', batch_size=self.current_batch_size),
                                      validation_steps = 10,
                                      callbacks=[checkpointingMSSSIMMetric, checkpointingLoss, csv_logger, tb_callback, cycle_train_val_sets, lrCallback, EarlyStoppingCallback])




        end = time.time()

        with open('./output/' + self.fnames_prefix + 'model.json', "w") as json_file:
            json_file.write(self.template_model.to_json())
        self.template_model.save('./output/'+ self.fnames_prefix +'weights.h5', include_optimizer=False)

        print('loading test data')
        test_generator = datagen.flow(None, batch_size=self.current_batch_size, category='test')

        print('evaluating final model...')
        final_eval_result = model.evaluate_generator(test_generator, steps=datagen.dataset_length_batches(category='test', batch_size=self.current_batch_size))
        print(final_eval_result)

        K.clear_session()
        gc.collect()

        print("training time for %d epochs: %.2fs = %.2fHrs" % (len(list(history.history.values())[0]), (end - start), (end - start) / 3600.0))
        with open('./logs/' + self.fnames_prefix + 'train_summary.txt', 'w') as f:
            f.writelines("training time for %d epochs: %.2fs = %.2fHrs" % (len(list(history.history.values())[0]), (end - start), (end - start) / 3600.0))
            f.writelines('\n\nevaluating final model:')
            f.writelines(str(final_eval_result))







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
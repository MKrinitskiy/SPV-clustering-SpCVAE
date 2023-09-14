import tensorflow as tf
from tensorflow.compat.v1 import disable_eager_execution
from libs.parse_args import *
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, Nadam
from os.path import join, isfile, isdir
from libs.copy_tree import copytree_multi
from ModelCreator import ModelCreator
from ssim_metric import *
from msssim_metric import *
from support_defs import *
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, LearningRateScheduler, CSVLogger, TensorBoard
from DataGeneratorImgaug import *
from CustomModelCheckpoint import *
from libs.SGDRScheduler import SGDRScheduler
from break_callback import BreakLearningCallback
from IntermediateVisualizations_Callback import IntermediateVisualizations
import time
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'



def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    curr_run_name = args.run_name
    EPOCHS = args.epochs
    if 'steps_per_epoch' in args:
        STEPS_PER_EPOCH = args.steps_per_epoch
    else:
        STEPS_PER_EPOCH = None

    if 'val_steps' in args:
        VAL_STEPS = args.val_steps
    else:
        VAL_STEPS = None



    try:
        EnsureDirectoryExists('./logs/%s/' % curr_run_name)
    except:
        print('logs directory couldn`t be found and couldn`t be created:\n./logs/%s/' % curr_run_name)
        raise FileNotFoundError('logs directory couldn`t be found and couldn`t be created:\n./logs/%s/' % curr_run_name)

    try:
        EnsureDirectoryExists('./scripts_backup/%s/' % (curr_run_name))
    except:
        print('backup directory couldn`t be found and couldn`t be created:\n./logs/%s/scripts_backup/' % curr_run_name)
        raise FileNotFoundError('backup directory couldn`t be found and couldn`t be created:\n./logs/%s/scripts_backup/' % curr_run_name)



    #region backing up the scripts configuration
    ignore_func = lambda dir, files: [f for f in files if (isfile(join(dir, f)) and f[-3:] != '.py')] + [d for d in files if ((isdir(d)) & (d.endswith('scripts_backup') |
                                                                                                                                            d.endswith('.ipynb_checkpoints') |
                                                                                                                                            d.endswith('__pycache__') |
                                                                                                                                            d.endswith('build') |
                                                                                                                                            d.endswith('logs') |
                                                                                                                                            d.endswith('snapshots')))]
    copytree_multi('./',
                   './scripts_backup/%s/' % curr_run_name,
                   ignore=ignore_func)
    #endregion backing up the scripts configuration

    image_size = (256, 256)
    current_batch_size = args.batch_size
    bottleneck_dims = args.embeddims

    # optionally choose specific GPU
    # if args.gpu:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                      log_device_placement=False,
                                      device_count={'CPU': 1, 'GPU': 1})
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)



    model_creator = ModelCreator(image_size, curr_run_name=curr_run_name, variational=args.variational, debug=args.debug, bottleneck_dims=bottleneck_dims)
    # model = model_creator.ComposeModel()
    model = model_creator.ComposeModel_VGGlike()
    print("got CVAE")
    # quit()


    # def beta_value_vs_epoch(ep):
    #     return 1.
    #     # return (1. - 0.9 * np.exp(-0.1 * ep))
    #
    # def hgt_reconstruction_weight_vs_epoch(ep):
    #     # return (0.8 + 0.4*np.exp(-0.2*ep))
    #     return (image_size[0] * image_size[1])
    #
    # def pv_reconstruction_weight_vs_epoch(ep):
    #     # return (0.8 + 0.4*np.exp(-0.2*ep))
    #     return (image_size[0] * image_size[1])
    #
    # # def pvth_reconstruction_weight_vs_epoch(ep):
    # #     return (30 + 20.*np.exp(-0.1*ep))
    # def z_combined_weight_vs_epoch(ep):
    #     # return (0.8 - 1.0 * np.exp(-0.3 * ep) + 0.5 * np.exp(-0.1 * ep))
    #     return current_batch_size
    #
    # def wspregroutput_combined_weight_vs_epoch(ep):
    #     # return (0.5 - 0.5 * np.exp(-0.05 * ep))
    #     return current_batch_size

    # beta = K.variable(value=beta_value_vs_epoch(0))
    # hgt_reconstruction_weight = K.variable(value=hgt_reconstruction_weight_vs_epoch(0))
    # pv_reconstruction_weight = K.variable(value=pv_reconstruction_weight_vs_epoch(0))
    # pvth_reconstruction_weight      = K.variable(value=pvth_reconstruction_weight_vs_epoch(0))
    # z_combined_weight = K.variable(value=z_combined_weight_vs_epoch(0))
    # wspregroutput_combined_weight = K.variable(value=wspregroutput_combined_weight_vs_epoch(0))

    # def WeightsWarmup_schedule(epoch):
    #     beta_value = beta_value_vs_epoch(epoch)
    #     print("VAE KL loss beta: %f" % beta_value)
    #     K.set_value(beta, beta_value)
    #     K.set_value(hgt_reconstruction_weight, hgt_reconstruction_weight_vs_epoch(epoch))
    #     K.set_value(pv_reconstruction_weight, pv_reconstruction_weight_vs_epoch(epoch))
    #     # K.set_value(pvth_reconstruction_weight, pvth_reconstruction_weight_vs_epoch(epoch))
    #     K.set_value(z_combined_weight, z_combined_weight_vs_epoch(epoch))
    #     K.set_value(wspregroutput_combined_weight, wspregroutput_combined_weight_vs_epoch(epoch))

    # WeightsSchedulingCallback = LambdaCallback(on_epoch_end=lambda epoch, log: WeightsWarmup_schedule(epoch))

    optimizer = tf.optimizers.Adam(learning_rate=1.e-3, decay=0.95, amsgrad=True)
    # optimizer = Nadam(learning_rate=tf.keras.experimental.CosineDecay(1.e-4, decay_steps=10, alpha=1.e-5))

    # metrics = {'Decoder_hgt_MaskedOutput': [SSIMMetric(tag='HGT'),
    #                                         keras.metrics.MSE,
    #                                         MSSSIMMetric(average=False, debug=True, log_directory='./logs/', tag='HGT')],
    #            'Decoder_pv_MaskedOutput': [SSIMMetric(tag='PV'),
    #                                        keras.metrics.MSE,
    #                                        MSSSIMMetric(average=False, debug=True, log_directory='./logs/', tag='PV')]}
    metrics = {'Decoder_MaskedOutput': [SSIMMetric(tag=''),
                                        keras.metrics.MSE,
                                        MSSSIMMetric(average=False, debug=True, log_directory='./logs/', tag='')]}

    if args.variational:
        # model.compile(optimizer=optimizer,
        #               loss={'Decoder_hgt_MaskedOutput': keras.losses.mean_squared_error,
        #                     'Decoder_pv_MaskedOutput': keras.losses.mean_squared_error,
        #                     'z_combined': SpVAE_loss(rho=0.1,
        #                                              beta=beta,
        #                                              logfile='./logs/%s/SpVAE_loss_debug.log' % curr_run_name)},
        #               loss_weights={'Decoder_hgt_MaskedOutput': hgt_reconstruction_weight,
        #                             'Decoder_pv_MaskedOutput': pv_reconstruction_weight,
        #                             'z_combined': z_combined_weight},
        #               metrics=metrics)
        model.compile(optimizer=optimizer,
                      loss={'Decoder_MaskedOutput': keras.losses.mean_squared_error,
                            'z_combined': SpVAE_loss(beta=0.0, logfile='./logs/%s/SpVAE_loss_debug.log' % curr_run_name)},
                      loss_weights={'Decoder_MaskedOutput': (image_size[0] * image_size[1]),
                                    'z_combined': current_batch_size},
                      metrics=metrics)
    else:
        # model.compile(optimizer=optimizer,
        #     loss={'Decoder_hgt_MaskedOutput': keras.losses.mean_squared_error,
        #           'Decoder_pv_MaskedOutput': keras.losses.mean_squared_error},
        #     loss_weights={'Decoder_hgt_MaskedOutput': hgt_reconstruction_weight,
        #                   'Decoder_pv_MaskedOutput': pv_reconstruction_weight},
        #     metrics=metrics)
        model.compile(optimizer=optimizer,
                      loss={'Decoder_MaskedOutput': keras.losses.mean_squared_error},
                      loss_weights={'Decoder_MaskedOutput': 1.0},
                      metrics=metrics)

    print("loading data")
    # datagen = DataManager(srcdatafiles={'hgt': '../dataset_ge40North/hgt/hgt_data_projected_all.normed01.npy',
    #                                     'pv': '../dataset_ge40North/pv/pv_data_projected_all.normed01.npy'},
    #                       normdata_files={},
    #                       variational_model = args.variational,
    #                       input_vars=['hgt', 'pv'],
    #                       expand_dims_for_channel=['hgt', 'pv'],
    #                       imgaugment_vars=['hgt', 'pv'],
    #                       output_vars=['hgt', 'pv'],
    #                       mask_file='../dataset_ge40North/mask_256.npy',
    #                       mmap_vars=['hgt', 'pv'],
    #                       val_folds=5,
    #                       test_split_ratio=None,
    #                       image_size=image_size)
    datagen = DataManager(srcdatafiles={'hgt': '../dataset_ge40North/hgt/hgt_data_projected_all.normed01.npy'},
                          normdata_files={},
                          variational_model=args.variational,
                          input_vars=['hgt'],
                          expand_dims_for_channel=['hgt'],
                          imgaugment_vars=['hgt'],
                          output_vars=['hgt'],
                          mask_file='../dataset_ge40North/mask_256.npy',
                          mmap_vars=['hgt'],
                          val_folds=5,
                          test_split_ratio=None,
                          image_size=image_size)
    train_generator = datagen.flow(None, batch_size=current_batch_size, category='train')
    val_generator = datagen.flow(None, batch_size=current_batch_size, category='val')

    if STEPS_PER_EPOCH is None:
        STEPS_PER_EPOCH = datagen.dataset_length_batches(category='train', batch_size=current_batch_size)

    if VAL_STEPS is None:
        VAL_STEPS = datagen.dataset_length_batches(category='val', batch_size=current_batch_size)

    # def msssim_monitor(logs):
    #     return np.sum([logs['val_%s_MSSSIMMetric_%s' % (outputname, tag)] for (tag, outputname) in zip(['HGT', 'PV'], ['Decoder_hgt_MaskedOutput', 'Decoder_pv_MaskedOutput'])])
    # def msssim_monitor(logs):
    #     return logs['val_MSSSIMMetric_HGT']

    # def mse_monitor(logs):
    #     return np.sum([logs['val_%s_mean_squared_error' % outputname] for outputname in ['Decoder_hgt_MaskedOutput', 'Decoder_pv_MaskedOutput']])
    # def mse_monitor(logs):
    #     return np.sum([logs['val_%s_mean_squared_error' % outputname] for outputname in ['Decoder_MaskedOutput']])

    # filepathMSSSIMMetric = './logs/%s/checkpoint_ep{epoch:04d}_valMSSSIM_{monitorvalue:.6f}.hdf5' % curr_run_name
    # checkpointingMSSSIMMetric = CustomModelCheckpoint(model, filepathMSSSIMMetric, monitor=msssim_monitor, verbose=1, save_best_only=True, mode='min')
    # filepathLossMetric = './logs/%s/epoch{epoch:04d}-valLOSS_{monitorvalue:.6f}.hdf5' % curr_run_name
    # checkpointingLoss = CustomModelCheckpoint(model, filepathLossMetric, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # filepathLastCheckpoint = './logs/%s/checkpoint_epoch{epoch:04d}.hdf5' % curr_run_name
    # checkpointingLast = LastModelCheckpoint(model, filepathLastCheckpoint, verbose=1)

    filepathMetricsCheckpoint = './logs/%s/checkpoint_ep{epoch:04d}-{monitor_name}-{monitor_value:.6f}.hdf5' % curr_run_name
    filepathOverallCheckpoint = './logs/%s/checkpoint_ep{epoch:04d}.hdf5' % curr_run_name
    checkpointingCallback = CustomModelCheckpoint(model,
                                                  filepathMetricsCheckpoint,
                                                  filepathOverallCheckpoint,
                                                  monitors=['val_MSSSIMMetric',
                                                            'val_SSIMMetric',
                                                            'val_mean_squared_error',
                                                            'val_loss'],
                                                  modes=['min', 'min', 'min', 'min'])


    cycle_train_val_sets = KFoldCycleCallback(datagen, epochs_to_cycle = 5)
    csv_logger = CSVLogger('./logs/%s/train_progress.csv' % curr_run_name, separator=';', append=True)

    tb_callback = TensorBoard(log_dir='./logs/%s/TBoard/' % curr_run_name + 'warmup/', write_graph=True, update_freq=20)
    # InterVizCallback = IntermediateVisualizations(datagen, output_path='./logs/',
    #                                               model_prefix=self.fnames_prefix + '_warmup', model=model,
    #                                               mask=self.mask, picture_vars=['hgt', 'pv'], corr_vars=['reg_output'])
    stopping_callback = BreakLearningCallback(model, control_file='./logs/%s/break/triggerfile' % curr_run_name)

    InterVisCallback = IntermediateVisualizations(datagen,
                                                  output_path = './logs/%s/IntermediateVisualizations/' % curr_run_name,
                                                  model_prefix = '',
                                                  model = model,
                                                  mask = None,
                                                  picture_vars = ['hgt'],
                                                  corr_vars = [], )

    lr_schedule = SGDRScheduler(min_lr=1e-6,
                                max_lr=1e-3,
                                steps_per_epoch=datagen.dataset_length_batches(category='train'),
                                lr_decay=0.9,
                                cycle_length=3,
                                mult_factor=1.5,
                                logfile='./logs/%s/lr_history.pkl' % curr_run_name)

    start = time.time()

    history = model.fit_generator(train_generator,
                                  steps_per_epoch = STEPS_PER_EPOCH,
                                  epochs = EPOCHS,
                                  validation_data = val_generator,
                                  validation_steps = VAL_STEPS,
                                  callbacks=[checkpointingCallback, csv_logger, tb_callback, cycle_train_val_sets, stopping_callback, lr_schedule, InterVisCallback])
                                  # callbacks=[checkpointingCallback, csv_logger, tb_callback, cycle_train_val_sets, WeightsSchedulingCallback, stopping_callback])

    end = time.time()

    print('saving final weights...')

    model.save('./logs/%s/final_checkpoint.h5' % curr_run_name, include_optimizer=False)

    K.clear_session()

    print("training time for %d epochs: %.2fs = %.2fHrs" % (len(list(history.history.values())[0]), (end - start), (end - start) / 3600.0))
    with open('./logs/%s/train_summary.txt' % curr_run_name, 'w') as f:
        f.writelines("training time for %d epochs: %.2fs = %.2fHrs" % (len(list(history.history.values())[0]), (end - start), (end - start) / 3600.0))



if __name__ == '__main__':
    disable_eager_execution()
    main()

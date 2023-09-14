from tensorflow.python.keras.callbacks import Callback
import numpy as np
import os
from support_defs import EnsureDirectoryExists
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter
import cv2


class IntermediateVisualizations(Callback):
    def __init__(self,
                 datagenerator,
                 output_path,
                 model_prefix,
                 model,
                 mask,
                 picture_vars = ('hgt', 'pv'),
                 corr_vars = ('reg_output'),
                 period=1,
                 batch_size=32):

        super(IntermediateVisualizations, self).__init__()

        EnsureDirectoryExists(output_path)
        for var in picture_vars:
            EnsureDirectoryExists(os.path.join(output_path, var))
            EnsureDirectoryExists(os.path.join(output_path, var, 'fixed_examples'))
            EnsureDirectoryExists(os.path.join(output_path, var, 'fixed_examples', model_prefix))
            EnsureDirectoryExists(os.path.join(output_path, var))
            EnsureDirectoryExists(os.path.join(output_path, var, 'random_examples'))
            EnsureDirectoryExists(os.path.join(output_path, var, 'random_examples', model_prefix))
        for var in corr_vars:
            EnsureDirectoryExists(os.path.join(output_path, var))
            EnsureDirectoryExists(os.path.join(output_path, var, model_prefix))

        self.output_path                    = output_path
        self.datagen                        = datagenerator
        self.model_prefix                   = model_prefix
        self.model                          = model


        self.picture_vars                   = picture_vars
        self.corr_vars                      = corr_vars
        self.period                         = period
        self.prev_visialized_epoch          = None
        self.batch_size                     = batch_size

        val_data_datagen = self.datagen.flow(None,
                                             batch_size=self.batch_size,
                                             category='val',
                                             shuffle=False)

        fixed_samples_random_indices = np.random.randint(0, len(self.datagen.val_indices), 16)

        self.batch_x, self.batch_y = val_data_datagen._get_batches_of_transformed_samples(fixed_samples_random_indices)

        if mask:
            self.mask                       = mask
        else:
            self.mask = self.batch_x[-1][0,:,:,0]
        self.maskbool = np.invert(self.mask.astype(np.bool))

        self.datagen_inputs_indices = {}
        for idx, name in enumerate(self.datagen.input_vars):
            self.datagen_inputs_indices[name] = idx

        self.datagen_outputs_indices = {}
        for idx, name in enumerate(self.datagen.output_vars):
            self.datagen_outputs_indices[name] = idx




    def on_epoch_end(self, epoch, logs=None):
        if (self.prev_visialized_epoch is not None):
            if epoch - self.prev_visialized_epoch < self.period:
                return None

        print('====== visualizing intermediate results ======')

        preds = self.model.predict_on_batch(self.batch_x)
        for var in self.picture_vars:
            curr_png_fname = os.path.join(self.output_path, var, 'fixed_examples', self.model_prefix, 'ep%04d.png' % epoch)

            if isinstance(self.batch_y, np.ndarray):
                src_examples = np.squeeze(self.batch_y)
            elif ((isinstance(self.batch_y, tuple)) or (isinstance(self.batch_y, list))):
                idx = int(np.where(np.array(self.datagen.output_vars) == var)[0])
                src_examples = np.squeeze(self.batch_y[idx])

            if isinstance(preds, np.ndarray):
                curr_var_preds = np.squeeze(preds)
            elif ((isinstance(self.batch_y, tuple)) or (isinstance(self.batch_y, list))):
                idx = int(np.where(np.array(self.datagen.output_vars) == var)[0])
                curr_var_preds = np.squeeze(preds[idx])

            f = plt.figure(figsize=(12,7), dpi=300)
            for i in range(len(src_examples)):
                curr_src_example = np.ma.asarray(np.squeeze(src_examples[i]))
                curr_src_example.mask = self.maskbool
                curr_reconstructed_example = np.ma.asarray(curr_var_preds[i])
                curr_reconstructed_example.mask = self.maskbool

                p1 = plt.subplot(4,8,i*2+1)
                plt.imshow(curr_src_example, cmap='gray')
                plt.title('source')
                plt.axis('off')
                p1 = plt.subplot(4, 8, i*2+2)
                plt.imshow(curr_reconstructed_example, cmap='gray')
                plt.title('rec.-ed')
                plt.axis('off')
            plt.tight_layout()
            plt.savefig(curr_png_fname)
            plt.close(f)




        # batch_x, batch_y = all_data_datagen._get_batches_of_transformed_samples(np.random.randint(0, self.datagen.total_objects_count, 16))
        # preds = self.model.predict_on_batch(batch_x)
        # for var in self.picture_vars:
        #     curr_png_fname = os.path.join(self.output_path, var, 'random_examples', self.model_prefix, 'ep%04d.png' % epoch)
        #     src_examples = np.squeeze(batch_y[self.datagen_outputs_indices[var]])
        #     curr_var_preds = np.squeeze(preds[self.datagen_outputs_indices[var]])
        #     f = plt.figure(figsize=(12, 7), dpi=300)
        #     for i in range(src_examples.shape[0]):
        #         curr_src_example = np.ma.asarray(np.squeeze(src_examples[i]))
        #         curr_src_example.mask = self.maskbool
        #         curr_reconstructed_example = np.ma.asarray(curr_var_preds[i])
        #         curr_reconstructed_example.mask = self.maskbool
        #
        #         p1 = plt.subplot(4, 8, i * 2 + 1)
        #         plt.imshow(curr_src_example, cmap='gray')
        #         plt.title('source')
        #         plt.axis('off')
        #         p1 = plt.subplot(4, 8, i * 2 + 2)
        #         plt.imshow(curr_reconstructed_example, cmap='gray')
        #         plt.title('rec.-ed')
        #         plt.axis('off')
        #     plt.tight_layout()
        #     plt.savefig(curr_png_fname)
        #     plt.close(f)




        # batch_x, batch_y = all_data_datagen._get_batches_of_transformed_samples(np.random.randint(0, self.datagen.total_objects_count, 1024))
        # preds = self.model.predict(batch_x, batch_size=16, verbose=True)
        # for var in self.corr_vars:
        #     curr_png_fname = os.path.join(self.output_path, var, self.model_prefix, 'ep%04d.png' % epoch)
        #     curr_var_preds = preds[self.datagen_outputs_indices[var]]
        #     curr_var_true = batch_y[self.datagen_outputs_indices[var]]
        #     f = plt.figure(figsize=(9, 4), dpi=300)
        #     for i in range(curr_var_true.shape[1]):
        #         p = plt.subplot(1,2,i+1)
        #         plt.scatter(curr_var_true[:,i], curr_var_preds[:,i], s=1)
        #     plt.tight_layout()
        #     plt.savefig(curr_png_fname)
        #     plt.close(f)
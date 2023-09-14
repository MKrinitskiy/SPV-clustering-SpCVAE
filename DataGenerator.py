from tensorflow._api.v1.keras import backend as K
from tensorflow._api.v1.keras.preprocessing.image import ImageDataGenerator, Iterator, array_to_img, random_channel_shift
import numpy as np
from KFoldCycleCallback import *
from support_defs import *
import tensorflow._api.v1.keras as keras
import pickle



class DataManager(ImageDataGenerator):
    def __init__(self,
                 srcdatafiles, # dictionary, key - variable
                 normdata_files, # dictionary, key - variable
                 bottleneck_dims,
                 input_vars                     = ('hgt', 'pv', 'pv_th'),
                 output_vars                    = ('hgt', 'pv', 'pv_th', 'reg_vars'),
                 mask_file                      = './dataset_ge40North/mask_256.npy',
                 apply_mask_to                  = ('hgt', 'pv', 'pv_th'),
                 mmap_vars                      = ('hgt', 'pv', 'pv_th'),
                 expand_dims_for_channel        = ('hgt', 'pv', 'pv_th'),
                 val_folds                      = 5,
                 test_split_ratio               = 0.2,
                 image_size                     = (256,256),
                 shuffle                        = True):

        self.srcdatafiles                           = srcdatafiles
        self.normdata_files                         = normdata_files
        self.source_data                            = {}
        self.norm_data                              = {}
        self.input_vars                             = input_vars
        self.output_vars                            = output_vars
        self.test_split_ratio                       = test_split_ratio
        self.val_folds                              = val_folds
        self.image_size                             = image_size
        self.bottleneck_dims                        = bottleneck_dims
        self.mask                                   = np.load(mask_file)
        self.apply_mask_to                          = apply_mask_to
        self.mmap_vars                              = mmap_vars
        self.shuffle                                = shuffle
        self.expand_dims_for_channel                = expand_dims_for_channel
        self.read_data()


        super(DataManager, self).__init__()




    def flow(self, x, y=None, batch_size=32, shuffle=True, sample_weight=None, seed=None, save_to_dir=None, save_prefix='', save_format='png', category='train'):
        if category == 'train':
            self.train_flowDataGenerator = DataGenerator(self, batch_size, category, shuffle=shuffle)
            return self.train_flowDataGenerator
        elif category == 'val':
            self.val_flowDataGenerator = DataGenerator(self, batch_size, category, shuffle=shuffle)
            return self.val_flowDataGenerator
        elif category == 'test':
            self.test_flowDataGenerator = DataGenerator(self, batch_size, category, shuffle=shuffle)
            return self.test_flowDataGenerator
        elif category == 'all':
            self.all_flowDataGenerator = DataGenerator(self, batch_size, category, shuffle=shuffle)
            return self.all_flowDataGenerator




    def read_data(self):
        for inp_name in self.srcdatafiles.keys():
            if inp_name in self.mmap_vars:
                self.source_data[inp_name] = np.load(self.srcdatafiles[inp_name], mmap_mode='r')
            else:
                self.source_data[inp_name] = np.load(self.srcdatafiles[inp_name])

            if inp_name in self.normdata_files.keys():
                with open(self.normdata_files[inp_name], 'rb') as f:
                    dict1 = pickle.load(f)
                    self.norm_data[inp_name] = {'minval': [dict1[k] for k in dict1.keys() if 'minval' in k][0],
                                                'maxval': [dict1[k] for k in dict1.keys() if 'maxval' in k][0]}

        self.total_objects_count = self.source_data[self.input_vars[0]].shape[0]

        # split train+val vs test
        self.total_indices = np.arange(0, self.total_objects_count, 1)
        if self.shuffle:
            self.total_indices_permuted = np.random.permutation(self.total_objects_count)
        else:
            self.total_indices_permuted = np.arange(self.total_objects_count)

        if self.test_split_ratio is not None:
            test_indices_count = int(self.total_objects_count * self.test_split_ratio)
            self.test_indices = np.asarray(self.total_indices_permuted[:test_indices_count])

            self.trainval_indices = self.total_indices_permuted[test_indices_count:]
            self.KFold_valIndices_iterator = K_folds_iterator(x=self.trainval_indices, batch_size=int( len(self.trainval_indices) / self.val_folds))
            self.val_indices = np.asarray(self.KFold_valIndices_iterator.next())
            self.train_indices = np.asarray([self.trainval_indices[i] for i in range(len(self.trainval_indices)) if self.trainval_indices[i] not in self.val_indices])
        else:
            test_indices_count = 0
            self.test_indices = np.asarray([])

            self.trainval_indices = self.total_indices_permuted[:]
            self.KFold_valIndices_iterator = K_folds_iterator(x=self.trainval_indices, batch_size=int(len(self.trainval_indices) / self.val_folds))
            self.val_indices = np.asarray(self.KFold_valIndices_iterator.next())
            self.train_indices = np.asarray([self.trainval_indices[i] for i in range(len(self.trainval_indices)) if self.trainval_indices[i] not in self.val_indices])



        print('data_train samples: ', len(self.train_indices))
        print('data_val samples: ', len(self.val_indices))
        print('data_test samples: ', len(self.test_indices))




    def dataset_length_batches(self, category='train', batch_size = 32):
        indices_count = self.train_indices.shape[0]
        if category == 'train':
            indices_count = self.train_indices.shape[0]
        elif category == 'val':
            indices_count = self.val_indices.shape[0]
        elif category == 'test':
            indices_count = self.test_indices.shape[0]
        elif category == 'all':
            indices_count = self.total_indices.shape[0]

        if indices_count % batch_size == 0:
            return indices_count / batch_size
        else:
            return int(np.round(indices_count / float(batch_size))+1)




    def cycle_KFold_train_val_sets(self):
        print('cycling val,train sets inside train+val set')
        self.val_indices = np.asarray(self.KFold_valIndices_iterator.next())
        self.train_indices = np.asarray([self.trainval_indices[i] for i in range(len(self.trainval_indices)) if self.trainval_indices[i] not in self.val_indices])





class DataGenerator(Iterator):

    def __init__(self, parent, batch_size=32, category='train', shuffle = True):
        self.shuffle = shuffle
        self.parent = parent
        self.category = category
        if (self.category == 'train'):
            self.data_indices = self.parent.train_indices
        elif (self.category == 'val'):
            self.data_indices = self.parent.val_indices
        elif (self.category == 'test'):
            self.data_indices = self.parent.test_indices
        elif (self.category == 'all'):
            self.data_indices = self.parent.total_indices

        super(DataGenerator, self).__init__(len(self.data_indices), batch_size, seed=None, shuffle=self.shuffle)


    def next(self):
        with self.lock:
            next_indices_indices = next(self.index_generator)

        curr_indices = self.data_indices[next_indices_indices]
        batch_data = {}
        batch_target = {}
        for name in self.parent.input_vars:
            batch_data[name] = self.parent.source_data[name][curr_indices] # batch x W x H   or batch
            if name in self.parent.expand_dims_for_channel:
                batch_data[name] = np.expand_dims(batch_data[name], -1) # batch x W x H x C    or batch x 1
            if name in self.parent.norm_data.keys():
                minval = self.parent.norm_data[name]['minval']
                maxval = self.parent.norm_data[name]['maxval']
                batch_data[name] = ((batch_data[name]-minval)/(maxval-minval)).astype(np.float32)

        maskbatch = np.tile(self.parent.mask[np.newaxis, :, :, np.newaxis],
                            (batch_data[self.parent.input_vars[0]].shape[0], 1, 1, 1)).astype(np.float32)
        for k in self.parent.apply_mask_to:
            if k in batch_data.keys():
                batch_data[k] = (batch_data[k] * maskbatch).astype(np.float32)

        fake_z_target = np.zeros((len(curr_indices), self.parent.bottleneck_dims), dtype=np.float32)

        for name in self.parent.output_vars:
            batch_target[name] = self.parent.source_data[name][curr_indices]
            if name in self.parent.expand_dims_for_channel:
                batch_target[name] = np.expand_dims(batch_target[name], -1)  # batch x W x H x C
            if name in self.parent.norm_data.keys():
                minval = self.parent.norm_data[name]['minval']
                maxval = self.parent.norm_data[name]['maxval']
                batch_target[name] = ((batch_target[name] - minval) / (maxval - minval)).astype(np.float32)

        for k in self.parent.apply_mask_to:
            if k in batch_target.keys():
                batch_target[k] = (batch_target[k] * maskbatch).astype(np.float32)

        return [batch_data[n] for n in self.parent.input_vars] + [maskbatch], [batch_target[n] for n in self.parent.output_vars] + [fake_z_target]



    def _get_batches_of_transformed_samples(self, index_array):
        curr_indices = self.data_indices[index_array]
        batch_data = {}
        batch_target = {}
        for name in self.parent.input_vars:
            batch_data[name] = self.parent.source_data[name][curr_indices]  # batch x W x H
            if name in self.parent.expand_dims_for_channel:
                batch_data[name] = np.expand_dims(batch_data[name], -1)  # batch x W x H x C

            if name in self.parent.norm_data.keys():
                minval = self.parent.norm_data[name]['minval']
                maxval = self.parent.norm_data[name]['maxval']
                batch_data[name] = ((batch_data[name] - minval) / (maxval - minval)).astype(np.float32)

        maskbatch = np.tile(self.parent.mask[np.newaxis, :, :, np.newaxis], (batch_data[self.parent.input_vars[0]].shape[0], 1, 1, 1)).astype(np.float32)
        for k in self.parent.apply_mask_to:
            if k in batch_data.keys():
                batch_data[k] = (batch_data[k] * maskbatch).astype(np.float32)

        fake_z_target = np.zeros((len(curr_indices), self.parent.bottleneck_dims), dtype=np.float32)

        for name in self.parent.output_vars:
            batch_target[name] = self.parent.source_data[name][curr_indices]
            if name in self.parent.expand_dims_for_channel:
                batch_target[name] = np.expand_dims(batch_target[name], -1)  # batch x W x H x C
            if name in self.parent.norm_data.keys():
                minval = self.parent.norm_data[name]['minval']
                maxval = self.parent.norm_data[name]['maxval']
                batch_target[name] = ((batch_target[name] - minval) / (maxval - minval)).astype(np.float32)


        for k in self.parent.apply_mask_to:
            if k in batch_target.keys():
                batch_target[k] = (batch_target[k] * maskbatch).astype(np.float32)

        return [batch_data[n] for n in self.parent.input_vars] + [maskbatch], [batch_target[n] for n in self.parent.output_vars] + [fake_z_target]
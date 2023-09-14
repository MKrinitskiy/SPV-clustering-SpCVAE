from DataGeneratorImgaug import *
import numpy as np
from matplotlib import pyplot as plt





datagen = DataManager(srcdatafiles={'hgt': '../dataset_ge40North/hgt/hgt_data_projected_all.normed01.npy',
                                    'pv': '../dataset_ge40North/pv/pv_data_projected_all.normed01.npy',
                                    'reg_output': '../dataset_ge40North/regression_MeanZonalWindSpeed60N_TPolarCape.npy'},
                      normdata_files = {},
                      bottleneck_dims = 32,
                      input_vars=['hgt', 'pv'],
                      imgaugment_vars = ['hgt', 'pv'],
                      expand_dims_for_channel = ['hgt', 'pv'],
                      output_vars=['hgt', 'pv', 'reg_output'],
                      mask_file='../dataset_ge40North/mask_256.npy',
                      mmap_vars=['hgt', 'pv'],
                      val_folds=5,
                      test_split_ratio=None,
                      image_size = (256,256))



train_generator = datagen.flow(None, batch_size=16, category='train')


batch_x, batch_y = train_generator._get_batches_of_transformed_samples(np.random.permutation(train_generator.data_indices.shape[0])[:16])


idx = np.random.randint(0,16,1)[0]
f = plt.figure(figsize=(4,3), dpi=300)

plt.subplot(2,3,1)
plt.imshow(np.squeeze(batch_x[-1][idx]), cmap='gray')
plt.axis('off')

plt.subplot(2,3,2)
plt.imshow(np.squeeze(batch_x[0][idx]), cmap='gray')
plt.title('HGT')
plt.axis('off')

plt.subplot(2,3,3)
plt.imshow(np.squeeze(batch_y[0][idx]), cmap='gray')
plt.title('PV')
plt.axis('off')


plt.subplot(2,3,4)
plt.imshow(np.squeeze(batch_x[-1][idx]), cmap='gray')
plt.axis('off')

plt.subplot(2,3,5)
plt.imshow(np.squeeze(batch_x[1][idx]), cmap='gray')
plt.title('HGT')
plt.axis('off')

plt.subplot(2,3,6)
plt.imshow(np.squeeze(batch_y[1][idx]), cmap='gray')
plt.title('PV')
plt.axis('off')

plt.show()

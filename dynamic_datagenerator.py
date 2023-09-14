import numpy as np
import cv2
from numpy.random import randint
from sklearn.datasets import fetch_mldata



class dynamic_datagenerator():
    def __init__(self, image_size, target_label=8, toRGB = False, useOpenCV = True):
        mnist = fetch_mldata('MNIST original', data_home='./MNISTdata/')
        self.mnist_data = mnist.data
        self.mnist_targets = mnist.target
        self.image_size = image_size
        self.target_label = target_label
        self.toRGB = toRGB
        self.useOpenCV = useOpenCV


    def augment_brightness_camera_images(self, image):
        image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        random_bright = .25+np.random.uniform()
        #print(random_bright)
        image1[:,:,2] = image1[:,:,2]*random_bright
        image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
        return image1





    def transform_image(self, img,ang_range,shear_range,trans_range, zoom_range = 0.0, brightness=0):
        '''
        This function transforms images to generate new images.
        The function takes in following arguments,
        1- Image
        2- ang_range: Range of angles for rotation
        3- shear_range: Range of values to apply affine transform to
        4- trans_range: Range of values to apply translations over.

        A Random uniform distribution is used to generate different parameters for transformation

        '''
        # Rotation

        # zoom
        zoom_factor = 1.0 + np.random.uniform(-zoom_range, zoom_range, 1)
        # print(zoom_factor)
        img = cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor, interpolation = cv2.INTER_CUBIC)

        ang_rot = np.random.uniform(ang_range)-ang_range/2
    #     rows,cols,ch = img.shape
        rows,cols = img.shape
        Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

        # Translation
        tr_x = trans_range*np.random.uniform()-trans_range/2
        tr_y = trans_range*np.random.uniform()-trans_range/2
        Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

        # Shear
        pts1 = np.float32([[5,5],[20,5],[5,20]])

        pt1 = 5+shear_range*np.random.uniform()-shear_range/2
        pt2 = 20+shear_range*np.random.uniform()-shear_range/2

        # Brightness


        pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

        shear_M = cv2.getAffineTransform(pts1,pts2)

        img = cv2.warpAffine(img,Rot_M,(cols,rows))
        img = cv2.warpAffine(img,Trans_M,(cols,rows))
        img = cv2.warpAffine(img,shear_M,(cols,rows))




        # if brightness == 1:
        #     img = augment_brightness_camera_images(img)

        return img


    def rect_union(self, a,b):
        x = min(a[0], b[0])
        y = min(a[1], b[1])
        w = max(a[0]+a[2], b[0]+b[2]) - x
        h = max(a[1]+a[3], b[1]+b[3]) - y
        return (x, y, w, h)

    def rect_intersection(self, a,b):
        x = max(a[0], b[0])
        y = max(a[1], b[1])
        w = min(a[0]+a[2], b[0]+b[2]) - x
        h = min(a[1]+a[3], b[1]+b[3]) - y
        if w<0 or h<0: return None # or (0,0,0,0) ?
        return (x, y, w, h)



    def check_if_intersects(self, rect1, rect2):
        return (self.rect_intersection(rect1, rect2) is not None)

    def check_if_none_intersects(self, test_rect, other_rects):
        for rect in other_rects:
            if self.check_if_intersects(test_rect, rect):
                return False
        return True




    def new_sample(self, objects_to_place = 4): # digits_set, targets_set, target_label=8, image_size=(256, 256), target_thresholding = 0.1):

        selected_indices = randint(0, self.mnist_data.shape[0], objects_to_place)
        digits_set = self.mnist_data[selected_indices, :].reshape((len(selected_indices), 28, 28))
        targets_set = self.mnist_targets[selected_indices]
        while (self.target_label not in targets_set):
            selected_indices = randint(0, self.mnist_data.shape[0], objects_to_place)
            digits_set = self.mnist_data[selected_indices, :].reshape((len(selected_indices), 28, 28))
            targets_set = self.mnist_targets[selected_indices]

        rects = []

        if self.toRGB:
            dst_img = np.zeros(list(self.image_size) + list([3]), np.uint8)
            target_image = np.zeros(list(self.image_size) + list([3]), np.uint8)
        else:
            dst_img = np.zeros(self.image_size, np.uint8)
            target_image = np.zeros(self.image_size, np.uint8)

        for i in range(digits_set.shape[0]):
            curr_target_label = targets_set[i]
            digit_sample = np.zeros((40, 40), dtype=np.uint8)
            digit_sample[6:34, 6:34] = digits_set[i, :].reshape((28, 28))
            digit_sample = self.transform_image(digit_sample, 180, 10, 0, zoom_range=0.3)
            if self.toRGB:
                if self.useOpenCV:
                    digit_sample_bgr = cv2.cvtColor(digit_sample, cv2.COLOR_GRAY2BGR)
                else:
                    digit_sample_bgr = self.replicate_array_to_depth3(digit_sample_bgr)


            x_offset = randint(0, self.image_size[0] - digit_sample.shape[0])
            y_offset = randint(0, self.image_size[1] - digit_sample.shape[1])
            rect_current = (x_offset, y_offset, digit_sample.shape[0], digit_sample.shape[1])
            while (not self.check_if_none_intersects(rect_current, rects)):
                x_offset = randint(0, self.image_size[0] - digit_sample.shape[0])
                y_offset = randint(0, self.image_size[1] - digit_sample.shape[1])
                rect_current = (x_offset, y_offset, digit_sample.shape[0], digit_sample.shape[1])

            rects.append(rect_current)

            i_indices, j_indices = np.where(digit_sample > 0)

            for x, y in zip(i_indices, j_indices):
                if self.toRGB:
                    dst_img[x_offset + x, y_offset + y, :] = digit_sample_bgr[x, y, :]
                else:
                    dst_img[x_offset + x, y_offset + y] = digit_sample[x, y]



            if (curr_target_label == self.target_label):
                for x, y in zip(i_indices, j_indices):
                    if self.toRGB:
                        target_image[x_offset + x, y_offset + y, :] = digit_sample_bgr[x, y, :]
                    else:
                        target_image[x_offset + x, y_offset + y] = digit_sample[x, y]


        dst_img = np.divide(np.float32(dst_img), 255.0)
        target_image = np.divide(np.float32(target_image), 255.0)
        # target_image[np.where(target_image>target_thresholding)] = 1.0
        # target_image[np.where(target_image <= target_thresholding)] = 0.0
        # target_image[np.where(target_image > 0.0)] = 1.0

        return (dst_img, target_image)


    def replicate_array_to_depth3(self, arr):
        x = np.expand_dims(arr, -1)
        x = np.repeat(x, 3, -1)
        return x
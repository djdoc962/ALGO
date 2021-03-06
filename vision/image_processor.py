import cv2
import numpy as np
from typing import (List, Tuple)
from module.type import Polygon
import copy


class Config:
    # IMAGE_DIR = '/Users/jojolin/Documents/proj/ALGO/vision/image'
    # IMAGE_TYPE = 'bgr'
    # IMAGE_SHAPE = (512, 512, 3)

    # BRIGHTNESS_RATIO = (.5, 2.)
    # FLIP_ORIENTATION = 'horizontal_random'
    # SCALE_RATIO = (.8, 1.2)
    # DEGREE = (-5., 5.)
    # H_SHIFT = (-10, 10)
    # V_SHIFT = (-10, 10)

    IMAGE_DIR = '/Users/jojolin/Documents/proj/ALGO/vision/image'
    IMAGE_TYPE = 'bgr'
    IMAGE_SHAPE = None
    # IMAGE_SHAPE = (3496,2469,3)

    BRIGHTNESS_RATIO = None
    FLIP_ORIENTATION = None
    SCALE_RATIO = 0.8
    DEGREE = (-5., 5.)
    H_SHIFT = None
    V_SHIFT = None


class ImageProcessor:
    image_array: np.array
    image_shape: Tuple
    scale: Tuple

    def affine(self, origin_pts=None, horizontal=None, vertical=None):

        if len(self.image_shape) == 3:
            height, width, depth = self.image_shape
        else:
            raise ValueError(f'The length of {self.image_shape} is not equal to 3...')

        if origin_pts is None:
            origin_pts = tuple([np.random.randint(0, width), np.random.randint(0, height)] for _ in range(3))

        if isinstance(horizontal, tuple) or isinstance(horizontal, list):
            dw = tuple(np.random.randint(horizontal[0], horizontal[1]+1) for _ in range(3))
        elif horizontal is None:
            dw = (0, 0, 0)
        else:
            dw = horizontal

        if isinstance(vertical, tuple) or isinstance(vertical, list):
            dh = tuple(np.random.randint(vertical[0], vertical[1]+1) for _ in range(3))
        elif horizontal is None:
            dh = (0, 0, 0)
        else:
            dh = vertical

        if (not len(origin_pts) == 3) or (not len(horizontal) == 3) or (not len(vertical) == 3):
            raise ValueError('There must be 3 points provided...')

        target_pts = tuple()
        for origin_pt, dx, dy in zip(origin_pts, dw, dh):
            _x = origin_pts[0] + dx
            _x = width - 1 if _x >= width else _x
            _x = 0 if _x < 0 else _x

            _y = origin_pts[1] + dy
            _y = height - 1 if _y >= height else _y
            _y = 0 if _y < 0 else _y

            target_pts += ([_x, _y],)

        M = cv2.getAffineTransform(np.float32(origin_pts), np.float32(target_pts))
        self.image_array = cv2.warpAffine(self.image_array.copy(), M, (width, height))
        return self

    def brightness(self, ratio=None, image_type='rgb'):
        if ratio is None:
            return self

        image_array = self.image_array.copy()
        if image_type == 'rgb':
            hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        elif image_type == 'bgr':
            hsv = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
        elif image_type == 'gray':
            gray = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
            hsv = cv2.cvtColor(gray, cv2.COLOR_BGR2HSV)
        else:
            raise ValueError(f'{image_type} is not implemented...')

        if isinstance(ratio, tuple) or isinstance(ratio, list):
            factor = np.random.uniform(ratio[0], ratio[1])
        else:
            factor = ratio

        mask = hsv[..., 2] * factor > 255
        v_channel = np.where(mask, 255, hsv[:, :, 2] * factor)
        hsv[..., 2] = v_channel

        if image_type == 'rgb':
            self.image_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        if image_type == 'bgr':
            self.image_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        if image_type == 'gray':
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            self.image_array = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        return self

    def crop(self, xmin: int, ymin: int, xmax: int, ymax: int):
        image_array = self.image_array.copy()
        self.image_array = image_array[ymin:ymax, xmin:xmax]
        self.image_shape = self.image_array.shape
        return self

    def equalizer(self,
                  method='hist',
                  image_type='rgb',
                  clip_limit=None,
                  tile_grid_size_width=None,
                  tile_grid_size_height=None):
        if method not in ['hist', 'clahe']:
            raise ValueError(f'{method} is not implemented...')

        image_copied = np.copy(self.image_array)
        if method == 'hist':
            self.image_array = self._hist_equalizer(image_copied, image_type=image_type)
            return self
        if method == 'clahe':
            if clip_limit is None:
                clip_limit = (1., 2.)

            if tile_grid_size_width is None:
                tile_grid_size_width = (6, 10)

            if tile_grid_size_height is None:
                tile_grid_size_height = (6, 10)

            if isinstance(clip_limit, tuple) or isinstance(clip_limit, list):
                clip_factor = np.random.uniform(clip_limit[0], clip_limit[1])
            else:
                clip_factor = clip_limit

            if isinstance(tile_grid_size_width, tuple) or isinstance(tile_grid_size_width, list):
                width_size = np.random.randint(tile_grid_size_width[0], tile_grid_size_width[1])
            else:
                width_size = tile_grid_size_width

            if isinstance(tile_grid_size_height, tuple) or isinstance(tile_grid_size_height, list):
                height_size = np.random.randint(tile_grid_size_height[0], tile_grid_size_height[1])
            else:
                height_size = tile_grid_size_height

            self.image_array = self._clahe(image_copied,
                                           clip_limit=clip_factor,
                                           tile_grid_size=(height_size, width_size))
            return self

    def flip(self, orientation=None):
        """
        :param orientation: horizontal, vertical, random, horizontal_random, vertical_random
        """
        image_array = self.image_array.copy()

        if orientation == 'vertical':
            self.image_array = cv2.flip(image_array, 0)

        if orientation == 'horizontal':
            self.image_array = cv2.flip(image_array, 1)

        if orientation == 'horizontal_random':
            if np.random.randint(0, 2):
                self.image_array = cv2.flip(image_array, 1)
                orientation = 'horizontal'

        if orientation == 'vertical_random':
            if np.random.randint(0, 2):
                self.image_array = cv2.flip(image_array, 0)
                orientation = 'vertical'

        return orientation

    def perspective(self, angle_scale=0., irregularity=0., spikeyness=0.,):
        angle_scale = angle_scale if angle_scale is not None else 0.
        irregularity = irregularity if irregularity is not None else 0.
        spikeyness = spikeyness if spikeyness is not None else 0.

        avg_r = min(self.image_shape[:2]) / 6
        polygon1 = Polygon(x=self.image_shape[1],
                           y=self.image_shape[0], avg_r=avg_r,
                           angle_scale=angle_scale,
                           irregularity=irregularity,
                           spikeyness=spikeyness,
                           num_verts=4)
        polygon2 = Polygon(x=self.image_shape[1],
                           y=self.image_shape[0], avg_r=avg_r,
                           angle_scale=angle_scale,
                           irregularity=irregularity,
                           spikeyness=spikeyness,
                           num_verts=4)

        pts1 = polygon1.points
        pts2 = polygon2.points
        M = cv2.getPerspectiveTransform(pts1.astype(np.float32), pts2.astype(np.float32))
        self.image_array = cv2.warpPerspective(self.image_array, M, (self.image_shape[1], self.image_shape[0]))
        return M

    def resize(self, size, keep_aspect_ratio=False):
        image_array = self.image_array.copy()
        h, w = image_array.shape[:2]

        if not keep_aspect_ratio:
            self.image_array = cv2.resize(image_array, size)
            self.scale = (size[0] * self.scale[0] / w, size[1] * self.scale[1] / h)
        else:
            scale = min(size[1] / h, size[0] / w)
            self.image_array = cv2.resize(image_array, None, fx=scale, fy=scale)
            self.scale = (scale * self.scale[0], scale * self.scale[1])

        self.image_shape = self.image_array.shape
        return self

    def rotate(self, degree=None):
        if len(self.image_shape) == 3:
            height, width, depth = self.image_shape
        else:
            raise ValueError(f'The length of {self.image_shape} is not equal to 3...')

        if isinstance(degree, tuple) or isinstance(degree, list):
            factor = np.random.uniform(degree[0], degree[1])
        else:
            factor = degree

        M = cv2.getRotationMatrix2D((width / 2, height / 2), factor, 1)
        self.image_array = cv2.warpAffine(self.image_array.copy(), M, (width, height))
        print('rotate.factor => '+str(factor))
        return M

    def rescale(self, ratio=None):
        if len(self.image_shape) == 3:
            height, width, depth = self.image_shape
            print('height {}, width {}, depth {} '.format(str(height),str(width),str(depth)))
        else:
            raise ValueError(f'The length of {self.image_shape} is not equal to 3...')

        if isinstance(ratio, tuple) or isinstance(ratio, list):
            factor = np.random.uniform(ratio[0], ratio[1])
        else:
            factor = ratio

        self.scale = tuple([self.scale[idx] * factor for idx in range(2)])
        M = cv2.getRotationMatrix2D((width / 2, height / 2), 0, factor)
        self.image_array = cv2.warpAffine(self.image_array.copy(), M, (width, height))
        return M

    def translate(self, horizontal=None, vertical=None):
        if len(self.image_shape) == 3:
            height, width, depth = self.image_shape
        else:
            raise ValueError(f'The length of {self.image_shape} is not equal to 3...')

        if isinstance(horizontal, tuple) or isinstance(horizontal, list):
            dw = np.random.randint(horizontal[0], horizontal[1]+1)
        elif horizontal is None:
            dw = 0
        else:
            dw = horizontal

        if isinstance(vertical, tuple) or isinstance(vertical, list):
            dh = np.random.randint(vertical[0], vertical[1]+1)
        elif vertical is None:
            dh = 0
        else:
            dh = vertical

        # create the transformation matrix
        M = np.float32([[1, 0, dw], [0, 1, dh]])

        self.image_array = cv2.warpAffine(self.image_array.copy(), M, (width, height))
        print('translation ( H: {}, V: {})'.format(dw,dh))
        return M

    @staticmethod
    def _hist_equalizer(image: np.array, image_type='rgb'):
        if image_type == 'rgb':
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif image_type == 'bgr':
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            raise ValueError(f'{image_type} is not implemented...')

        hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])

        if image_type == 'rgb':
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        if image_type == 'bgr':
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def _clahe(image: np.array, clip_limit=2.0, tile_grid_size=(8, 8)):
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        if len(image.shape) == 2:
            return clahe.apply(image)
        elif len(image.shape) == 3:
            for channel in range(3):
                image[..., channel] = clahe.apply(image[..., channel])

            return image
        else:
            return image


class ImageInput(ImageProcessor):
    def __init__(self,
                 image_path,
                 image_type='rgb',
                 image_shape=None,
                 mode=None):
        self.image_path = image_path
        self.image_type = image_type.lower()
        self.image_shape = image_shape
        self.scale = (1, 1)

        # The shape of image_array returns (height, width, depth)
        if mode is not None:
            self.image_array = cv2.imread(self.image_path, mode)
        else:
            self.image_array = cv2.imread(self.image_path)

        if self.image_type == 'rgb':
            self.image_array = cv2.cvtColor(self.image_array, cv2.COLOR_BGR2RGB)

        if self.image_type == 'gray':
            self.image_array = cv2.cvtColor(self.image_array, cv2.COLOR_BGR2GRAY)

        if self.image_type == 'hsv':
            self.image_array = cv2.cvtColor(self.image_array, cv2.COLOR_BGR2HSV)

        if self.image_shape is not None:
            self.scale = tuple(self.image_shape[index] / self.image_array.shape[index] for index in range(2))
            self.image_array = cv2.resize(self.image_array, self.image_shape[:2])
        else:
            self.image_shape=self.image_array.shape

        if len(self.image_array.shape) < 3:
            self.image_array = self.image_array[..., np.newaxis]

    def convert(self, image_type):
        if self.image_type == 'bgr':
            if image_type == 'bgr':
                return self.image_array
            elif image_type == 'rgb':
                return cv2.cvtColor(self.image_array, cv2.COLOR_BGR2RGB)
            elif image_type == 'gray':
                return cv2.cvtColor(self.image_array, cv2.COLOR_BGR2GRAY)
            elif image_type == 'hsv':
                return cv2.cvtColor(self.image_array, cv2.COLOR_BGR2HSV)
        elif self.image_type == 'rgb':
            if image_type == 'bgr':
                return cv2.cvtColor(self.image_array, cv2.COLOR_RGB2BGR)
            elif image_type == 'rgb':
                return self.image_array
            elif image_type == 'gray':
                return cv2.cvtColor(self.image_array, cv2.COLOR_RGB2GRAY)
            elif image_type == 'hsv':
                return cv2.cvtColor(self.image_array, cv2.COLOR_RGB2HSV)
        elif self.image_type == 'hsv':
            if image_type == 'bgr':
                return cv2.cvtColor(self.image_array, cv2.COLOR_HSV2BGR)
            elif image_type == 'rgb':
                return cv2.cvtColor(self.image_array, cv2.COLOR_HSV2RGB)
            elif image_type == 'gray':
                image_array = cv2.cvtColor(self.image_array, cv2.COLOR_HSV2BGR)
                return cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            elif image_type == 'hsv':
                return self.image_array
        else:
            raise ValueError(f'The type {image_type} can be converted from {self.image_type}')


class ImageAugmentation:
    def __init__(self,
                 brightness_ratio=None,
                 flip_orientation=None,
                 scale_ratio=None,
                 degree=None,
                 h_shift=None,
                 v_shift=None,
                 angle_scale=None,
                 irregularity=None,
                 spikeyness=None):
        self.brightness_ratio = brightness_ratio
        self.flip_orientation = flip_orientation
        self.scale_ratio = scale_ratio
        self.degree = degree
        self.h_shift = h_shift
        self.v_shift = v_shift
        self.angle_scale = angle_scale
        self.irregularity = irregularity
        self.spikeyness = spikeyness

    def execute(self, image_input_list: List[ImageInput]):
        for image_input in image_input_list:
            print('do execute =>')
            image_type = image_input.image_type

            if self.brightness_ratio is not None:
                print('brightness =>'+str(self.brightness_ratio))
                image_input.brightness(ratio=self.brightness_ratio,
                                       image_type=image_type)

            if self.flip_orientation is not None:
                print('flip =>'+ str(self.flip_orientation))
                image_input.flip(orientation=self.flip_orientation)

            if self.scale_ratio is not None:
                print('scale =>'+str(self.scale_ratio))
                image_input.rescale(ratio=self.scale_ratio)

            if self.degree is not None:
                print('roate => {} degrees '.format(str(self.degree)))
                image_input.rotate(degree=self.degree)

            if self.h_shift is not None or self.v_shift is not None:
                print('horizon/vertical shift => ')
                image_input.translate(horizontal=self.h_shift, vertical=self.v_shift)

            if self.angle_scale is not None or self.irregularity is not None or self.spikeyness is not None:
                print('angle scale =>')
                image_input.perspective(angle_scale=self.angle_scale,
                                        irregularity=self.irregularity,
                                        spikeyness=self.spikeyness)

            if len(image_input.image_array.shape) < 3:
                image_input.image_array = image_input.image_array[..., np.newaxis]




if __name__ == '__main__':
    source_ref = './image/insurance2_template.png'
    config = Config()
    augment = ImageAugmentation(brightness_ratio=config.BRIGHTNESS_RATIO,
                                flip_orientation=config.FLIP_ORIENTATION,
                                scale_ratio=config.SCALE_RATIO,
                                degree=config.DEGREE,
                                h_shift=config.H_SHIFT,
                                v_shift=config.V_SHIFT)
    
    image_input = ImageInput(source_ref,
                            image_type=config.IMAGE_TYPE,
                            image_shape=config.IMAGE_SHAPE)
    print('get image => ')
    image_input_list = []
    image_input_list.append(copy.deepcopy(image_input))
    
    # ???init ImageAugmentation???????????????????????????????????????????????????call execute()??????image_input_list??????????????????????????????ImageInput?????????
    print('image_input_list len => '+str(len(image_input_list)))
    i = 0
    for image_input in image_input_list:
        # cv2.imshow('Augment', image_input.image_array)
        augment.execute(image_input_list=image_input_list)
        cv2.imwrite('./outcome/affine/final{}.png'.format(str(i)),image_input.image_array)
        i+=1
        # key = cv2.waitKey(5000)
        # if key == ord('q') or key == 27:
        #     break
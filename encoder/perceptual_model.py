import os, sys
import bz2
import PIL.Image

import keras

import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.utils import get_file
from keras.applications.vgg16 import VGG16, preprocess_input
import keras.backend as K
import traceback

import time

FACE_MODEL_ROOT='/home/anpei/Desktop/Dense_FA_3d'
sys.path.append(FACE_MODEL_ROOT)
from face_worker_util import load_default_models, gen_texture_mapping


def load_images(images_list, image_size=256):
    loaded_images = list()
    for img_path in images_list:
      img = PIL.Image.open(img_path).convert('RGB').resize((image_size,image_size),PIL.Image.LANCZOS)
      img = np.array(img)
      img = np.expand_dims(img, 0)
      loaded_images.append(img)
    loaded_images = np.vstack(loaded_images)
    # print('> loaded images: ', loaded_images.shape)
    return loaded_images

def load_images_with_weight(images_list, image_size=256):
    # load image weights
    loaded_images = list()
    for img_path in images_list:
      img = PIL.Image.open(img_path).convert('RGB').resize(
          (image_size, image_size), PIL.Image.LANCZOS)
      weight_img_path = os.path.basename(img_path).split('_')
      img = np.array(img)
      img = np.expand_dims(img, 0)
      loaded_images.append(img)
    loaded_images = np.vstack(loaded_images)
    return loaded_images

def gen_textures(imgs, face_models, texture_size=1024, output_dir=None, imgs_id=None):
    """Generate textures and match placeholder shapes."""
    # tex_coord originally [B, H, W, 4], weight [B, H, W, 4]
    # transfer to [4, B, H, W, 3]
    try:
        tex_coord, weight, conf_map = gen_texture_mapping(
            imgs, texture_size=texture_size, 
            output_dir=output_dir,
            img_id=imgs_id,
            **face_models)
    except ValueError as e: 
        print('Failed to generated textures: ', e)
        return None, None, None

    B, H, W, _ = tex_coord.shape
    tex_coords = np.zeros((4, B, H, W, 3))
    weights = np.zeros((4, B, H, W, 3))

    indexing_dim = np.tile(
        np.reshape(
            np.array(range(B)), 
            (-1, 1, 1, 1)), 
        (1, H, W, 1))

    tex_coords[0] = np.concatenate((indexing_dim, tex_coord[:, :, :, [1, 0]]), 3)
    tex_coords[1] = np.concatenate((indexing_dim, tex_coord[:, :, :, [1, 2]]), 3)
    tex_coords[2] = np.concatenate((indexing_dim, tex_coord[:, :, :, [3, 0]]), 3)
    tex_coords[3] = np.concatenate((indexing_dim, tex_coord[:, :, :, [3, 2]]), 3)

    weights[0] = np.tile(np.expand_dims(weight[:, :, :, 0], 3), (1, 1, 1, 3))
    weights[1] = np.tile(np.expand_dims(weight[:, :, :, 1], 3), (1, 1, 1, 3))
    weights[2] = np.tile(np.expand_dims(weight[:, :, :, 2], 3), (1, 1, 1, 3))
    weights[3] = np.tile(np.expand_dims(weight[:, :, :, 3], 3), (1, 1, 1, 3))

    return tex_coords, weights, conf_map

def tf_custom_l1_loss(img1,img2):
  return tf.math.reduce_mean(tf.math.abs(img2-img1), axis=None)

def tf_weighted_l1_loss(ref_img, generated_img, weights):
  return tf.math.reduce_mean(tf.math.abs(generated_img - ref_img) * weights)

def tf_custom_logcosh_loss(img1,img2):
  return tf.math.reduce_mean(tf.keras.losses.logcosh(img1,img2))

def tf_weighted_logcosh_loss(ref_img, generated_img, weights):
  return tf.math.reduce_mean(
    tf.keras.losses.logcosh(ref_img, generated_img) * weights)

def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

class PerceptualModel:
    def __init__(self, args, batch_size=1, perc_model=None, sess=None):
        self.sess = tf.get_default_session() if sess is None else sess
        K.set_session(self.sess)
        self.epsilon = 0.00000001
        self.lr = args.lr
        self.decay_rate = args.decay_rate
        self.decay_steps = args.decay_steps
        self.img_size = args.image_size
        self.layer = args.use_vgg_layer
        self.vgg_loss = args.use_vgg_loss
        self.face_mask = args.face_mask
        self.use_grabcut = args.use_grabcut
        self.scale_mask = args.scale_mask
        self.mask_dir = args.mask_dir
        self.texture_size = 1024

        self.output_dir = args.generated_images_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.face_models = None

        self.losses = {}

        if (self.layer <= 0 or self.vgg_loss <= self.epsilon):
            self.vgg_loss = None
        self.pixel_loss = args.use_pixel_loss
        if (self.pixel_loss <= self.epsilon):
            self.pixel_loss = None
        self.mssim_loss = args.use_mssim_loss
        if (self.mssim_loss <= self.epsilon):
            self.mssim_loss = None
        self.lpips_loss = args.use_lpips_loss
        if (self.lpips_loss <= self.epsilon):
            self.lpips_loss = None
        self.l1_penalty = args.use_l1_penalty
        if (self.l1_penalty <= self.epsilon):
            self.l1_penalty = None
        self.tex_loss = args.use_tex_loss
        if (self.tex_loss <= self.epsilon):
            self.tex_loss = None

        self.batch_size = batch_size
        if perc_model is not None and self.lpips_loss is not None:
            self.perc_model = perc_model
        else:
            self.perc_model = None

        self.ref_img = None
        self.ref_weight = None
        self.perceptual_model = None
        self.ref_img_features = None
        self.features_weight = None
        self.loss = None

        if self.face_mask:
            import dlib
            self.detector = dlib.get_frontal_face_detector()
            LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
            landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                                    LANDMARKS_MODEL_URL, cache_subdir='temp'))
            self.predictor = dlib.shape_predictor(landmarks_model_path)

    def compare_images(self,img1,img2):
        if self.perc_model is not None:
            return self.perc_model.get_output_for(tf.transpose(img1, perm=[0,3,2,1]), tf.transpose(img2, perm=[0,3,2,1]))
        return 0

    def add_placeholder(self, var_name):
        var_val = getattr(self, var_name)
        setattr(self, var_name + "_placeholder", tf.placeholder(var_val.dtype, shape=var_val.get_shape()))
        setattr(self, var_name + "_op", var_val.assign(getattr(self, var_name + "_placeholder")))

    def assign_placeholder(self, var_name, var_val):
        self.sess.run(getattr(self, var_name + "_op"), {getattr(self, var_name + "_placeholder"): var_val})

    def build_perceptual_model(self, generator, style_generator=None):

        # Learning rate
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        incremented_global_step = tf.assign_add(global_step, 1)
        self._reset_global_step = tf.assign(global_step, 0)
        self.learning_rate = tf.train.exponential_decay(self.lr, incremented_global_step,
                self.decay_steps, self.decay_rate, staircase=True)
        self.sess.run([self._reset_global_step])

        generated_image_tensor = generator.generated_image
        generated_image = tf.image.resize_nearest_neighbor(generated_image_tensor,
                                                                (self.img_size, self.img_size), align_corners=True)

        self.ref_img = tf.get_variable('ref_img', shape=generated_image.shape,
                                                dtype='float32', initializer=tf.initializers.zeros())
        self.ref_weight = tf.get_variable('ref_weight', shape=generated_image.shape,
                                            dtype='float32', initializer=tf.initializers.zeros())
        self.add_placeholder("ref_img")
        self.add_placeholder("ref_weight")            

        if (self.vgg_loss is not None):
            vgg16 = VGG16(include_top=False, input_shape=(self.img_size, self.img_size, 3))
            self.perceptual_model = Model(vgg16.input, vgg16.layers[self.layer].output)
            generated_img_features = self.perceptual_model(preprocess_input(self.ref_weight * generated_image))
            self.ref_img_features = tf.get_variable('ref_img_features', shape=generated_img_features.shape,
                                                dtype='float32', initializer=tf.initializers.zeros())
            self.features_weight = tf.get_variable('features_weight', shape=generated_img_features.shape,
                                            dtype='float32', initializer=tf.initializers.zeros())
            self.sess.run([self.features_weight.initializer, self.features_weight.initializer])
            self.add_placeholder("ref_img_features")
            self.add_placeholder("features_weight")

        if self.face_models is None:
            self.face_models = load_default_models()[0]

        self.loss = 0
        # L1 loss on VGG16 features
        if (self.vgg_loss is not None):
            self.losses["vgg"] = self.vgg_loss * tf_custom_l1_loss(self.features_weight * self.ref_img_features, self.features_weight * generated_img_features)
            self.loss += self.losses["vgg"]
        # + logcosh loss on image pixels
        if (self.pixel_loss is not None):
            self.losses["pixel"] = self.pixel_loss * tf_custom_logcosh_loss(self.ref_weight * self.ref_img, self.ref_weight * generated_image)
            self.loss += self.losses["pixel"]
        # + MS-SIM loss on image pixels
        if (self.mssim_loss is not None):
            self.losses["mssim"] = self.mssim_loss * tf.math.reduce_mean(1-tf.image.ssim_multiscale(self.ref_weight * self.ref_img, self.ref_weight * generated_image, 1))
            self.loss += self.losses["mssim"]
        # + extra perceptual loss on image pixels
        if self.perc_model is not None and self.lpips_loss is not None:
            self.losses["lpips"] = self.lpips_loss * tf.math.reduce_mean(self.compare_images(self.ref_weight * self.ref_img, self.ref_weight * generated_image))
            self.loss += self.losses["lpips"]
        # + L1 penalty on dlatent weights
        if self.l1_penalty is not None:
            self.losses["l1_penalty"] = self.l1_penalty * 512 * tf.math.reduce_mean(tf.math.abs(generator.dlatent_variable-generator.get_dlatent_avg()))
            self.loss += self.losses["l1_penalty"]
        # pixel wise loss in texture space
        if self.tex_loss is not None:
            self.tex_output_dir = os.path.join(self.output_dir, 'textures')
            os.makedirs(self.tex_output_dir, exist_ok=True)
            self.generate_textures(generator.generated_image_uint8, step=global_step)
            self.losses["tex"] = self.tex_loss * tf_custom_logcosh_loss(self.ref_tex * self.ref_tex_conf, self.gen_tex * self.gen_tex_conf)
            self.loss += self.losses["tex"]

        with  tf.contrib.summary.create_file_writer(self.output_dir).as_default():
            for key in self.losses:
                tf.contrib.summary.scalar('loss/%s'%key, self.losses[key], step=global_step)

        self.summary_ops = tf.contrib.summary.all_summary_ops()

    def generate_textures(self, generated_image, step=0):
        with tf.name_scope('gen_textures'):
            self.ref_tex = tf.get_variable(
                'ref_tex', shape=(self.batch_size, self.texture_size, self.texture_size, 3), dtype='float32', initializer=tf.initializers.zeros())
            self.gen_tex = tf.get_variable(
                'gen_tex', shape=(self.batch_size, self.texture_size, self.texture_size, 3), dtype='float32', initializer=tf.initializers.zeros())

            self.ref_tex_coord = tf.get_variable(
                'ref_tex_coord', shape=(4, self.batch_size, self.texture_size, self.texture_size, 3), dtype='int32', initializer=tf.initializers.zeros())
            self.gen_tex_coord = tf.get_variable(
                'gen_tex_coord', shape=(4, self.batch_size, self.texture_size, self.texture_size, 3), dtype='int32', initializer=tf.initializers.zeros())
            
            self.ref_tex_weight = tf.get_variable(
                'ref_tex_weight', shape=(4, self.batch_size, self.texture_size, self.texture_size, 3), dtype='float32', initializer=tf.initializers.zeros())
            self.gen_tex_weight = tf.get_variable(
                'gen_tex_weight', shape=(4, self.batch_size, self.texture_size, self.texture_size, 3), dtype='float32', initializer=tf.initializers.zeros())
            
            self.ref_tex_conf = tf.get_variable(
                'ref_tex_conf', shape=(self.batch_size, self.texture_size, self.texture_size, 3), dtype='float32', initializer=tf.initializers.zeros())
            self.gen_tex_conf = tf.get_variable(
                'gen_tex_conf', shape=(self.batch_size, self.texture_size, self.texture_size, 3), dtype='float32', initializer=tf.initializers.zeros())

            self.add_placeholder("ref_tex_coord")
            self.add_placeholder("gen_tex_coord")

            self.add_placeholder("ref_tex_weight")
            self.add_placeholder("gen_tex_weight")
            
            self.add_placeholder("ref_tex_conf")
            self.add_placeholder("gen_tex_conf")

            # analysis generated images
            # start = time.time()
            gen_imgs = self.sess.run(generated_image)
            # print('> gen imgs: %f ms'%(time.time() - start)*1000)
            # start = time.time()
            tex_coord, gen_weights, conf_map = gen_textures(
                gen_imgs, self.face_models, self.texture_size, self.tex_output_dir, 'gen_imgs_%d'%(self.sess.run(step)))

            if (tex_coord is None):
                ref_tex = tf.zeros((self.batch_size, self.texture_size, self.texture_size, 3))
                gen_tex = tf.zeros((self.batch_size, self.texture_size, self.texture_size, 3))
                return ref_tex, gen_tex

            self.assign_placeholder("gen_tex_coord", tex_coord)
            self.assign_placeholder("gen_tex_weight", gen_weights)
            self.assign_placeholder("gen_tex_conf", conf_map)
            # print('> gen textures: %f ms'%(time.time() - start))

            self.ref_tex =  tf.gather_nd(self.ref_img, self.ref_tex_coord[0]) * self.ref_tex_weight[0] + \
                            tf.gather_nd(self.ref_img, self.ref_tex_coord[1]) * self.ref_tex_weight[1] + \
                            tf.gather_nd(self.ref_img, self.ref_tex_coord[2]) * self.ref_tex_weight[2] + \
                            tf.gather_nd(self.ref_img, self.ref_tex_coord[3]) * self.ref_tex_weight[3]
            print('>>> ref_tex.shape = ', self.ref_tex.shape)

            generated_image = tf.cast(generated_image, tf.float32)
            self.gen_tex =  tf.gather_nd(generated_image, self.gen_tex_coord[0]) * self.gen_tex_weight[0] + \
                            tf.gather_nd(generated_image, self.gen_tex_coord[1]) * self.gen_tex_weight[1] + \
                            tf.gather_nd(generated_image, self.gen_tex_coord[2]) * self.gen_tex_weight[2] + \
                            tf.gather_nd(generated_image, self.gen_tex_coord[3]) * self.gen_tex_weight[3]
            print('>>> gen_tex.shape = ', self.gen_tex.shape)

    def generate_face_mask(self, im):
        from imutils import face_utils
        import cv2
        rects = self.detector(im, 1)
        # loop over the face detections
        for (j, rect) in enumerate(rects):
            """
            Determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
            """
            shape = self.predictor(im, rect)
            shape = face_utils.shape_to_np(shape)

            # we extract the face
            vertices = cv2.convexHull(shape)
            mask = np.zeros(im.shape[:2],np.uint8)
            cv2.fillConvexPoly(mask, vertices, 1)
            if self.use_grabcut:
                bgdModel = np.zeros((1,65),np.float64)
                fgdModel = np.zeros((1,65),np.float64)
                rect = (0,0,im.shape[1],im.shape[2])
                (x,y),radius = cv2.minEnclosingCircle(vertices)
                center = (int(x),int(y))
                radius = int(radius*self.scale_mask)
                mask = cv2.circle(mask,center,radius,cv2.GC_PR_FGD,-1)
                cv2.fillConvexPoly(mask, vertices, cv2.GC_FGD)
                cv2.grabCut(im,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
                mask = np.where((mask==2)|(mask==0),0,1)
            return mask

    def set_reference_images(self, images_list, step=0):
        assert(len(images_list) != 0 and len(images_list) <= self.batch_size)
        loaded_image = load_images(images_list, self.img_size)
        image_features = None
        if self.perceptual_model is not None:
            image_features = self.perceptual_model.predict_on_batch(preprocess_input(loaded_image))
            weight_mask = np.ones(self.features_weight.shape)

        if self.tex_loss is not None:
            # print('> loaded images:', loaded_image.max(), loaded_image.min(), loaded_image.dtype)
            tex_coord, weight, conf_map = gen_textures(
                loaded_image, self.face_models, self.texture_size, self.output_dir, 'ref_imgs_%d'%step)
            
            if tex_coord is not None:
                self.assign_placeholder("ref_tex_coord", tex_coord)
                self.assign_placeholder("ref_tex_weight", weight)
                self.assign_placeholder("ref_tex_conf", conf_map)

        if self.face_mask:
            image_mask = np.zeros(self.ref_weight.shape)

            for (i, im) in enumerate(loaded_image):
                try:
                    _, img_name = os.path.split(images_list[i])
                    mask_img = os.path.join(self.mask_dir, f'{img_name}')
                    if (os.path.isfile(mask_img)):
                        print("Loading mask " + mask_img)
                        imask = PIL.Image.open(mask_img).convert('L')
                        mask = np.array(imask)/255
                        mask = np.expand_dims(mask,axis=-1)
                    else:
                        mask = self.generate_face_mask(im)
                        imask = (255*mask).astype('uint8')
                        imask = PIL.Image.fromarray(imask, 'L')
                        print("Saving mask " + mask_img)
                        imask.save(mask_img, 'PNG')
                        mask = np.expand_dims(mask,axis=-1)
                    mask = np.ones(im.shape,np.float32) * mask

                except Exception as e:
                    print("Exception in mask handling for " + mask_img)
                    traceback.print_exc()
                    mask = np.ones(im.shape[:2],np.uint8)
                    mask = np.ones(im.shape,np.float32) * np.expand_dims(mask,axis=-1)
                image_mask[i] = mask
            img = None
        else:
            image_mask = np.ones(self.ref_weight.shape)

        if len(images_list) != self.batch_size:
            if image_features is not None:
                features_space = list(self.features_weight.shape[1:])
                existing_features_shape = [len(images_list)] + features_space
                empty_features_shape = [self.batch_size - len(images_list)] + features_space
                existing_examples = np.ones(shape=existing_features_shape)
                empty_examples = np.zeros(shape=empty_features_shape)
                weight_mask = np.vstack([existing_examples, empty_examples])
                image_features = np.vstack([image_features, np.zeros(empty_features_shape)])

            images_space = list(self.ref_weight.shape[1:])
            existing_images_space = [len(images_list)] + images_space
            empty_images_space = [self.batch_size - len(images_list)] + images_space
            existing_images = np.ones(shape=existing_images_space)
            empty_images = np.zeros(shape=empty_images_space)
            image_mask = image_mask * np.vstack([existing_images, empty_images])
            loaded_image = np.vstack([loaded_image, np.zeros(empty_images_space)])

        if image_features is not None:
            self.assign_placeholder("features_weight", weight_mask)
            self.assign_placeholder("ref_img_features", image_features)
        self.assign_placeholder("ref_weight", image_mask)
        self.assign_placeholder("ref_img", loaded_image)

    def optimize(self, vars_to_optimize, iterations=200):
        vars_to_optimize = vars_to_optimize if isinstance(vars_to_optimize, list) else [vars_to_optimize]
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        min_op = optimizer.minimize(self.loss, var_list=[vars_to_optimize])

        self.sess.run(tf.variables_initializer(optimizer.variables()))
        self.sess.run(self._reset_global_step)
        fetch_ops = [min_op, self.loss, self.learning_rate, self.summary_ops]

        for i in range(iterations):
            output_losses = {}
            for key in self.losses:
                output_losses[key] = (self.sess.run([self.losses[key]]))[0]

            _, loss, lr, _ = self.sess.run(fetch_ops)

            output_losses["loss"] = loss
            output_losses["lr"] = lr
            yield output_losses

        print('Save textures...')
        ref_tex = self.sess.run(self.ref_tex).astype(np.uint8)
        tex_output_dir = os.path.join(self.output_dir, 'textures')
        for i in range(ref_tex.shape[0]):
            img = PIL.Image.fromarray(ref_tex[i, :, :, :], 'RGB')
            img.save(os.path.join(tex_output_dir, 'ref_%d.jpg'%(i)))

        gen_tex = self.sess.run(self.gen_tex).astype(np.uint8)
        for i in range(gen_tex.shape[0]):
            img = PIL.Image.fromarray(ref_tex[i, :, :, :], 'RGB')
            img.save(os.path.join(tex_output_dir, 'gen_%d.jpg'%(i)))
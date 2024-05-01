import numpy as np
import tensorflow as tf
import random
import _pickle as pkl
import matplotlib.pyplot as plt
from pylab import rcParams
import scipy
import scipy.stats as stats
from tensorflow.python.ops import gen_nn_ops
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth = True

MEAN_IMAGE = np.zeros((1,227,227,3)).astype(np.float32)
MEAN_IMAGE[:,:,:,0]=103.939
MEAN_IMAGE[:,:,:,1]=116.779
MEAN_IMAGE[:,:,:,2]=123.68
EPSILON = 1e-12
MIN_INPUT = -MEAN_IMAGE
MAX_INPUT = 255 * np.ones_like(MEAN_IMAGE).astype(np.float32) - MEAN_IMAGE

def dataReader():
    X = np.zeros((100,227,227,3))
    y = np.zeros(100)
    for num in range(4):
        with open("./ImagenetValidationSamples/imagenet_sample_{}.pkl".format(num),"rb") as inputs:
            dic_temp = pkl.load(inputs)
            X[num*20:num*20+20] = dic_temp["X"]
            y[num*20:num*20+20] = dic_temp["y"]
            labels = dic_temp["labels"]
    return X, y.astype(int), labels


class SimpleGradientAttack(object):
    
    def __init__(self, mean_image, sess, test_image , original_label, NET, NET2=None, k_top=1000, target_map=None):
        """
        Args:            
            mean_image: The mean image of the data set(The assumption is that the images are mean subtracted)
            sess: Session containing model(and surrogate model's) graphs
            test_image: Mean subtracted test image
            original_label: True label of the image
            NET: Original neural network. It's assumed that NET.saliency is the saliency map tensor and 
            NET.saliency_flatten is its flatten version.
            NET2: Surrogate neural network with the same structure and weights of the orignal network but
            with activations replaced by softplus function
            (necessary only when the activation function of the original function
            does not have second order gradients, ex: ReLU). It's assumed that NET.saliency is the 
            saliency map tensor and NET2.saliency_flatten is its flatten version.
            k_top: the topK parameter of the attack (refer to the original paper)
        """
        if len(test_image.shape)!=3:
            raise ValueError("Invalid Test Image Dimensions")
        if NET.input.get_shape()[-3]!=test_image.shape[-3] or NET.input.get_shape()[-2]!=test_image.shape[-2] or\
        NET.input.get_shape()[-1]!=test_image.shape[-1]:
            raise ValueError("Model's input dimensions is not Compatible with the provided test image!")
        if self.check_prediction(sess,original_label,test_image,NET):
            return
        self.sess = sess
        self.target_map = target_map
        self.create_extra_ops(NET,test_image.shape[-3],test_image.shape[-2],k_top)
        if NET2 is None:
            NET2 = NET
        else:
            self.create_extra_ops(NET2,test_image.shape[-3],test_image.shape[-2],k_top)
            if NET2.input.get_shape()[-3]!=test_image.shape[-3] or NET2.input.get_shape()[-2]!=test_image.shape[-2] or\
            NET2.input.get_shape()[-1]!=test_image.shape[-1]:
                raise ValueError("Surrogate model's input dimensions is not Compatible with the provided test image!")
        self.NET = NET
        self.NET2 = NET2
        self.test_image = test_image
        self.original_label = original_label
        self.mean_image = mean_image
        self.k_top=k_top
        self.saliency1, self.topK = self.run_model(self.sess,[NET.saliency, NET.top_idx], self.test_image, NET)
        self.saliency1_flatten = np.reshape(self.saliency1,[test_image.shape[-3]*test_image.shape[-2]])
        self.create_attack_ops(NET2,test_image.shape[-3],test_image.shape[-2])
            
    def check_prediction(self,sess,original_label,image,NET):
        
        """ If the network's prediction is incorrect in the first place, attacking has no meaning."""
        predicted_scores = sess.run(NET.output,feed_dict={NET.input:image if len(image.shape)==4 else [image]})
        if np.argmax(predicted_scores,1) != original_label:
            print("Network's Prediction is Already Incorrect!")
            return True
        else:
            self.original_confidence = np.max(predicted_scores)
            return False
        
    def create_extra_ops(self,NET,w,h,k_top):
        
        top_val, NET.top_idx = tf.nn.top_k(NET.saliency_flatten,k_top)
        y_mesh, x_mesh = np.meshgrid(np.arange(h), np.arange(w))
        NET.mass_center = tf.stack([tf.reduce_sum(NET.saliency*x_mesh)/(w*h),tf.reduce_sum(NET.saliency*y_mesh)/(w*h)])
        
    def create_attack_ops(self,NET,w,h):
        
        elem1 = np.argsort(np.reshape(self.saliency1,[w*h]))[-self.k_top:]
        self.elements1 = np.zeros(w*h)
        self.elements1[elem1] = 1
        topK_loss = tf.reduce_sum((NET.saliency_flatten*self.elements1))
        self.topK_direction= -tf.gradients(topK_loss, NET.input)[0]
        self.mass_center1 = self.run_model(self.sess, NET.mass_center, self.test_image, NET).astype(int)
        mass_center_loss = -tf.reduce_sum((NET.mass_center-self.mass_center1)**2)
        self.mass_center_direction= -tf.gradients(mass_center_loss, NET.input)[0]
        if self.target_map is not None:
            target_loss = -tf.reduce_sum((NET.saliency * self.target_map))
            self.target_direction = -tf.gradients(target_loss, NET.input)[0]
    
   
    def run_model(self, sess, operation, feed, NET):
        
        if len(feed.shape) == 3:
            return sess.run(operation,feed_dict={NET.input:[feed],NET.label_ph:self.original_label})
        elif len(feed.shape) == 4:
            return sess.run(operation,feed_dict={NET.input:feed,NET.label_ph:self.original_label})
        else:
            raise RuntimeError("Input image shape invalid!") 
        
    def give_simple_perturbation(self,attack_method,in_image):
        
        w,h,c = self.test_image.shape
        if attack_method == "random":
            perturbation = np.random.normal(size=(w,h,c))
        elif attack_method == "topK":
            perturbation = self.run_model(self.sess, self.topK_direction, in_image, self.NET2)
            perturbation = np.reshape(perturbation,[w,h,c])
        elif attack_method == "mass_center":
            perturbation = self.run_model(self.sess, self.mass_center_direction, in_image, self.NET2)
            perturbation = np.reshape(perturbation,[w,h,c])
        elif attack_method == "target":
            if self.target_map is None:
                raise ValueError("No target region determined!")
            else:
                perturbation = self.run_model(self.sess, self.target_direction, in_image, self.NET2)
                perturbation = np.reshape(perturbation,[w,h,c])
        return np.sign(perturbation)

    
    def apply_perturb(self, in_image, pert, alpha, bound=256):
        
        out_image = self.test_image + np.clip(in_image+alpha*pert-self.test_image, -bound, bound)
        out_image = np.clip(out_image, -self.mean_image, 255-self.mean_image)
        return out_image
    
    def check_measure(self, test_image_pert, measure):
        
        prob = self.run_model(self.sess, self.NET.output, test_image_pert, self.NET)
        if np.argmax(prob,1) == self.original_label:
            if measure=="intersection":
                top2 = self.run_model(self.sess, self.NET.top_idx, test_image_pert, self.NET)
                criterion = float(len(np.intersect1d(self.topK,top2)))/self.k_top
            elif measure=="correlation":
                saliency2_flatten = self.run_model(self.sess, self.NET.saliency_flatten, test_image_pert, 
                                                   self.NET)
                criterion = scipy.stats.spearmanr(self.saliency1_flatten,saliency2_flatten)[0]
            elif measure=="mass_center":
                center2 = self.run_model(self.sess, self.NET.mass_center, test_image_pert, self.NET).astype(int)
                criterion = -np.linalg.norm(self.mass_center1-center2)
            else:
                raise ValueError("Invalid measure!")
            return criterion
        else:
            return 1.
        
    def iterative_attack(self, attack_method, epsilon, iters=100, alpha=1,  measure="intersection", target = None):
        """
        Args:
            attack_method: One of "mass_center", "topK" or "random"
            epsilon: Allowed maximum $ell_infty$ of perturbations, eg:8
            iters: number of maximum allowed attack iterations
            alpha: perturbation size in each iteration of the attack
            measure: measure for success of the attack (one of "correlation", "mass_center" or "intersection")
        Returns:
            intersection: The portion of the top K salient pixels in the original picture that are in the 
            top K salient pixels of the perturbed image devided
            correlation: The rank correlation between saliency maps of original and perturbed image
            center_dislocation: The L2 distance between saliency map mass centers in original and perturbed images
            confidence: The prediction confidence of the perturbed image
        """

        w,h,c = self.test_image.shape
        test_image_pert = self.test_image.copy()
        min_criterion = 1.
        perturb_size = 0.
        for counter in range(iters):
            if counter % int(iters/5) == 0:
                print("Iteration : {}".format(counter))
            pert = self.give_simple_perturbation(attack_method, test_image_pert)
            test_image_pert = self.apply_perturb(test_image_pert, pert, alpha, epsilon)
            criterion = self.check_measure(test_image_pert, measure)
            if criterion < min_criterion:
                min_criterion = criterion
                self.perturbed_image = test_image_pert.copy()
                perturb_size = np.max(np.abs(self.test_image-self.perturbed_image))
        if min_criterion==1.:
            print("The attack was not successfull for maximum allowed perturbation size equal to {}".format(epsilon))
            return 1., 1., self.original_confidence, 0.
        print('''For maximum allowed perturbation size equal to {}, the resulting perturbation size was equal to {}
        '''.format(epsilon,np.max(np.abs(self.test_image-self.perturbed_image))))
        predicted_scores = self.run_model(self.sess,self.NET.output, self.perturbed_image, self.NET)
        confidence = np.max(predicted_scores)
        self.saliency2, self.top2, self.mass_center2= self.run_model\
        (self.sess, [self.NET.saliency, self.NET.top_idx, self.NET.mass_center], self.perturbed_image, self.NET)
        correlation = scipy.stats.spearmanr(self.saliency1_flatten, np.reshape(self.saliency2,[w*h]))[0]
        intersection = float(len(np.intersect1d(self.topK, self.top2)))/self.k_top
        center_dislocation = np.linalg.norm(self.mass_center1-self.mass_center2.astype(int))
        return intersection, correlation, center_dislocation, confidence, perturb_size

class IntegratedGradientsAttack(object):
    
    def __init__(self, sess, mean_image, test_image, original_label, NET, NET2=None, 
                 k_top=1000, num_steps=100, reference_image=None, target_map = None):
        """
        Args:
            mean_image: The mean image of the data set(The assumption is that the images are mean subtracted)
            sess: Session containing model(and surrogate model's) graphs
            test_image: Mean subtracted test image
            original_label: True label of the image
            NET: Original neural network. It's assumed that NET.saliency is the saliency map tensor and 
            NET.saliency_flatten is its flatten version.
            NET2: Surrogate neural network with the same structure and weights of the orignal network but
            with activations replaced by softplus function
            (necessary only when the activation function of the original function
            does not have second order gradients, ex: ReLU). It's assumed that NET.saliency is the 
            saliency map tensor and NET2.saliency_flatten is its flatten version.
            k_top: the topK parameter of the attack (refer to the original paper)
            num_steps: Number of steps in Integrated Gradients Algorithm
            reference_image: Mean subtracted reference image of Integrated Gradients Algorithm
        """
        if len(test_image.shape)!=3:
            raise ValueError("Invalid Test Image Dimensions")
        if sum([NET.input.get_shape()[-i]!=test_image.shape[-i] for i in [1,2,3]]):
            raise ValueError("Model's input dimensions is not Compatible with the provided test image!")
        if self.check_prediction(sess,original_label,test_image,NET):
            return
        self.sess = sess
        self.target_map = target_map
        self.create_extra_ops(NET,test_image.shape[-3],test_image.shape[-2],k_top)
        if NET2 is None:
            NET2 = NET
        else:
            self.create_extra_ops(NET2,test_image.shape[-3],test_image.shape[-2],k_top)
            if sum([NET2.input.get_shape()[-i]!=test_image.shape[-i] for i in [1,2,3]]):
                raise ValueError("Surrogate model's input dimensions is not Compatible with the provided test image!")
        self.NET = NET
        self.NET2 = NET2
        self.test_image = test_image
        self.original_label = original_label
        self.mean_image = mean_image
        self.k_top = k_top
        self.num_steps = num_steps
        self.reference_image = np.zeros_like(test_image) if reference_image is None else reference_image
        counterfactuals = self.create_counterfactuals(test_image)
        self.saliency1, self.topK = self.run_model(self.sess,[NET.saliency, NET.top_idx], counterfactuals, NET)
        self.saliency1_flatten = np.reshape(self.saliency1,[test_image.shape[-3]*test_image.shape[-2]])
        self.create_attack_ops(NET2,test_image.shape[-3],test_image.shape[-2])
    
    def check_prediction(self,sess,original_label,image,NET):
        
        """ If the network's prediction is incorrect in the first place, attacking has no meaning."""
        predicted_scores = sess.run(NET.output,feed_dict={NET.input:image if len(image.shape)==4 else [image]})
        if np.argmax(predicted_scores,1) != original_label:
            print("Network's Prediction is Already Incorrect!")
            return True
        else:
            self.original_confidence = np.max(predicted_scores)
            return False
        
    def create_extra_ops(self,NET,w,h,k_top):
        
        top_val, NET.top_idx = tf.nn.top_k(NET.saliency_flatten,k_top)
        y_mesh, x_mesh = np.meshgrid(np.arange(h), np.arange(w))
        NET.mass_center = tf.stack([tf.reduce_sum(NET.saliency*x_mesh)/(w*h),tf.reduce_sum(NET.saliency*y_mesh)/(w*h)])
        
    def create_attack_ops(self,NET,w,h):
        
        elem1 = np.argsort(np.reshape(self.saliency1,[w*h]))[-self.k_top:]
        self.elements1 = np.zeros(w*h)
        self.elements1[elem1] = 1
        topK_loss = tf.reduce_sum((NET.saliency_flatten * self.elements1))
        NET.topK_direction= -tf.gradients(topK_loss, NET.input)[0]
        counterfactuals = self.create_counterfactuals(self.test_image)
        self.mass_center1 = self.run_model(self.sess, NET.mass_center, counterfactuals, NET).astype(int)
        mass_center_loss = -tf.reduce_sum((NET.mass_center - self.mass_center1)**2)
        NET.mass_center_direction= -tf.gradients(mass_center_loss, NET.input)[0]
        if self.target_map is not None:
            target_loss = -tf.reduce_sum((NET.saliency * self.target_map))
            NET.target_direction = -tf.gradients(target_loss, NET.input)[0]
    
    def create_counterfactuals(self, in_image):
        
        ref_subtracted = in_image - self.reference_image
        counterfactuals = np.array([(float(i+1)/self.num_steps) * ref_subtracted + self.reference_image\
                                    for i in range(self.num_steps)])
        return np.array(counterfactuals)
   
    def run_model(self, sess, operation, feed, NET):
        
        if len(feed.shape) == 3:
            return sess.run(operation,feed_dict={NET.input:[feed],NET.label_ph:self.original_label,
                                                NET.reference_image:self.reference_image})
        elif len(feed.shape) == 4:
            return sess.run(operation,feed_dict={NET.input:feed,NET.label_ph:self.original_label,
                                                NET.reference_image:self.reference_image})
        else:
            raise RuntimeError("Input image shape invalid!") 
        
    def give_simple_perturbation(self,attack_method,in_image):
        
        counterfactuals = self.create_counterfactuals(in_image)
        w,h,c = self.test_image.shape
        if attack_method == "random":
            perturbation = np.random.normal(size=(self.num_steps,w,h,c))
        elif attack_method == "topK":
            perturbation = self.run_model(self.sess, self.NET2.topK_direction, counterfactuals, self.NET2)
            perturbation = np.reshape(perturbation,[self.num_steps,w,h,c])
        elif attack_method == "mass_center":
            perturbation = self.run_model(self.sess, self.NET2.mass_center_direction, counterfactuals, self.NET2)
            perturbation = np.reshape(perturbation,[self.num_steps,w,h,c])
        elif attack_method == "target":
            if self.target_map is None:
                raise ValueError("No target region determined!")
            else:
                perturbation = self.run_model(self.sess, self.NET2.target_direction, counterfactuals, self.NET2)
                perturbation = np.reshape(perturbation,[self.num_steps,w,h,c])
        perturbation_summed = np.sum(np.array([float(i+1)/self.num_steps*perturbation[i]\
                                               for i in range(self.num_steps)]),0)
        return np.sign(perturbation_summed)

    
    def apply_perturb(self, in_image, pert, alpha, bound=256):
        
        out_image = self.test_image + np.clip(in_image + alpha*pert - self.test_image, -bound, bound)
        out_image = np.clip(out_image, -self.mean_image, 255 - self.mean_image)
        return out_image
    
    def check_measure(self, test_image_pert, measure):
        
        prob = self.run_model(self.sess, self.NET.output, test_image_pert, self.NET)
        if np.argmax(prob,1) == self.original_label:
            counterfactuals = self.create_counterfactuals(test_image_pert)
            if measure=="intersection":
                top2 = self.run_model(self.sess, self.NET.top_idx, counterfactuals, self.NET)
                criterion = float(len(np.intersect1d(self.topK,top2)))/self.k_top
            elif measure=="correlation":
                saliency2_flatten = self.run_model(self.sess, self.NET.saliency_flatten, counterfactuals, 
                                                   self.NET)
                criterion = scipy.stats.spearmanr(self.saliency1_flatten,saliency2_flatten)[0]
            elif measure=="mass_center":
                center2 = self.run_model(self.sess, self.NET.mass_center, counterfactuals, self.NET).astype(int)
                criterion = -np.linalg.norm(self.mass_center1-center2)
            else:
                raise ValueError("Invalid measure!")
            return criterion
        else:
            return 1.
        
    def iterative_attack(self, attack_method, epsilon, iters=100, alpha=1,  measure="intersection"):
        """
        Args:
            attack_method: One of "mass_center", "topK" or "random"
            epsilon: set of allowed maximum $ell_infty$ of perturbations, eg:[2,4]
            iters: number of maximum allowed attack iterations
            alpha: perturbation size in each iteration of the attack
            measure: measure for success of the attack (one of "correlation", "mass_center" or "intersection")
        Returns:
            intersection: The portion of the top K salient pixels in the original picture that are in the 
            top K salient pixels of the perturbed image devided
            correlation: The rank correlation between saliency maps of original and perturbed image
            center_dislocation: The L2 distance between saliency map mass centers in original and perturbed images
            confidence: The prediction confidence of the perturbed image
        """

        w,h,c = self.test_image.shape
        test_image_pert = self.test_image.copy()
        min_criterion = 1.
        for counter in range(iters):
            if counter % int(iters/5) == 0:
                print("Iteration : {}".format(counter))
            pert = self.give_simple_perturbation(attack_method, test_image_pert)
            test_image_pert = self.apply_perturb(test_image_pert, pert, alpha, epsilon)
            criterion = self.check_measure(test_image_pert, measure)
            if criterion < min_criterion:
                min_criterion = criterion
                self.perturbed_image = test_image_pert.copy()
                perturb_size = np.max(np.abs(self.test_image-self.perturbed_image))
        if min_criterion==1.:
            print("The attack was not successfull for maximum allowed perturbation size equal to {}".format(epsilon))
            return 1., 1., self.original_confidence, 0.
        
        print('''For maximum allowed perturbation size equal to {}, the resulting perturbation size was equal to {}
        '''.format(epsilon,np.max(np.abs(self.test_image-self.perturbed_image))))
        predicted_scores = self.run_model(self.sess,self.NET.output, self.perturbed_image, self.NET)
        confidence = np.max(predicted_scores)
        counterfactuals = self.create_counterfactuals(self.perturbed_image)
        self.saliency2, self.top2, self.mass_center2= self.run_model\
        (self.sess, [self.NET.saliency, self.NET.top_idx, self.NET.mass_center], counterfactuals, self.NET)
        correlation = scipy.stats.spearmanr(self.saliency1_flatten, np.reshape(self.saliency2,[w*h]))[0]
        intersection = float(len(np.intersect1d(self.topK, self.top2)))/self.k_top
        center_dislocation = np.linalg.norm(self.mass_center1-self.mass_center2.astype(int))
        return intersection, correlation, center_dislocation, confidence, perturb_size




class PropagationMethodsAttack(object):
    
    def __init__(self, mean_image, sess, test_image , original_label, NET, k_top=1000, target_map=None):
        """
        Args:
            mean_image: The mean image of the data set(The assumption is that the images are mean subtracted)
            sess: Session containing model(and surrogate model's) graphs
            test_image: Mean subtracted test image
            original_label: True label of the image
            NET: Original neural network
            (necessary only when the activation function of the original function
            does not have second order gradients, ex: ReLU)
            k_top: the topK parameter of the attack
            target_map: a binary 2D map with the same size as the original image to determine the target
            area of the image for the saliency scores to be focused on after perturbation (Only required for
            targeted attack method)
        """
        if len(test_image.shape)!=3:
            raise ValueError("Invalid Test Image Dimensions")
        if NET.input.get_shape()[-3]!=test_image.shape[-3] or NET.input.get_shape()[-2]!=test_image.shape[-2] or\
        NET.input.get_shape()[-1]!=test_image.shape[-1]:
            raise ValueError("Model's input dimensions is not Compatible with the provided test image!")
        if self.check_prediction(sess,original_label,test_image,NET):
            return
        self.target_map = target_map
        self.sess = sess
        self.create_extra_ops(NET,test_image.shape[-3],test_image.shape[-2],k_top)
        self.NET = NET
        self.test_image = test_image
        self.original_label = original_label
        self.mean_image = mean_image
        self.k_top=k_top
        self.saliency1, self.topK = self.run_model(self.sess,[NET.saliency, NET.top_idx], self.test_image, NET)
        self.saliency1_flatten = np.reshape(self.saliency1,[test_image.shape[-3]*test_image.shape[-2]])
        self.create_attack_ops(NET,test_image.shape[-3],test_image.shape[-2])
    
    def check_prediction(self,sess,original_label,image,NET):
        
        """ If the network's prediction is incorrect in the first place, attacking has no meaning."""
        predicted_scores = sess.run(NET.output,feed_dict={NET.input:image if len(image.shape)==4 else [image]})
        if np.argmax(predicted_scores,1) != original_label:
            print("Network's Prediction is Already Incorrect!")
            return True
        else:
            self.original_confidence = np.max(predicted_scores)
            return False
        
    def create_extra_ops(self,NET,w,h,k_top):
        
        top_val, NET.top_idx = tf.nn.top_k(NET.saliency_flatten,k_top)
        y_mesh, x_mesh = np.meshgrid(np.arange(h), np.arange(w))
        NET.mass_center = tf.stack([tf.reduce_sum(NET.saliency*x_mesh)/(w*h),tf.reduce_sum(NET.saliency*y_mesh)/(w*h)])
        
    def create_attack_ops(self,NET,w,h):
        
        elem1 = np.argsort(np.reshape(self.saliency1,[w*h]))[-self.k_top:]
        self.elements1 = np.zeros(w*h)
        self.elements1[elem1] = 1
        topK_loss = tf.reduce_sum((NET.saliency_flatten*self.elements1))
        NET.topK_direction= -tf.gradients(topK_loss, NET.input)[0]
        self.mass_center1 = self.run_model(self.sess, NET.mass_center, self.test_image, NET).astype(int)
        mass_center_loss = -tf.reduce_sum((NET.mass_center-self.mass_center1)**2)
        NET.mass_center_direction= -tf.gradients(mass_center_loss, NET.input)[0]
        if self.target_map is not None:
            target_loss = -tf.reduce_sum((NET.saliency * self.target_map))
            NET.target_direction = -tf.gradients(target_loss, NET.input)[0]
            
    
   
    def run_model(self, sess, operation, feed, NET):
        
        if len(feed.shape) == 3:
            return sess.run(operation,feed_dict={NET.input:[feed],NET.label_ph:self.original_label})
        elif len(feed.shape) == 4:
            return sess.run(operation,feed_dict={NET.input:feed,NET.label_ph:self.original_label})
        else:
            raise RuntimeError("Input image shape invalid!") 
        
    def give_simple_perturbation(self,attack_method,in_image):
        
        w,h,c = self.test_image.shape
        if attack_method == "random":
            perturbation = np.random.normal(size=(w,h,c))
        elif attack_method == "topK":
            perturbation = self.run_model(self.sess, self.NET.topK_direction, in_image, self.NET)
            perturbation = np.reshape(perturbation,[w,h,c])
        elif attack_method == "mass_center":
            perturbation = self.run_model(self.sess, self.NET.mass_center_direction, in_image, self.NET)
            perturbation = np.reshape(perturbation,[w,h,c])
        elif attack_method == "target":
            perturbation = self.run_model(self.sess, self.NET.target_direction, in_image, self.NET)
            perturbation = np.reshape(perturbation,[w,h,c])
        return np.sign(perturbation)

    
    def apply_perturb(self, in_image, pert, alpha, bound=256):
        
        out_image = self.test_image + np.clip(in_image+alpha*pert-self.test_image, -bound, bound)
        out_image = np.clip(out_image, -self.mean_image, 255-self.mean_image)
        return out_image
    
    def check_measure(self, test_image_pert, measure):
        
        prob = self.run_model(self.sess, self.NET.output, test_image_pert, self.NET)
        if np.argmax(prob,1) == self.original_label:
            if measure=="intersection":
                top2 = self.run_model(self.sess, self.NET.top_idx, test_image_pert, self.NET)
                criterion = float(len(np.intersect1d(self.topK,top2)))/self.k_top
            elif measure=="correlation":
                saliency2_flatten = self.run_model(self.sess, self.NET.saliency_flatten, test_image_pert, 
                                                   self.NET)
                criterion = scipy.stats.spearmanr(self.saliency1_flatten,saliency2_flatten)[0]
            elif measure=="mass_center":
                center2 = self.run_model(self.sess, self.NET.mass_center, test_image_pert, self.NET).astype(int)
                criterion = -np.linalg.norm(self.mass_center1-center2)
            else:
                raise ValueError("Invalid measure!")
            return criterion
        else:
            return 1.
        
    def iterative_attack(self, attack_method, epsilon, iters=100, alpha=1,  measure="intersection"):
        """
        Args:
            attack_method: One of "mass_center", "topK" or "random"
            epsilon: set of allowed maximum $ell_infty$ of perturbations, eg:[2,4]
            iters: number of maximum allowed attack iterations
            alpha: perturbation size in each iteration of the attack
            measure: measure for success of the attack (one of "correlation", "mass_center" or "intersection")
        Returns:
            intersection: The portion of the top K salient pixels in the original picture that are in the 
            top K salient pixels of the perturbed image devided
            correlation: The rank correlation between saliency maps of original and perturbed image
            center_dislocation: The L2 distance between saliency map mass centers in original and perturbed images
            confidence: The prediction confidence of the perturbed image
        """

        w,h,c = self.test_image.shape
        test_image_pert = self.test_image.copy()
        min_criterion = 1.
        for counter in range(iters):
            if iters>5 and counter % int(iters/5) == 0:
                print("Iteration : {}".format(counter))
            pert = self.give_simple_perturbation(attack_method, test_image_pert)
            test_image_pert = self.apply_perturb(test_image_pert, pert, alpha, epsilon)
            criterion = self.check_measure(test_image_pert, measure)
            if criterion < min_criterion:
                min_criterion = criterion
                self.perturbed_image = test_image_pert.copy()
                perturb_size = np.max(np.abs(self.test_image-self.perturbed_image))
        if min_criterion == 1.:
            print("The attack was not successfull for maximum allowed perturbation size equal to {}".format(epsilon))
            return 1., 1., self.original_confidence, 0., 0.
        
        print('''For maximum allowed perturbation size equal to {}, the resulting perturbation size was{}
        '''.format(epsilon,perturb_size))
        predicted_scores = self.run_model(self.sess,self.NET.output, self.perturbed_image, self.NET)
        confidence = np.max(predicted_scores)
        self.saliency2, self.top2, self.mass_center2= self.run_model\
        (self.sess, [self.NET.saliency, self.NET.top_idx, self.NET.mass_center], self.perturbed_image, self.NET)
        correlation = scipy.stats.spearmanr(self.saliency1_flatten, np.reshape(self.saliency2,[w*h]))[0]
        intersection = float(len(np.intersect1d(self.topK, self.top2)))/self.k_top
        center_dislocation = np.linalg.norm(self.mass_center1-self.mass_center2.astype(int))
        return intersection, correlation, center_dislocation, confidence, perturb_size


def propagetion(w,h,c,num_classes,sess,model,reference_image, back_window_size=40, mode='DeepLIFT'):
    
    model_layers = model.layers
    model_tensors = []
    layer_names = []
    layer_types = []
    layer_dic = {}
    for layer in model.layers[::-1]:
        layer_names.append(layer.name)
        layer_types.append(type(layer).__name__)
        layer_dic[layer.name] = layer
    expands = [256,256,192,192,128,128,64,64][::-1]
    squeezes = [64,64,48,48,32,32,16,16][::-1]
    model.label_ph = tf.placeholder(tf.int32,shape=())
    mask = tf.one_hot(model.label_ph,1000)
    back = back_tensor(sess,model,layer_dic[layer_names[0]].input, reference_image,mode=mode) * mask
    forward = back
    back = back_tensor(sess,model,layer_dic[layer_names[1]].input, reference_image,mode=mode)
    temp = tf.reshape(tf.constant(np.arange(1000),dtype=tf.int32),(1,1,1,1000))
    m = (tf.constant(1./169*np.ones((1,13,13,1000)),dtype=tf.float32)*\
    tf.cast(tf.equal(temp,tf.reshape(model.label_ph,[-1,1,1,1])),tf.float32))
    epsilon = 1e-20
    for layer_type,layer_name in zip(layer_types[2:66],layer_names[2:66]):
        if layer_name[:4]=="fire":
            num_fire = int(layer_name[4])-2
            if layer_type=="Concatenate":
                forward = back
                back_left = back_tensor(sess,model,layer_dic[layer_name].input[0], reference_image,mode=mode)
                back_right = back_tensor(sess,model,layer_dic[layer_name].input[1], reference_image,mode=mode)
                m_left = m[:,:,:,:expands[num_fire]]
                m_right = m[:,:,:,expands[num_fire]:]
            if layer_type=="Activation":
                if layer_name[-9:]=="expand1x1":
                    forward_left = back_left
                    back_left = back_tensor(sess,model,layer_dic[layer_name].input, reference_image,mode=mode)
                    m_left = m_left*forward_left/(back_left+epsilon)
                elif layer_name[-9:]=="expand3x3":
                    forward_right = back_right
                    back_right = back_tensor(sess,model,layer_dic[layer_name].input, reference_image,mode=mode)
                    m_right = m_right*forward_right/(back_right+epsilon)
                elif layer_name[-10:]=="squeeze1x1":
                    forward = back
                    m = m_left+m_right
                    back = back_tensor(sess,model,layer_dic[layer_name].input, reference_image,mode=mode)
                    m = m * forward/(back + epsilon)
                else:
                    raise ValueError("There is a problem!")
            elif layer_type=="Conv2D":
                if layer_name[-9:]=="expand1x1":
                    forward_left = back_left
                    back = back_tensor(sess, model, layer_dic[layer_name].input, reference_image,mode=mode)
                    m_left = back_conv(m_left, layer_dic[layer_name], forward_left, back, mode = mode)
                elif layer_name[-9:]=="expand3x3":
                    forward_right = back_right
                    back = back_tensor(sess, model, layer_dic[layer_name].input, reference_image,mode=mode)
                    m_right = back_conv(m_right, layer_dic[layer_name], forward_right, back, mode = mode)
                elif layer_name[-10:]=="squeeze1x1":
                    forward = back
                    back = back_tensor(sess, model, layer_dic[layer_name].input, reference_image,mode=mode)
                    m = back_conv(m, layer_dic[layer_name], forward, back, mode = mode)
                else:
                    raise ValueError("There is a problem!")
        else:
            if layer_type =="Activation":
                forward = back
                back = back_tensor(sess,model,layer_dic[layer_name].input,reference_image,mode=mode)
                m = m*forward/(back+epsilon)
            if layer_type == "Conv2D":
                if layer_name != 'conv1' or mode!='DTD':
                    forward = back
                    back = back_tensor(sess,model,layer_dic[layer_name].input,reference_image,mode=mode)
                    m = back_conv(m ,layer_dic[layer_name], forward, back, mode = mode)
                else:
                    forward = back
                    back = back_tensor(sess,model,layer_dic[layer_name].input,reference_image,mode=mode)
                    m = back_conv(m ,layer_dic[layer_name], forward, back, mode = 'DTD_input')
            if layer_type == "MaxPooling2D":
                back = back_tensor(sess,model,layer_dic[layer_name].input,reference_image,mode=mode)
                m = back_maxpool(m ,layer_dic[layer_name], forward, back, back_window_size)
    return m


def run_model(sess, model, tensor, inputs):
    
    if len(inputs.shape) == 3:
        inputs = np.expand_dims(inputs, 0)
    elif len(inputs.shape) == 4:
        pass
    else:
        raise ValueError('Invalid input dimensions!')

    return sess.run(tensor, feed_dict={model.input: inputs})


def back_tensor(sess, model, tensor, reference=None, mode='DeepLIFT'):
    
    if mode=='DeepLIFT':
        out = tensor - run_model(sess, model, tensor, reference)
    else:
        out = tensor
    return out

def back_conv(m, layer, forward=None, back=None, mode='DeepLIFT'):
    
    size = layer.input.shape
    out_size = (1, size[1].value, size[2].value, size[3].value)
    kernel_size = layer.get_config()["kernel_size"]
    padding = layer.get_config()["padding"].upper()
    strides = layer.get_config()["strides"]
    stride = (1, strides[0], strides[1], 1)
    weights, biases = layer.get_weights()
    if mode=='DeepLIFT':
        out = tf.nn.conv2d_transpose(m, weights, out_size, stride, padding)
    elif mode=='LP':
        m = m * (forward/(forward - biases + EPSILON))
        out = tf.nn.conv2d_transpose(m, weights, out_size, stride, padding)
    elif mode=='DTD':
        w_pos = tf.maximum(0., weights)
        forward_plus = tf.nn.conv2d(back, w_pos, stride, padding)
        m = m * forward / (forward_plus + EPSILON)
        out = tf.nn.conv2d_transpose(m, w_pos, out_size, stride, padding)
    elif mode=='DTD_input':
        w_pos = tf.maximum(0., weights)
        w_neg = tf.minimum(0., weights)
        forward_unbiased = forward - biases
        forward_max = tf.nn.conv2d(MAX_INPUT, w_neg, stride, padding)
        forward_min = tf.nn.conv2d(MIN_INPUT, w_pos, stride, padding) 
        m = m * forward / (forward_unbiased - forward_max -  forward_min + EPSILON)
        out_temp = tf.nn.conv2d_transpose(m, weights, out_size, stride, padding)
        hw = tf.nn.conv2d_transpose(m, w_neg, out_size, stride, padding)
        lw = tf.nn.conv2d_transpose(m, w_pos, out_size, stride, padding)
        out = (out_temp * back - lw * MIN_INPUT - hw * MAX_INPUT)
    return out


def back_maxpool(m, layer, forward=None, back=None, shatranj=33):
    
    size = layer.input.shape
    out_size = (1, size[1].value, size[2].value, size[3].value)
    size = layer.output.shape
    in_size = (1, size[1].value, size[2].value, size[3].value)
    in_length = in_size[1] * in_size[2] * in_size[3]
    out_length = out_size[1] * out_size[2] * out_size[3]
    square_size = out_size[1] * out_size[2]
    padding = layer.get_config()["padding"].upper()
    strides = layer.get_config()["strides"]
    stride = (1, strides[0], strides[1], 1)
    pool_sizes = layer.get_config()["pool_size"]
    pool_size = (1, pool_sizes[0], pool_sizes[1], 1)
    _, mask = tf.nn.max_pool_with_argmax(layer.input, pool_size, stride, padding)
    mymap = np.reshape(np.arange(out_length), out_size)
    lst = []
    for counter in range(out_size[3]):
        temp_lst = []
        for index1 in range(int(np.ceil(out_size[1] / shatranj))):
            lst_t = []
            sh1 = index1 * shatranj
            py1 = min((index1 + 1) * shatranj, out_size[1])
            begin1 = max(0, int((sh1 - pool_size[1]+1) / stride[1]))
            end1 = min(in_size[1], int((py1 - pool_size[1]) / stride[1]) + int(np.ceil((pool_size[1]+1)/stride[1])))
            for index2 in range(int(np.ceil(out_size[2] / shatranj))):
                sh2 = index2 * shatranj
                py2 = min((index2 + 1) * shatranj, out_size[2])
                begin2 = max(0, int((sh2 - pool_size[2]+1) / stride[2]))
                end2 = min(in_size[2], int((py2 - pool_size[2]) / stride[2]) + int(np.ceil((pool_size[2]+1)/stride[2])))

                mymap_slice = mymap[:, sh1:py1, sh2:py2, counter:counter + 1]
                mask_slice = mask[:, begin1:end1, begin2:end2, counter:counter + 1]
                m_slice = m[:, begin1:end1, begin2:end2, counter:counter + 1]

                local_map = tf.reshape(tf.reshape(mymap_slice, [-1]), [-1, 1, 1, 1])
                local_mask = tf.cast(tf.equal(local_map, mask_slice), tf.float32)
                lst_t.append(tf.reshape(tf.reduce_sum(local_mask * m_slice, (1, 2, 3)),
                                             [py1 - sh1, py2 - sh2]))
            temp_lst.append(tf.concat(lst_t, axis=-1))
        lst.append(tf.expand_dims(tf.concat(temp_lst, axis=0), 0))
    out = tf.stack(lst, axis=-1)
    return out
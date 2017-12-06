import tensorflow as tf
import numpy as np
from scipy.optimize import fmin_ncg
import matplotlib.pyplot as plt
from numpy.linalg import norm

'''
Author: Abubakar Abid, with input from Amirata Ghorbani and some code adapted from: https://github.com/kohpangwei/influence-release/
'''

class Influence(object):
    '''
    tf_session: the session that contains the trained network
    trainable_weights: a list of all of the trainable weights in your network
    loss: the loss function which the gradient/hessian will be taken with respect to
    inp: the input tensor (to feed values in)
    out: the outpout tensor (to feed labels in)
    X_train: training features
    y_train: training labels
    '''
    def __init__(self, graph, tf_session, trainable_weights, loss, inp, out, X_train, y_train, more_params=dict()):
        # Basic tensors and operations 
        self.trainable_weights = trainable_weights
        self.loss = loss
        self.gradient = tf.gradients(loss, trainable_weights)
        self.tf_session = tf_session
        
        # Tensors and operations used to approximation the HVP
        self.preturb_tensors = list()
        self.preturb_ops = list()
        self.assign_tensors = list()
        self.assign_ops = list()
        
        with graph.as_default():
            for weight in trainable_weights:
                self.preturb_tensors.append(tf.placeholder(tf.float32, weight.get_shape().as_list()))
                self.preturb_ops.append(tf.assign_add(weight,self.preturb_tensors[-1]))
                self.assign_tensors.append(tf.placeholder(tf.float32, weight.get_shape().as_list()))
                self.assign_ops.append(tf.assign(weight,self.assign_tensors[-1]))

            self.original_weights = list()
            for weight in trainable_weights:
                weight_value = self.tf_session.run(weight)
                self.original_weights.append(weight_value)
        
            # Training data
            self.X_train = X_train
            self.y_train = y_train
            self.x = inp
            self.y_ = out
        
        self.graph = graph
        self.more_params = more_params
        
        # Gradients
        self.cached_training_gradients = [None] * X_train.shape[0]
        self.hess = None
        self.inverse_hessians = dict()
        
    # just a useful helper method to combine outputs of operations into one array
    def list_of_arrays_to_vector(self, arrays):
        return np.concatenate([a.flatten() for a in arrays])
    
    # a helper method to evaluate gradients with respect to certain data
    def evaluate_gradients(self, X,y):
        feed_dict = {**self.more_params} # copies the dictionary
        feed_dict[self.x] = X
        feed_dict[self.y_] = y
        eval_gradients = self.tf_session.run(self.gradient,feed_dict=feed_dict)
        eval_gradients = self.list_of_arrays_to_vector(eval_gradients)
        return eval_gradients
    
    # just a useful helper method to print basic stats about a vector
    def print_summary(self, v):
        v = np.array(v)
        print("Max:",v.max(),"Min:",v.min(),"Mean:",v.mean(),"Size:",v.size(),"# Non-zero:",np.count_nonzero(v))
            
    '''
    Calculates the gradient of training examples [start_idx: start_idx+num_examples] with respect to the parameters of the model
    Caches the results to save future computation (could be useful if the number of params is a lot...)
    '''
    def gradient_of_training_example_wrt_weights(self, start_idx, num_examples=1, verbose=False):
        # only check cache if num_examples is 1, that simplifies things just a little bit
        if (num_examples==1) and not(self.cached_training_gradients[start_idx] is None):
            return self.cached_training_gradients[start_idx]
        # if there is no cache...
        eval_gradients = self.evaluate_gradients(self.X_train[start_idx:start_idx+num_examples], self.y_train[start_idx:start_idx+num_examples])
        self.cached_training_gradients[start_idx] = eval_gradients
        if (verbose):
            self.print_summary(eval_gradients)
        return eval_gradients 
    
    '''
    Calculates the gradient of test examples [start_idx: start_idx+num_examples] with respect to the parameters of the model
    '''
    def gradient_of_test_example_wrt_weights(self, X_test, y_test, verbose=False):
        eval_gradients = self.evaluate_gradients(X_test.reshape(1,-1), y_test.reshape(1,-1))
        if (verbose):
            self.print_summary(eval_gradients)
        return eval_gradients
    
    # A helper method that preturbs the trainable_weights by a certain preturbation vector whose length should equal num of params
    def preturb_weights(self, preturbation):
        t_index = 0
        for j, weights in enumerate(self.trainable_weights):
            shape = weights.get_shape().as_list()
            size = np.product(shape)
            pret = preturbation[t_index:t_index+size].reshape(shape)
            self.tf_session.run(self.preturb_ops[j], feed_dict={self.preturb_tensors[j]:pret})
            t_index += size
    
    def restore_weights(self):
        for j, weights in enumerate(self.trainable_weights):
            self.tf_session.run(self.assign_ops[j], feed_dict={self.assign_tensors[j]:self.original_weights[j]})
        
    
    '''
    Approximates the Hessian vector product of the Hessian against an arbitrary vector t, of dimensionality equal to the number of params.
    The Hessian here is the empirical Hessian, averaged over all of the training examples (this seems to work best, although scaling shouldn't affect the results theoretically).
    params 'start_idx' and 'num_examples' are only used for testing purposes
    '''
    def approx_hvp(self, t, start_idx=0, num_examples=None, r= 0.001):
        preturbation = np.array(r*t) #calculate the preturbation
        #print("Pret:",preturbation[:5])
        if not(num_examples):
            num_examples = self.X_train.shape[0]

        # positive preturbation
        self.preturb_weights(preturbation)        
        #print("Before Eval Grad:",self.tf_session.run(self.trainable_weights)[0].flatten()[:5])
        plus_gradients = self.evaluate_gradients(self.X_train[start_idx:start_idx+num_examples], self.y_train[start_idx:start_idx+num_examples])/num_examples
        #print("After Eval Grad:",self.tf_session.run(self.trainable_weights)[0].flatten()[:5])

        # negative preturbation (two-sided approximation is more numerically stable)
        self.preturb_weights(-2*preturbation)
        #print("Minus preturb:",self.tf_session.run(self.trainable_weights)[0].flatten()[:5])
        minus_gradients = self.evaluate_gradients(self.X_train[start_idx:start_idx+num_examples], self.y_train[start_idx:start_idx+num_examples])/num_examples
        #print("Minus preturb post grad:",self.tf_session.run(self.trainable_weights)[0].flatten()[:5])

        #restore to base weights
        #self.preturb_weights(preturbation)
        self.restore_weights()

        hvp = (plus_gradients-minus_gradients)/(2*r)
        #print("End of loop:",self.tf_session.run(self.trainable_weights)[0].flatten()[:5])
        return hvp

    def get_cached_inverse_hessian(self, damping):
        if damping in self.inverse_hessians:
            return self.inverse_hessians[damping]
        else:
            hess = self.get_cached_hessian()
            damped_hess = hess + damping * np.identity(hess.shape[0])
            inverse_hessian = np.linalg.inv(damped_hess)
            self.inverse_hessians[damping] = inverse_hessian
            return inverse_hessian    
    
    def get_cached_hessian(self):
        if not(self.hess is None):
            hess = self.hess
        else:
            with self.graph.as_default():
                hess = self.tf_session.run(tf.hessians(self.loss, self.trainable_weights),feed_dict={self.x:self.X_train,self.y_:self.y_train})
                self.hess = hess
        if (len(hess)>1):
            print("Warning: only Hessians with respect to the first trainable weight will be computed")
        hess = hess[0]/self.X_train.shape[0]
        return hess
    
    def compute_exact_influence(self, X_test, y_test, damping=0.01, idx_train=None):
        v = self.gradient_of_test_example_wrt_weights(X_test, y_test)
        inverse_hessian = self.get_cached_inverse_hessian(damping)
        ihvp = inverse_hessian.dot(v)
        
        influences = list()
        for j in range(self.X_train.shape[0]):
            g = self.gradient_of_training_example_wrt_weights(j)
            influence = ihvp.dot(g)
            influences.append(influence)

        self.influences = np.array(influences)/norm(np.array(influences))
        if not(idx_train is None):
            return self.influences[idx_train]
        return self.influences
                 
    '''
    Wrapper functions for Newton-CG solver, which is needed to avoid computing the inverse of the Hessian
    '''
    def get_fmin_loss_fn(self, v):
        def get_fmin_loss(x):
            hessian_vector_val = self.approx_hvp(x)
            return 0.5 * np.dot(hessian_vector_val, x) - np.dot(v, x)
        return get_fmin_loss


    def get_fmin_grad_fn(self, v):
        def get_fmin_grad(x):
            hessian_vector_val = self.approx_hvp(x)
            return hessian_vector_val - v
        return get_fmin_grad


    def get_fmin_hvp(self, x, p):
        hessian_vector_val = self.approx_hvp(p)
        return hessian_vector_val
    
    
    def print_objective_callback(self,X_test, y_test,verbose):
        v = self.gradient_of_test_example_wrt_weights(X_test,y_test)
        def print_objective(x):
            if verbose:
                self.n_iters += 1
                print('\r',end="Iter #"+str(self.n_iters)+", Objective value:"+str(self.get_fmin_loss_fn(v)(x)))
        return print_objective
    
    # --------- End wrapper functions for Newton-CG solver
    
    '''
    This is the primary function for this class. It computes the influences, across all training examples of the test
    data point that is provided:
    '''
    def compute_influence(self, X_test, y_test, verbose=True, max_iters=100):
        self.n_iters = 0
        v = self.gradient_of_test_example_wrt_weights(X_test, y_test)
        #print(v)
        print("Gradient of test example has been computed")
        #print(self.get_fmin_loss_fn(v)(v))
        
        #v = np.random.normal(0,1,v.size)
        
        influences = list()
        fmin_results = fmin_ncg(
            f=self.get_fmin_loss_fn(v),
            x0=v,
            fprime=self.get_fmin_grad_fn(v),
            fhess_p=self.get_fmin_hvp,
            avextol=1e-8,
            maxiter=max_iters,
            full_output=verbose,
            callback=self.print_objective_callback(X_test, y_test,verbose),
            retall=True) 
        
        ihvp = fmin_results[0] #inverse Hessian vector product
        influences = list()
        for j in range(self.X_train.shape[0]):
            g = self.gradient_of_training_example_wrt_weights(start_idx=j)
            influence = ihvp.dot(g)
            influences.append(influence)
        
        self.influences = np.array(influences)
        if (verbose):
            print("Influences computed")
        
        return self.influences
    
    def gradient_ascent_on_influence(self, X_test, y_test, idx_train, damping=0.01):
        hess = self.get_cached_hessian()
        inverse_hessian = self.get_cached_inverse_hessian(damping)            
        grad_train = self.gradient_of_training_example_wrt_weights(idx_train)
        ihvp = inverse_hessian.dot(grad_train).flatten()
        
        with self.graph.as_default():
            ihvp_tensor = tf.placeholder(tf.float32, shape=ihvp.shape)
            grad_test_op = tf.gradients(self.loss, self.trainable_weights)
            influence_op = tf.reduce_sum(tf.multiply(ihvp_tensor, grad_test_op))
            grad_ascent_op = tf.gradients(influence_op, self.x)

            feed_dict = {**self.more_params} # copies the dictionary
            feed_dict[self.x] = X_test.reshape(1,-1)
            feed_dict[self.y_] = y_test.reshape(1,-1)
            feed_dict[ihvp_tensor] = ihvp

            gradient_values = self.tf_session.run(grad_ascent_op, feed_dict=feed_dict)
            
        return gradient_values[0].flatten()

    def test_hvp_additivity(self, verbose=False):
        v = self.gradient_of_training_example_wrt_weights(0)
        w_rand = np.random.normal(0,1,v.size)
        hvp1 = self.approx_hvp(w_rand,start_idx=0,num_examples=1)
        hvp2 = self.approx_hvp(w_rand,start_idx=1,num_examples=1)
        hvp_sum = self.approx_hvp(w_rand,start_idx=0,num_examples=2)
        sum_hvp = hvp1 + hvp2
        if np.allclose(hvp_sum, sum_hvp):
            print("The approximator is additive :(")
        elif np.allclose(2*hvp_sum, sum_hvp):
            print("The approximator is averagitive :)")
        elif np.allclose(hvp_sum, 2*sum_hvp):
            print("The approximator is ... anti-additive :(")
        else:
            print("The approximator failed all tests :(")
        if (verbose):
            print("HVP1",hvp1)
            print("HVP2",hvp2)
            print("HVP Sum",hvp_sum)
            print("Sum HVP",sum_hvp)
    
    
    def box_plot(self):
        plt.figure()
        plt.boxplot(self.influences)
    
    def get_most_neutral_least(self,N=5, return_influence_values_for_most=False):
        idxs_most = np.argpartition(self.influences, -N, axis=0)[-N:]
        idxs_most = idxs_most[np.argsort(self.influences[idxs_most])]
        
        non_influences = np.square(np.zeros(self.influences.shape)-self.influences)
        idxs_neutral = np.argpartition(non_influences, N, axis=0)[:N]
        idxs_neutral = idxs_neutral[np.argsort(non_influences[idxs_neutral])]

        idxs_least = np.argpartition(self.influences, N, axis=0)[:N]
        idxs_least = idxs_least[np.argsort(self.influences[idxs_least])]
        if return_influence_values_for_most:
            return idxs_most, np.round(self.influences[idxs_most],2)
        return (idxs_most, idxs_neutral, idxs_least)

def plot_mnist(infl, influences, X_test, N = 5, only_image=False, only_top=False):
    plot_cifar(infl=infl, influences=influences, X_test=X_test, N=N, only_image=only_image, img_shape=(28,28))
    
def plot_cifar(infl, influences, X_test, N = 5, img_shape = (32,32,3), only_image=False):
    plt.figure()
    plt.imshow(X_test.reshape(img_shape),cmap='gray')
    plt.title("Test Image")
    if (only_image):
        return
    most, neutral, least = infl.get_most_neutral_least(N=N)
    plt.figure(figsize=[15,4])
    for i, idx in enumerate(most):
        plt.subplot(1,5,i+1)
        plt.imshow(infl.X_train[idx].reshape(img_shape), cmap='gray')
        plt.title(str(idx)+" Most Influential: " + str(np.round(influences[idx],2)))
    return #JUST FOR NOW
    plt.figure(figsize=[15,4])
    for i, idx in enumerate(neutral):
        plt.subplot(1,5,i+1)
        plt.imshow(infl.X_train[idx].reshape(img_shape), cmap='gray')
        plt.title(str(idx)+"Most Neutral: " + str(np.round(influences[idx],2)))
    plt.figure(figsize=[15,4])
    for i, idx in enumerate(least):
        plt.subplot(1,5,i+1)
        plt.imshow(infl.X_train[idx].reshape(img_shape), cmap='gray')
        plt.title(str(idx)+" Most Harmful: " + str(np.round(influences[idx],2)))    

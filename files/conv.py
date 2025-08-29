import tensorflow as tf
import math
from datetime import datetime
from utils import *
import argparse

class ConvNet(object):
	def __init__(self, version=1, restore=False, saved_ckpt_path=None, yparams=3, width=200, height=200, depth=1, lr=1e-3, filter_size=5, nfilters=32):
		self.width = width
		self.height = height
		self.depth = depth
		self.yparams = yparams
		self.lr = lr
		self.nfilters = nfilters
		self.filter_size = filter_size
		self.version = version
		self.build_graph()
		self.init_session(restore, saved_ckpt_path)

	def build_graph(self):
		self.graph = tf.Graph()
		with self.graph.as_default():
			self._define_placeholders()
			self._simple_graph()
			self._add_loss_op()
			self.init = tf.global_variables_initializer()
			self.saver = tf.train.Saver()

	def _define_placeholders(self):
		self.X = tf.placeholder(tf.float32, [None, self.width, self.height, self.depth], 'X')
		self.y = tf.placeholder(tf.int64, [None, self.yparams], 'y')
		self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

	def _simple_graph(self, hidden_dim=1024, stride=1, pool_stride=2, pool_size=2):
		data = self.X
		xavier = tf.contrib.layers.xavier_initializer(dtype=tf.float64)

		N = 4 # N conv-relu-pool layers
		for n in range(N):
			W, b = "Wc%d" % (n+1), "bc%d" % (n+1) 
			if self.version == 1: W += "-1"; b += "-1"
			layer = "Conv%d" % (n+1)
			with tf.variable_scope(layer):
				Wc = tf.get_variable(W, initializer=xavier,
					shape=[self.filter_size,self.filter_size,data.shape[3],self.nfilters])
				tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, Wc)
				bc = tf.get_variable(b, shape=[self.nfilters], initializer=tf.constant_initializer(0))
				conv = tf.nn.conv2d(data, Wc, strides=[1,stride,stride,1], padding='SAME') + bc
				relu = tf.nn.relu(conv)
				norm = tf.contrib.layers.batch_norm(relu, center=True, scale=True)
				pool = tf.nn.max_pool(norm, [1,pool_size,pool_size,1], 
					strides=[1,pool_stride,pool_stride,1], padding='VALID')
				data = pool
		
		M = 2 # M affine-relu layers
		aff_input_dim = int(data.shape[1])**2 * self.nfilters
		for m in range(M):
			W, b, layer = "Wa%d" % (m+1), "ba%d" % (m+1), "Affine%d" % (m+1)
			with tf.variable_scope(layer):
				Wa = tf.get_variable(W, shape=[aff_input_dim,hidden_dim], initializer=xavier)
				tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, Wa)
				ba = tf.get_variable(b, shape=[hidden_dim], initializer=tf.constant_initializer(0))
				data_flat = tf.reshape(data, [-1, aff_input_dim])
				aff = tf.matmul(data_flat, Wa) + ba
				relu = tf.nn.relu(aff)
				if self.version == 1:
					norm = tf.contrib.layers.batch_norm(relu, center=True, scale=True)
				else:
					norm = relu
				drop = tf.nn.dropout(norm, self.keep_prob)
				data = drop
				aff_input_dim = hidden_dim
		
		with tf.variable_scope("Affine_final"):
			Wf = tf.get_variable("W_final", shape=[hidden_dim,self.yparams], initializer=xavier)
			tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, Wf)
			bf = tf.get_variable("b_final", shape=[self.yparams], initializer=tf.constant_initializer(0))
			data_flat = tf.reshape(data, [-1, hidden_dim])
			y_out = tf.matmul(data_flat, Wf) + bf
		
		self.preds = y_out

	def _add_loss_op(self, reg_scale=1e-6):
		regularizer = tf.contrib.layers.l2_regularizer(scale=reg_scale)
		reg_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
		reg_loss = tf.contrib.layers.apply_regularization(regularizer, reg_vars)
		self.loss = tf.losses.mean_squared_error(labels=self.y, predictions=self.preds) + reg_loss
		optimizer = tf.train.AdamOptimizer(self.lr)
		self.train_op = optimizer.minimize(self.loss)    

	def init_session(self, restore, saved_ckpt_path):
		self.sess = tf.Session(graph=self.graph)
		if restore: 
			self.saver.restore(self.sess, saved_ckpt_path)
		else: 
			self.sess.run(self.init)

	def train(self, train_data, train_labels, val_data, val_labels, nepochs=1, keep_prob=0.80, batch_size=64, print_every=100, save_dir='/files/model-1'):
		print("Beginning training...")
		iter_count = 0
		for e in range(nepochs):
			losses, results = [], []
			for i in range(int(math.ceil(train_data.shape[0]/batch_size))):
				# Batch training
				start_idx = (i * batch_size) % train_data.shape[0]
				X_batch = train_data[start_idx:start_idx+batch_size, :]
				y_batch = train_labels[start_idx:start_idx+batch_size, :]
				loss, preds, _ = self.sess.run([self.loss, self.preds, self.train_op], 
					feed_dict={self.X:X_batch, self.y:y_batch, self.keep_prob:1.0})         
				losses.append(loss * X_batch.shape[0])

				if iter_count % print_every == 0:
					train_acc = accuracy(preds, y_batch)
					print("{0}: iteration {1}: batch training loss = {2:.3g}, acc = {3:.3g}".format(
						datetime.now().strftime("%m/%d/%Y-%H:%M:%S"), iter_count, loss, train_acc))
				
				iter_count += 1

			# Report stats every epoch
			train_loss = np.sum(losses) / train_data.shape[0]
			val_loss, val_preds = self.sess.run([self.loss, self.preds], 
				feed_dict={self.X:val_data, self.y:val_labels, self.keep_prob:1.0})
			val_acc = accuracy(val_preds, val_labels)

			print("Epoch {0}, Train loss = {1:.3g}, Val loss = {2:.3g}, Val Acc = {3:.3g}".format(
				e+1, train_loss, val_loss, val_acc))

			# Save
			ckpt_path = self.saver.save(self.sess, os.path.join(save_dir, "model.ckpt-%d" % e))
			print("Model checkpoint (epoch {0}, iteration {1}) saved to file '{2}'".format(
				e+1, iter_count, ckpt_path))

	def serve_predictions(self, data):
		return self.sess.run(self.preds, {self.X:data, self.keep_prob:1.0})


def get_data(n=1000, width=200, height=200, channels=1, yparams=3):
	X, y = np.zeros((n, width, height, channels)), np.zeros((n, yparams))
	for i in range(n):
		params, img = noisy_circle(200, 50, 2)
		X[i] = np.reshape(img, [width, height, channels])
		y[i] = params
	return X, y

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--saved_ckpt_path', type=str, default=None)
	args = parser.parse_args()

	nn = ConvNet(restore=args.saved_ckpt_path!=None, saved_ckpt_path=args.saved_ckpt_path)
	train_x, train_y = get_data(n=64000)
	val_x, val_y = get_data(n=1000)
	nn.train(train_x, train_y, val_x, val_y, nepochs=25)


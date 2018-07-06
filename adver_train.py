# -*- coding:utf-8 -*-
import WGAN as wgan
from params import Params as pm
from data_loader import Data_loader
import tensorflow as tf
import linear as linear
import conv1d as conv1d
import numpy as np


class WGAN(object):
	def __init__(self, data_path, batch_size, epochs, vocab_size, seq_length, embed_dims, dis_epoch, example_num, learning_rate, charmap, lamb=10):
		self.data_path = data_path
		self.batch_size = batch_size
		self.epochs = epochs
		self.vocab_size = vocab_size
		self.seq_length = seq_length
		self.embed_dims = embed_dims
		self.dis_epoch = dis_epoch
		self.example_num = example_num
		self.lamb = lamb
		self.learning_rate = learning_rate
		self.charmap = charmap

		# input_settings
		self.real_inputs_discrete = tf.placeholder(tf.int32, shape=[self.batch_size, self.seq_length])
		self.real_inputs = tf.one_hot(self.real_inputs_discrete, len(self.charmap), 1.0, 0.0)
		self.fake_inputs = self.Generator(self.batch_size, self.charmap)
		self.fake_inputs_discrete = tf.argmax(self.fake_inputs, self.fake_inputs.get_shape().ndims-1)

		disc_real = self.Discriminator(self.real_inputs, self.charmap)
		disc_fake = self.Discriminator(self.fake_inputs, self.charmap)

		self.disc_loss = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
		self.gen_loss = -tf.reduce_mean(disc_fake)

		# WGAN lipschitz-penalty
		epsilon = tf.random_uniform(
			shape=[self.batch_size, 1, 1],
			minval=0.0,
			maxval=1.0
		)
		differences = self.fake_inputs - self.real_inputs
		interpolates = self.real_inputs + epsilon * differences
		gradients = tf.gradients(self.Discriminator(interpolates, self.charmap), [interpolates])[0]
		slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
		gradients_penalty = tf.reduce_mean((slopes - 1.0) ** 2)
		self.disc_loss += lamb * gradients_penalty

		# gen_params = linear._params
		gen_params = wgan.params_with_name("Generator")
		disc_params = wgan.params_with_name("Discriminator")

		self.gen_train_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9).minimize(self.gen_loss, var_list=gen_params)
		self.disc_train_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9).minimize(self.disc_loss, var_list=disc_params)

	def print_model_settings(self, locals):
		print("Uppercase local vars:")
		all_vars = [(k, v) for (k, v) in locals.items() if (k.isupper and k != 'T' and k != "SETTINGS" and k != "ALL_SETTINGS")]
		all_vars = sorted(all_vars, key=lambda x: x[0])
		for var_name, var_value in all_vars:
			print("\t{}: {}".format(var_name, var_value))

	def Generator(self, n_samples, charmap):
		output = self.make_noise(shape=[n_samples, 128])
		output = linear.Linear("Generator.Input", 128, self.seq_length * self.embed_dims, output)
		output = tf.reshape(output, [-1, self.embed_dims, self.seq_length])
		output = self.ResBlock("Generator.1", output)
		output = self.ResBlock("Generator.2", output)
		output = self.ResBlock("Generator.3", output)
		output = self.ResBlock("Generator.4", output)
		output = self.ResBlock("Generator.5", output)
		output = conv1d.Conv1D("Generator.Output", self.embed_dims, len(charmap), 1, output)
		output = tf.transpose(output, perm=[0, 2, 1])
		output = self.softmax(output, charmap)
		return output

	def Discriminator(self, inputs, charmap):
		output = tf.transpose(inputs, [0, 2, 1])
		output = conv1d.Conv1D("Discriminator.Input", len(charmap), self.embed_dims, 1, output)
		output = self.ResBlock("Discriminator.1", output)
		output = self.ResBlock("Discriminator.2", output)
		output = self.ResBlock("Discriminator.3", output)
		output = self.ResBlock("Discriminator.4", output)
		output = self.ResBlock("Discriminator.5", output)
		output = tf.reshape(output, [-1, self.seq_length])
		output = linear.Linear("Discriminator.Output", self.seq_length, 1, output)
		return output

	def make_noise(self, shape):
		return tf.random_normal(shape, stddev=0.1)

	def ResBlock(self, name, inputs):
		output = inputs
		output = tf.nn.relu(output)
		output = conv1d.Conv1D(name + ".1", self.embed_dims, self.embed_dims, 5, output)
		output = tf.nn.relu(output)
		output = conv1d.Conv1D(name + ".2", self.embed_dims, self.embed_dims, 5, output)
		return inputs + (0.3 * output)

	def softmax(self, logits, charmap):
		return tf.reshape(tf.nn.softmax(tf.reshape(logits, [-1, len(charmap)])), tf.shape(logits))


def generate_samples(sess, model, inv_charmap):
	samples = sess.run(model.fake_inputs)
	samples = np.argmax(samples, axis=2)
	decoded_samples = []
	for i in range(len(samples)):       # batch_size
		decoded = []
		for j in range(len(samples[i])):        # seq_length
			decoded.append(inv_charmap[samples[i][j]])
		decoded_samples.append(tuple(decoded))
	return decoded_samples


if __name__ == "__main__":
	data_loader = Data_loader(pm.batch_size)

	lines, charmap, inv_charmap = data_loader.load_datasets(
		max_length=pm.seq_length,
		example_num=pm.example_num,
		vocab_size=pm.vocab_size,
		data_path=pm.data_path
	)

	model = WGAN(pm.data_path, pm.batch_size, pm.epochs, pm.vocab_size, pm.seq_length, pm.embed_dims, pm.dis_epochs, pm.example_num, pm.learning_rate, charmap, pm.lamb)

	if len(pm.data_path) == 0:
		raise Exception("Please specify path to data directory in adver_train.py!")

	# model.print_model_settings(locals().copy())

	data_loader.mini_batch()

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(pm.epochs):

			# train generator
			if epoch > 0:
				_ = sess.run(model.gen_train_optimizer)

			# train critic
			disc_loss = 0
			for i in range(pm.dis_epochs):
				batch = data_loader.next_batch()
				disc_loss, _ = sess.run([model.disc_loss, model.disc_train_optimizer], feed_dict={model.real_inputs_discrete: batch})

			if epoch % 100 == 0:
				samples = []
				for i in range(10):
					samples.extend(generate_samples(sess, model, inv_charmap))

				with open("samples_{}.txt".format(epoch), 'w') as f:
					for s in samples:
						s = "".join(s)
						f.write(s + '\n')
						f.flush()
				print("MSG : Epoch = [{}/{}], Loss = {}".format(epoch, pm.epochs, disc_loss))

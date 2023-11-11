import numpy as np
import pandas as pd
import warnings
import time
from parallel_pandas import ParallelPandas

warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

ParallelPandas.initialize(n_cpu=4, disable_pr_bar=True)

K.clear_session()
tf.compat.v1.reset_default_graph()
tf.keras.utils.set_random_seed(0)
tf.config.experimental.enable_op_determinism()
#tf.reset_default_graph()

def model_1(ser):
	import numpy as np
	import pandas as pd
	import os, sys
	import tensorflow as tf
	from tensorflow import keras
	from tensorflow.keras.models import Sequential
	from tensorflow.keras import backend as K
	from tensorflow.keras.layers import SimpleRNN, GRU, Dense, LSTM, Bidirectional, ConvLSTM1D, TimeDistributed, MultiHeadAttention, Normalization, Conv1D, Reshape, Conv1DTranspose, IntegerLookup
	from tensorflow.keras.optimizers import Adam, Ftrl, AdamW, Nadam, Adagrad, Adafactor, SGD, Adamax, Lion, Adadelta
	from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
	from tensorflow.keras.losses import Loss, MeanAbsolutePercentageError, Huber, Poisson, MeanAbsoluteError, CosineSimilarity, SparseCategoricalCrossentropy
	from tensorflow.keras.activations import gelu, swish, exponential, selu
	#from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution
	from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer, minmax_scale, LabelEncoder, RobustScaler, StandardScaler
	from sklearn.feature_selection import SelectKBest, r_regression, mutual_info_regression
	from sklearn.neighbors import KNeighborsClassifier
	import datetime
	import time
	from scipy.interpolate import splrep, splev
	from tensorflow.keras.utils import get_custom_objects
	from tensorflow.keras.regularizers import L1L2, L2
	import tensorflow_probability as tfp
	import tensorflow_addons as tfa
	from sklearn.linear_model import LinearRegression
	from scipy import stats
	from tsfresh import extract_features
	from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters
	from tsfresh.utilities.dataframe_functions import roll_time_series
	from tsfresh.utilities.dataframe_functions import make_forecasting_frame
	import tensorflow_models as tfm


	#enable_eager_execution()

	y_pred_org_1 = 0
	try:
		#keys = Your feature column names here as a list of strings
		price_pd_org = pd.read_csv('features_ETH-USD.csv', index_col=False, names=keys, header=None, dtype=object, error_bad_lines=False, warn_bad_lines=False).head(500)
		price_pd_org = price_pd_org.apply(pd.to_numeric, errors = 'coerce')
		price_pd_org['prediction'] = np.nan
		price_pd = price_pd_org.loc[ser.index]
		price_pd = price_pd.drop_duplicates(subset=['Time'])
		current_time = float(price_pd['Time'].iloc[-1])
		current_price = float(price_pd.Price.iloc[-1])
		trend = float(price_pd.Trend.iloc[-1])
		residual_max_pos = float(price_pd.residual_max_pos_last_arr3.iloc[-1])
		residual_max_neg = float(price_pd.residual_max_neg_last_arr3.iloc[-1])

		final_y = price_pd.Price.pct_change().shift(-1).fillna(0)
		price_pd['final_y'] = final_y
		new_row = price_pd.tail(1)
		price_pd.drop(price_pd.tail(1).index,inplace=True)
		price_pd = price_pd.loc[price_pd.final_y.apply(lambda x: 1 if x>0 else -1).shift(-1) != price_pd.final_y.apply(lambda x: 1 if x>0 else -1)]
		price_pd['final_y'] = price_pd.Price.pct_change().shift(-1).fillna(0)
		price_pd = price_pd.append(new_row)

		columns_1 = price_pd.columns

		total_pc = 1
		
		#'''
		price_pd_temp = price_pd
		zc = 6
		while True:
			try:
				price_pd = price_pd_temp.tail(zc)
				zc += 1
				section_price = price_pd.Price.to_numpy().astype(float)
				section_time = np.arange(section_price.size)
				section_price_hat = splrep(section_time,section_price,k=3,s=5)
				section_price_hat = splev(section_time,section_price_hat)
				section_der = np.diff(section_price_hat)
				zero_crossings = np.argwhere(np.diff(np.sign(section_der))).flatten().astype(int)
				if zero_crossings.size >= total_pc or len(price_pd) == len(price_pd_temp):
					break
			except Exception as e:
				exc_type, exc_obj, exc_tb = sys.exc_info()
				fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
				print(exc_type, fname, exc_tb.tb_lineno)
				print(e)
				break

		price_pd = price_pd.reset_index(drop=True)

		zero_crossings = np.sort(np.unique(zero_crossings))
		try:
			rows_back = int(section_price_hat.size-zero_crossings[-total_pc])
		except:
			rows_back = len(price_pd)

		if rows_back < 3:
			rows_back = 3
		#'''

		
		price_pd = price_pd.tail(rows_back)
		price_pd = price_pd.reset_index(drop=True)
		price_pd["id"] = price_pd.index

		extraction_settings = MinimalFCParameters()

		final_y = price_pd.final_y
		
		price_pd = extract_features(price_pd.drop(columns=['final_y']), column_id='id', column_sort='Time', default_fc_parameters=extraction_settings, show_warnings=False, disable_progressbar=True)

		selector = SelectKBest(r_regression, k=10)
		selector.fit(price_pd.fillna(0), final_y.fillna(0))
		cols_idxs = selector.get_support(indices=True)
		price_pd = price_pd.iloc[:,cols_idxs]

		price_pd['final_y'] = final_y

		final_y_numpy_final = price_pd.final_y.to_numpy().flatten()
		final_y_numpy = final_y_numpy_final[:-1]

		max_fy = np.amax(final_y_numpy)
		min_fy = np.amin(final_y_numpy)
		if np.abs(max_fy) > np.abs(min_fy):
			max_fy = np.abs(max_fy)
			min_fy = -np.abs(max_fy)
		elif np.abs(max_fy) < np.abs(min_fy):
			max_fy = np.abs(min_fy)
			min_fy = -np.abs(min_fy)

		price_pd = price_pd.replace([np.inf, -np.inf], np.nan).fillna(0)

		columns_1 = price_pd.columns
		scaler = QuantileTransformer(output_distribution='normal')
		price_pdx = scaler.fit_transform(price_pd.iloc[:,:-1])
		price_pdx = pd.DataFrame(data=price_pdx, columns=columns_1[:-1])
		scalery = QuantileTransformer(output_distribution='normal')
		price_pdy = scalery.fit_transform(price_pd.iloc[:-1,-1:])
		price_pdy_scaled = price_pdy
		price_pdy = pd.DataFrame(data=price_pdy, columns=columns_1[-1:])
		price_pdy = price_pdy.append(pd.DataFrame(data=np.array([0]), columns=columns_1[-1:]), ignore_index=True)
		price_pd = pd.concat([price_pdx, price_pdy], axis=1)
		price_pd = price_pd.reset_index(drop=True)
		price_pd = price_pd.replace([np.inf, -np.inf], np.nan).fillna(0)
		dataset = price_pd


		Y_train = dataset.iloc[:-1,-1:]
		X_train = dataset.iloc[:-1,:-1]

		columns_1 = dataset.columns

		Y_train = dataset.iloc[:-1,-1:]
		X_train = dataset.iloc[:-1,:-1]
		to_Predict = dataset.iloc[-1:,:-1]

		
		column_length = len(X_train.columns)
		l = len(X_train)
		lr = l
		final_samples = lr

		to_Predict = dataset.iloc[-lr:,:-1]

		to_Predict = np.add(np.clip(np.multiply(to_Predict.to_numpy().flatten(), 10000000).astype(int), -100000, 100000), 100000)

		state = np.add(np.clip(np.multiply(X_train.to_numpy().flatten(), 10000000).astype(int), -100000, 100000), 100000)
		reward = np.add(np.clip(np.multiply(Y_train.to_numpy().flatten(), 10000000).astype(int), -100000, 100000), 100000)
		vocab = np.arange(0, 200001, 1).astype(int)
		vocab = np.delete(vocab, np.where(vocab == [-1]), axis=0)
		vocab = tf.convert_to_tensor(vocab)
		state = tf.convert_to_tensor(state)
		reward = tf.convert_to_tensor(reward)
		vocab_layer = IntegerLookup(vocabulary=vocab)
		state = vocab_layer(state)
		reward = vocab_layer(reward)
		to_Predict = vocab_layer(to_Predict).numpy().astype(int).reshape((lr, column_length))
		i_vocab_layer = IntegerLookup(vocabulary=vocab, invert=True)
		vocab = vocab_layer(vocab).numpy().flatten()
		state = state.numpy().astype(int).reshape((lr, column_length))
		reward = reward.numpy().astype(int).reshape((lr, 1))
		vocab_size = vocab.size+1

		def positional_encoding(length, depth):
			depth = depth/2

			positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
			depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

			angle_rates = 1 / (10000**depths)         # (1, depth)
			angle_rads = positions * angle_rates      # (pos, depth)

			pos_encoding = np.concatenate(
				[np.sin(angle_rads), np.cos(angle_rads)],
				axis=-1)

			return tf.cast(pos_encoding, dtype=tf.float32)

		class PositionalEmbedding(tf.keras.layers.Layer):
			def __init__(self, vocab_size, d_model):
				super().__init__()
				self.d_model = d_model
				self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=1) 
				self.pos_encoding = positional_encoding(length=lr, depth=column_length)

			def compute_mask(self, *args, **kwargs):
				return self.embedding.compute_mask(*args, **kwargs)

			def call(self, x):
				length = tf.shape(x)
				x = self.embedding(x)
				# This factor sets the relative scale of the embedding and positonal_encoding.
				x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
				x = x + tf.expand_dims(self.pos_encoding, axis=-1)
				return x

		class PositionalEmbedding_Decoder(tf.keras.layers.Layer):
			def __init__(self, vocab_size, d_model):
				super().__init__()
				self.d_model = d_model
				self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=1) 
				self.pos_encoding = positional_encoding(length=1, depth=1)

			def compute_mask(self, *args, **kwargs):
				return self.embedding.compute_mask(*args, **kwargs)

			def call(self, x):
				length = tf.shape(x)
				x = self.embedding(x)
				# This factor sets the relative scale of the embedding and positonal_encoding.
				x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
				x = x + self.pos_encoding
				return x

		class BaseAttention(tf.keras.layers.Layer):
			def __init__(self, **kwargs):
				super().__init__()
				self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
				self.layernorm = tf.keras.layers.LayerNormalization()
				self.add = tf.keras.layers.Add()

		class CrossAttention(BaseAttention):
			def call(self, x, context):
				attn_output, attn_scores = self.mha(
					query=x,
					key=context,
					value=context,
					return_attention_scores=True)

				# Cache the attention scores for plotting later.
				self.last_attn_scores = attn_scores

				x = self.add([x, attn_output])
				x = self.layernorm(x)

				return x


		class GlobalSelfAttention(BaseAttention):
			def call(self, x):
				attn_output = self.mha(
					query=x,
					value=x,
					key=x)
				x = self.add([x, attn_output])
				x = self.layernorm(x)
				return x

		class CausalSelfAttention(BaseAttention):
			def call(self, x):
				attn_output = self.mha(
					query=x,
					value=x,
					key=x,
					use_causal_mask = True)
				x = self.add([x, attn_output])
				x = self.layernorm(x)
				return x

		class FeedForward(tf.keras.layers.Layer):
			def __init__(self, d_model, dff, dropout_rate=0):
				super().__init__()
				self.seq = tf.keras.Sequential([
				tf.keras.layers.Dense(dff, activation='relu'),
				tf.keras.layers.Dense(d_model, activation='linear'),
				tf.keras.layers.Dropout(dropout_rate)
				])
				self.add = tf.keras.layers.Add()
				self.layer_norm = tf.keras.layers.LayerNormalization()

			def call(self, x):
				x = self.add([x, self.seq(x)])
				x = self.layer_norm(x) 
				return x

		class EncoderLayer(tf.keras.layers.Layer):
			def __init__(self,*, d_model, num_heads, dff, dropout_rate):
				super().__init__()

				self.self_attention = GlobalSelfAttention(
					num_heads=num_heads,
					key_dim=d_model,
					dropout=dropout_rate)

				self.ffn = FeedForward(d_model, dff)

			def call(self, x):
				x = self.self_attention(x)
				x = self.ffn(x)
				return x

		class Encoder(tf.keras.layers.Layer):
			def __init__(self, *, num_layers, d_model, num_heads,
						dff, vocab_size, dropout_rate):
				super().__init__()

				self.d_model = d_model
				self.num_layers = num_layers

				self.pos_embedding = PositionalEmbedding(
					vocab_size=vocab_size, d_model=d_model)

				self.enc_layers = [
					EncoderLayer(d_model=d_model,
								num_heads=num_heads,
								dff=dff,
								dropout_rate=dropout_rate)
					for _ in range(num_layers)]
				self.dropout = tf.keras.layers.Dropout(dropout_rate)

			def call(self, x):
				# `x` is token-IDs shape: (batch, seq_len)
				x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

				# Add dropout.
				x = self.dropout(x)

				for i in range(self.num_layers):
					x = self.enc_layers[i](x)

				return x  # Shape `(batch_size, seq_len, d_model)`.	

		class DecoderLayer(tf.keras.layers.Layer):
			def __init__(self,
						*,
						d_model,
						num_heads,
						dff,
						dropout_rate):
				super(DecoderLayer, self).__init__()

				self.causal_self_attention = CausalSelfAttention(
					num_heads=num_heads,
					key_dim=d_model,
					dropout=dropout_rate)

				self.cross_attention = CrossAttention(
					num_heads=num_heads,
					key_dim=d_model,
					dropout=dropout_rate)

				self.ffn = FeedForward(d_model, dff)

			def call(self, x, context):
				x = self.causal_self_attention(x=x)
				x = self.cross_attention(x=x, context=context)

				# Cache the last attention scores for plotting later
				self.last_attn_scores = self.cross_attention.last_attn_scores

				x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
				return x

		class Decoder(tf.keras.layers.Layer):
			def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
						dropout_rate):
				super(Decoder, self).__init__()

				self.d_model = d_model
				self.num_layers = num_layers

				self.pos_embedding = PositionalEmbedding_Decoder(vocab_size=vocab_size,
														d_model=d_model)
				self.dropout = tf.keras.layers.Dropout(dropout_rate)
				self.dec_layers = [
					DecoderLayer(d_model=d_model, num_heads=num_heads,
								dff=dff, dropout_rate=dropout_rate)
					for _ in range(num_layers)]

				self.last_attn_scores = None

			def call(self, x, context):
				# `x` is token-IDs shape (batch, target_seq_len)
				x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

				x = self.dropout(x)

				for i in range(self.num_layers):
					x  = self.dec_layers[i](x, context)

				self.last_attn_scores = self.dec_layers[-1].last_attn_scores

				# The shape of x is (batch_size, target_seq_len, d_model).
				return x

		class Transformer(tf.keras.Model):
			def __init__(self, *, n_gradients, labels, vocab, num_layers, d_model, num_heads, dff,
						input_vocab_size, target_vocab_size, dropout_rate):
				super().__init__()
				self.n_gradients = tf.constant(n_gradients, dtype=tf.int32)
				self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
				self.gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False) for v in self.trainable_variables]

				self.labels = labels
				self.vocab = vocab
				self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
									num_heads=num_heads, dff=dff,
									vocab_size=input_vocab_size,
									dropout_rate=dropout_rate)

				self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
									num_heads=num_heads, dff=dff,
									vocab_size=target_vocab_size,
									dropout_rate=dropout_rate)


				self.final_layer = tf.keras.layers.Dense(vocab_size, activation='linear')

			def call(self, inputs):
				# To use a Keras model with `.fit` you must pass all your inputs in the
				# first argument.
				context = inputs
				x = tf.convert_to_tensor(np.arange(lr).reshape((lr, 1)))

				context = self.encoder(context)  # (batch_size, context_len, d_model)

				x = self.decoder(x, context)  # (batch_size, target_len, d_model)

				# Final linear layer output.
				logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

				try:
					del logits._keras_mask
				except AttributeError:
					pass
				return logits

			def apply_accu_gradients(self):
				# apply accumulated gradients
				self.optimizer.apply_gradients(zip(self.gradient_accumulation, self.trainable_variables))

				# reset
				self.n_acum_step.assign(0)
				for i in range(len(self.gradient_accumulation)):
					self.gradient_accumulation[i].assign(tf.zeros_like(self.trainable_variables[i], dtype=tf.float32))
					
			def train_step(self, data):
				self.n_acum_step.assign_add(1)

				x, y = data
				# Gradient Tape
				with tf.GradientTape() as tape:
					y_pred = self(x, training=True)
					loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
				# Calculate batch gradients
				gradients = tape.gradient(loss, self.trainable_variables)
				# Accumulate batch gradients
				for i in range(len(self.gradient_accumulation)):
					self.gradient_accumulation[i].assign_add(gradients[i])
		
				# If n_acum_step reach the n_gradients then we apply accumulated gradients to update the variables otherwise do nothing
				tf.cond(tf.equal(self.n_acum_step, self.n_gradients), self.apply_accu_gradients, lambda: None)

				# update metrics
				self.compiled_metrics.update_state(y, y_pred)
				return {m.name: m.result() for m in self.metrics}


		tensors = (tf.convert_to_tensor(state.reshape((lr, column_length))), tf.convert_to_tensor(reward.reshape((lr, 1))))


		num_layers = 1
		d_model = 1
		dff = 1
		num_heads = lr
		dropout_rate = 0

		transformer = Transformer(
			n_gradients=1,
			labels=reward.flatten(),
			vocab=vocab,
			num_layers=num_layers,
			d_model=d_model,
			num_heads=num_heads,
			dff=dff,
			input_vocab_size=vocab_size,
			target_vocab_size=vocab_size,
			dropout_rate=dropout_rate)

		plateau = 1/lr
		if plateau == 1:
			plateau = .99
		earlyStopping = ReduceLROnPlateau(monitor='loss', factor=plateau, patience=1, verbose=0)
		lr_schedule = keras.optimizers.schedules.ExponentialDecay(
			initial_learning_rate=1.0,
			decay_steps=lr,
			decay_rate=1/lr)
		opt = Adadelta(learning_rate=lr_schedule)
		opt = tfm.optimization.ExponentialMovingAverage(opt)
		opt.shadow_copy(transformer)
		transformer.compile(
			loss='mse',
			optimizer=opt)
		train_dataset = tf.data.Dataset.from_tensors(tensors)
		history = transformer.fit(train_dataset, batch_size=lr, shuffle=False, epochs=lr, verbose=0, callbacks=[earlyStopping])		
		hist_loss = np.asarray(history.history['loss'])[-1]
		to_Predict[np.isnan(to_Predict)] = 0
		to_predict_shape = tuple(list(to_Predict.shape))
		opt.swap_weights()
		y_pred_org_1 = np.squeeze(transformer.predict_step(to_Predict.reshape(to_predict_shape)))
		final_y_numpy = final_y_numpy[-lr:]

		y_pred_org_1 = vocab[np.argmax(y_pred_org_1, axis=-1)]
		y_pred_org_1 = y_pred_org_1.flatten()[-1]
		y_pred_org_0 = i_vocab_layer(int(y_pred_org_1)).numpy()
		y_pred_org_1 = np.divide(np.subtract(y_pred_org_1, 100000), 10000000)

		y_pred_org_1 = np.array([y_pred_org_1])
		price_pdy_scaled = price_pdy_scaled.flatten()
		price_pdy_scaled = np.append(price_pdy_scaled[:price_pdy_scaled.size-y_pred_org_1.size], y_pred_org_1[-y_pred_org_1.size:])
		y_pred_org_1 = scalery.inverse_transform(price_pdy_scaled.reshape(-1, 1)).flatten()
		y_pred_org_1 = y_pred_org_1[-1]
		
		del transformer
		K.clear_session()
		tf.compat.v1.reset_default_graph()
		tf.keras.utils.set_random_seed(0)
		tf.config.experimental.enable_op_determinism()
		gc.collect()
		
		print('Data_Predict')
		print(datetime.datetime.fromtimestamp(current_time*1000000000))
		print(current_price)
		print(y_pred_org_1)
		print(trend)
		print(residual_max_pos)
		print(residual_max_neg)
		print(hist_loss)
		print(total_pc)
		print(rows_back)


	except Exception as e:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)
		print(e)

	return y_pred_org_1
	

if __name__ == '__main__':
	start = time.time()
	#keys = Your feature column names here as a list of strings
	price_pd_org = pd.read_csv('features_ETH-USD.csv', index_col=False, names=keys, header=None, dtype=object, error_bad_lines=False, warn_bad_lines=False).head(500)
	price_pd_org = price_pd_org.apply(pd.to_numeric, errors = 'coerce')
	price_pd_org['prediction'] = np.nan
	price_pd_org.fillna(0).Time.rolling(25).p_apply(model_1, raw=False, executor='processes')
	end = time.time()
	print((end-start)/60)
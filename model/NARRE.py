"""
NARRE

@author:
Chong Chen (cstchenc@163.com)

@created:
27/8/2017
@references:
Chong Chen, Min Zhang, Yiqun Liu, và Shaoping Ma. 2018. Neural Attentional Rating Regression with Review-level Explanations. In WWW'18.
"""

import tensorflow as tf

class NARRE(object):
    def __init__(
            self, review_num_u, review_num_i, review_len_u, review_len_i, user_num, item_num, num_classes,
            user_vocab_size, item_vocab_size, n_latent, embedding_id, attention_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Define placeholders
        self.input_u = tf.compat.v1.placeholder(tf.int32, [None, review_num_u, review_len_u], name="input_u")
        self.input_i = tf.compat.v1.placeholder(tf.int32, [None, review_num_i, review_len_i], name="input_i")
        self.input_reuid = tf.compat.v1.placeholder(tf.int32, [None, review_num_u], name='input_reuid')
        self.input_reiid = tf.compat.v1.placeholder(tf.int32, [None, review_num_i], name='input_reiid')
        self.input_y = tf.compat.v1.placeholder(tf.float32, [None, 1], name="input_y")
        self.input_uid = tf.compat.v1.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_iid = tf.compat.v1.placeholder(tf.int32, [None, 1], name="input_iid")
        self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")
        self.drop0 = tf.compat.v1.placeholder(tf.float32, name="dropout0")
        
        # Define variables
        iidW = tf.Variable(tf.random.uniform([item_num + 2, embedding_id], -0.1, 0.1), name="iidW")
        uidW = tf.Variable(tf.random.uniform([user_num + 2, embedding_id], -0.1, 0.1), name="uidW")
        
        self.uidW2 = tf.Variable(tf.constant(0.1, shape=[user_num + 2]), name="uidW2")
        self.iidW2 = tf.Variable(tf.constant(0.1, shape=[item_num + 2]), name="iidW2")
        
        # Initialize L2 loss
        l2_loss = tf.constant(0.0)

        
        # User Embedding
        with tf.name_scope("user_embedding"):
            self.W1 = tf.Variable(
                tf.random.uniform([user_vocab_size, embedding_size], -1.0, 1.0),
                name="W1")
            self.embedded_user = tf.nn.embedding_lookup(self.W1, self.input_u)
            self.embedded_users = tf.expand_dims(self.embedded_user, -1)

        # Item Embedding
        with tf.name_scope("item_embedding"):
            self.W2 = tf.Variable(
                tf.random.uniform([item_vocab_size, embedding_size], -1.0, 1.0),
                name="W2")
            self.embedded_item = tf.nn.embedding_lookup(self.W2, self.input_i)
            self.embedded_items = tf.expand_dims(self.embedded_item, -1)

        # User Convolution and Pooling
        pooled_outputs_u = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope(f"user_conv-maxpool-{filter_size}"):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.compat.v1.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                
                # Reshape embedded_users
                self.embedded_users = tf.reshape(self.embedded_users, [-1, review_len_u, embedding_size, 1])

                conv = tf.nn.conv2d(
                    self.embedded_users,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, review_len_u - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_u.append(pooled)
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool_u = tf.concat(pooled_outputs_u, axis=3)  # Sửa cách dùng tf.concat
        self.h_pool_flat_u = tf.reshape(self.h_pool_u, [-1, review_num_u, num_filters_total])

        # Item Convolution and Pooling
        pooled_outputs_i = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope(f"item_conv-maxpool-{filter_size}"):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.compat.v1.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                
                # Reshape embedded_items
                self.embedded_items = tf.reshape(self.embedded_items, [-1, review_len_i, embedding_size, 1])
                conv = tf.nn.conv2d(
                    self.embedded_items,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, review_len_i - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_i.append(pooled)
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool_i = tf.concat(pooled_outputs_i, axis=3)  # Sửa cách dùng tf.concat
        self.h_pool_flat_i = tf.reshape(self.h_pool_i, [-1, review_num_i, num_filters_total])
        
        # Dropout
        with tf.name_scope("dropout"):
            self.h_drop_u = tf.nn.dropout(self.h_pool_flat_u, self.drop0)
            self.h_drop_i = tf.nn.dropout(self.h_pool_flat_i, self.drop0)
        
        # Attention Mechanism
        with tf.name_scope("attention"):
            # Define attention weights
            Wau = tf.Variable(tf.random.uniform([num_filters_total, attention_size], -0.1, 0.1), name='Wau')
            Wru = tf.Variable(tf.random.uniform([embedding_id, attention_size], -0.1, 0.1), name='Wru')
            Wpu = tf.Variable(tf.random.uniform([attention_size, 1], -0.1, 0.1), name='Wpu')
            bau = tf.Variable(tf.constant(0.1, shape=[attention_size]), name="bau")
            bbu = tf.Variable(tf.constant(0.1, shape=[1]), name="bbu")

            # Sanitize self.input_reuid for User Attention
            sanitized_reuid = tf.where(
                tf.less(self.input_reuid, tf.shape(iidW)[0]),
                self.input_reuid,
                tf.zeros_like(self.input_reuid)
            )
            self.iid_a = tf.nn.embedding_lookup(iidW, sanitized_reuid)

            self.u_j = tf.einsum(
                'ajk,kl->ajl',
                tf.nn.relu(
                    tf.einsum('ajk,kl->ajl', self.h_drop_u, Wau) +
                    tf.einsum('ajk,kl->ajl', self.iid_a, Wru) +
                    bau
                ),
                Wpu
            ) + bbu  # None*u_len*1

            self.u_a = tf.nn.softmax(self.u_j, axis=1)  # none*u_len*1
            tf.print("User attention weights:", self.u_a)

            # Define item attention weights
            Wai = tf.Variable(tf.random.uniform([num_filters_total, attention_size], -0.1, 0.1), name='Wai')
            Wri = tf.Variable(tf.random.uniform([embedding_id, attention_size], -0.1, 0.1), name='Wri')
            Wpi = tf.Variable(tf.random.uniform([attention_size, 1], -0.1, 0.1), name='Wpi')
            bai = tf.Variable(tf.constant(0.1, shape=[attention_size]), name="bai")
            bbi = tf.Variable(tf.constant(0.1, shape=[1]), name="bbi")

            # Sanitize self.input_reiid for Item Attention
            sanitized_reiid = tf.where(
                tf.less(self.input_reiid, tf.shape(uidW)[0]),
                self.input_reiid,
                tf.zeros_like(self.input_reiid)
            )
            self.uid_a = tf.nn.embedding_lookup(uidW, sanitized_reiid)

            self.i_j = tf.einsum(
                'ajk,kl->ajl',
                tf.nn.relu(
                    tf.einsum('ajk,kl->ajl', self.h_drop_i, Wai) +
                    tf.einsum('ajk,kl->ajl', self.uid_a, Wri) +
                    bai
                ),
                Wpi
            ) + bbi  # none*len*1

            self.i_a = tf.nn.softmax(self.i_j, axis=1)  # none*len*1
            tf.print("Item attention weights:", self.i_a)

            # L2 Regularization
            l2_loss += tf.nn.l2_loss(Wau)
            l2_loss += tf.nn.l2_loss(Wru)
            l2_loss += tf.nn.l2_loss(Wri)
            l2_loss += tf.nn.l2_loss(Wai)

        # Add Reviews
        with tf.name_scope("add_reviews"):
            self.u_feas = tf.reduce_sum(tf.multiply(self.u_a, self.h_drop_u), axis=1)
            self.u_feas = tf.nn.dropout(self.u_feas, self.dropout_keep_prob)
            self.i_feas = tf.reduce_sum(tf.multiply(self.i_a, self.h_drop_i), axis=1)
            self.i_feas = tf.nn.dropout(self.i_feas, self.dropout_keep_prob)
        
        # Get Features
        with tf.name_scope("get_fea"):
            # Define item and user embedding matrices
            iidmf = tf.Variable(tf.random.uniform([item_num + 2, embedding_id], -0.1, 0.1), name="iidmf")
            uidmf = tf.Variable(tf.random.uniform([user_num + 2, embedding_id], -0.1, 0.1), name="uidmf")
            
            # Sanitize input_iid to ensure indices are within range
            sanitized_iid = tf.where(
                tf.less(self.input_iid, tf.shape(iidmf)[0]),  # Check if input_iid < max index of iidmf
                self.input_iid,                              # If valid, keep the value
                tf.zeros_like(self.input_iid)               # If invalid, replace with 0
            )
            
            # Sanitize input_uid similarly
            sanitized_uid = tf.where(
                tf.less(self.input_uid, tf.shape(uidmf)[0]),  # Check if input_uid < max index of uidmf
                self.input_uid,                              # If valid, keep the value
                tf.zeros_like(self.input_uid)               # If invalid, replace with 0
            )
            
            # Perform embedding lookup with sanitized indices
            self.uid = tf.nn.embedding_lookup(uidmf, sanitized_uid)
            self.iid = tf.nn.embedding_lookup(iidmf, sanitized_iid)
            
            # Reshape embeddings
            self.uid = tf.reshape(self.uid, [-1, embedding_id])
            self.iid = tf.reshape(self.iid, [-1, embedding_id])
            
            # Combine features with latent space
            Wu = tf.Variable(tf.random.uniform([num_filters_total, n_latent], -0.1, 0.1), name='Wu')
            bu = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bu")
            self.u_feas = tf.matmul(self.u_feas, Wu) + self.uid + bu

            Wi = tf.Variable(tf.random.uniform([num_filters_total, n_latent], -0.1, 0.1), name='Wi')
            bi = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bi")
            self.i_feas = tf.matmul(self.i_feas, Wi) + self.iid + bi


        # Neural Collaborative Filtering
        with tf.name_scope('ncf'):
            # Sanitize input_uid to ensure indices are within range
            sanitized_uid = tf.where(
                tf.less(self.input_uid, tf.shape(self.uidW2)[0]),  # Check if input_uid < max index of uidW2
                self.input_uid,                                  # Keep the valid indices
                tf.zeros_like(self.input_uid)                   # Replace invalid indices with 0
            )
            self.FM = tf.multiply(self.u_feas, self.i_feas)
            self.FM = tf.nn.relu(self.FM)

            self.FM=tf.nn.dropout(self.FM,self.dropout_keep_prob)

            Wmul = tf.Variable(
                tf.random.uniform([n_latent, 1], -0.1, 0.1), name='wmul'
            )
            self.mul=tf.matmul(self.FM,Wmul)
            self.score = tf.reduce_sum(self.mul, axis=1, keepdims=True)
            
            # Sanitize input_iid similarly
            sanitized_iid = tf.where(
                tf.less(self.input_iid, tf.shape(self.iidW2)[0]),  # Check if input_iid < max index of iidW2
                self.input_iid,                                   # Keep the valid indices
                tf.zeros_like(self.input_iid)                    # Replace invalid indices with 0
            )

            # User and item biases
            self.u_bias = tf.gather(self.uidW2, sanitized_uid)
            self.i_bias = tf.gather(self.iidW2, sanitized_iid)
            self.Feature_bias = self.u_bias + self.i_bias

            # Bias term
            self.bised = tf.Variable(tf.constant(0.1), name='bias')

            # Predictions
            self.predictions = self.score + self.Feature_bias + self.bised


        # Loss Calculation
        with tf.name_scope("loss"):
            losses = tf.nn.l2_loss(tf.subtract(self.predictions, self.input_y))
            self.loss = losses + l2_reg_lambda * l2_loss

        # Accuracy Metrics
        with tf.name_scope("accuracy"):
            self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.predictions, self.input_y)))
            self.accuracy = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y))))

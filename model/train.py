import tensorflow as tf
import argparse
import numpy as np
import pickle
import datetime
import NARRE
import gensim.downloader as api

# Disable eager execution for TensorFlow 1.x compatibility
tf.compat.v1.disable_eager_execution()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train NARRE Model")
    
    # File paths
    parser.add_argument("--word2vec", type=str, default="../data/google.bin",
                        help="Word2vec file with pre-trained embeddings (default: None)")
    # parser.add_argument("--valid_data", type=str, default="../data/AB/dataset.test",
    #                     help="Data for validation")
    # parser.add_argument("--para_data", type=str, default="../data/AB/dataset.para",
    #                     help="Data parameters")
    # parser.add_argument("--train_data", type=str, default="../data/AB/dataset.train",
    #                     help="Data for training")
    
    # parser.add_argument("--valid_data", type=str, default="../data/DM/dataset.test",
    #                     help="Data for validation")
    # parser.add_argument("--para_data", type=str, default="../data/DM/dataset.para",
    #                     help="Data parameters")
    # parser.add_argument("--train_data", type=str, default="../data/DM/dataset.train",
    #                     help="Data for training")
    
    parser.add_argument("--valid_data", type=str, default="../data/TG/dataset.test",
                        help="Data for validation")
    parser.add_argument("--para_data", type=str, default="../data/TG/dataset.para",
                        help="Data parameters")
    parser.add_argument("--train_data", type=str, default="../data/TG/dataset.train",
                        help="Data for training")
    
    # Model Hyperparameters
    parser.add_argument("--embedding_dim", type=int, default=300,
                        help="Dimensionality of character embedding")
    parser.add_argument("--filter_sizes", type=str, default="3",
                        help="Comma-separated filter sizes")
    parser.add_argument("--num_filters", type=int, default=100,
                        help="Number of filters per filter size")
    parser.add_argument("--dropout_keep_prob", type=float, default=0.5,
                        help="Dropout keep probability")
    parser.add_argument("--l2_reg_lambda", type=float, default=0.001,
                        help="L2 regularization lambda")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Batch Size")
    parser.add_argument("--num_epochs", type=int, default=60,
                        help="Number of training epochs")
    
    # Misc Parameters
    parser.add_argument("--allow_soft_placement", action='store_true',
                        help="Allow device soft device placement")
    parser.add_argument("--log_device_placement", action='store_true',
                        help="Log placement of ops on devices")
    
    return parser.parse_args()

def train_step(sess, u_batch, i_batch, uid, iid, reuid, reiid, y_batch, batch_num, deep, train_op, global_step, args):
    feed_dict = {
        deep.input_u: u_batch,
        deep.input_i: i_batch,
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.input_y: y_batch,
        deep.input_reuid: reuid,
        deep.input_reiid: reiid,
        deep.drop0: 0.8,
        deep.dropout_keep_prob: args.dropout_keep_prob
    }
    _, step, loss, accuracy, mae, u_a, i_a, fm = sess.run(
        [train_op, global_step, deep.loss, deep.accuracy, deep.mae, deep.u_a, deep.i_a, deep.score],
        feed_dict
    )
    time_str = datetime.datetime.now().isoformat()
    print(f"{time_str}: step {step}, loss {loss:.4f}, RMSE {accuracy:.4f}, MAE {mae:.4f}")
    return accuracy, mae, u_a, i_a, fm

def dev_step(sess, u_batch, i_batch, uid, iid, reuid, reiid, y_batch, deep, global_step):
    feed_dict = {
        deep.input_u: u_batch,
        deep.input_i: i_batch,
        deep.input_y: y_batch,
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.input_reuid: reuid,
        deep.input_reiid: reiid,
        deep.drop0: 1.0,
        deep.dropout_keep_prob: 1.0
    }
    step, loss, accuracy, mae = sess.run(
        [global_step, deep.loss, deep.accuracy, deep.mae],
        feed_dict
    )
    return loss, accuracy, mae

def main():
    args = parse_arguments()
    
    print("\nParameters:")
    for arg in vars(args):
        print(f"{arg.upper()}={getattr(args, arg)}")
    print("")
    
    print("Loading data...")
    with open(args.para_data, 'rb') as pkl_file:
        para = pickle.load(pkl_file)
    user_num = para['user_num']
    item_num = para['item_num']
    review_num_u = para['review_num_u']
    review_num_i = para['review_num_i']
    review_len_u = para['review_len_u']
    review_len_i = para['review_len_i']
    vocabulary_user = para['user_vocab']
    vocabulary_item = para['item_vocab']
    train_length = para['train_length']
    test_length = para['test_length']
    u_text = para['u_text']
    i_text = para['i_text']
    
    print(f"Users: {user_num}, Items: {item_num}, Train length: {train_length}, Test length: {test_length}")
    
    with open(args.train_data, 'rb') as f:
        train_data = np.asarray(pickle.load(f), dtype="object")
    with open(args.valid_data, 'rb') as f:
        valid_data = np.asarray(pickle.load(f), dtype="object")

    data_size_train = len(train_data)
    data_size_test = len(valid_data)
    num_batches_train = (data_size_train + args.batch_size - 1) // args.batch_size
    num_batches_valid = (data_size_test + args.batch_size - 1) // args.batch_size

    best_rmse = float('inf')
    best_mae = float('inf')
    
    with tf.Graph().as_default():
        session_conf = tf.compat.v1.ConfigProto(
            allow_soft_placement=args.allow_soft_placement,
            log_device_placement=args.log_device_placement
        )
        session_conf.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=session_conf)
        with sess.as_default():
            deep = NARRE.NARRE(
                review_num_u=review_num_u,
                review_num_i=review_num_i,
                review_len_u=review_len_u,
                review_len_i=review_len_i,
                user_num=user_num,
                item_num=item_num,
                num_classes=1,
                user_vocab_size=len(vocabulary_user),
                item_vocab_size=len(vocabulary_item),
                embedding_size=args.embedding_dim,
                embedding_id=32,
                filter_sizes=list(map(int, args.filter_sizes.split(","))),
                num_filters=args.num_filters,
                l2_reg_lambda=args.l2_reg_lambda,
                attention_size=32,
                n_latent=32
            )
            
            tf.compat.v1.set_random_seed(2017)
            global_step = tf.compat.v1.Variable(0, name="global_step", trainable=False)

            optimizer = tf.compat.v1.train.AdamOptimizer(0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)
            train_op = optimizer.minimize(deep.loss, global_step=global_step)

            sess.run(tf.compat.v1.global_variables_initializer())

            # Training loop
            for epoch in range(args.num_epochs):
                print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
                train_rmse = 0
                train_mae = 0

                shuffle_indices = np.random.permutation(np.arange(data_size_train))
                shuffled_data = train_data[shuffle_indices]

                for batch_num in range(num_batches_train):
                    start_index = batch_num * args.batch_size
                    end_index = min((batch_num + 1) * args.batch_size, data_size_train)
                    data_batch = shuffled_data[start_index:end_index]

                    uid, iid, reuid, reiid, y_batch = zip(*data_batch)
                    u_batch = np.array([u_text[uid[i][0]] for i in range(len(uid))])
                    i_batch = np.array([i_text[iid[i][0]] for i in range(len(iid))])

                    batch_rmse, batch_mae, _, _, _ = train_step(
                        sess, u_batch, i_batch, uid, iid, reuid, reiid, y_batch, batch_num,
                        deep, train_op, global_step, args
                    )
                    train_rmse += batch_rmse
                    train_mae += batch_mae

                print(f"Training RMSE: {train_rmse / num_batches_train:.4f}, MAE: {train_mae / num_batches_train:.4f}")

                # Validation
                val_rmse, val_mae = 0, 0
                for batch_num in range(num_batches_valid):
                    start_index = batch_num * args.batch_size
                    end_index = min((batch_num + 1) * args.batch_size, data_size_test)
                    data_batch = valid_data[start_index:end_index]

                    uid, iid, reuid, reiid, y_batch = zip(*data_batch)
                    u_batch = np.array([u_text[uid[i][0]] for i in range(len(uid))])
                    i_batch = np.array([i_text[iid[i][0]] for i in range(len(iid))])

                    _, batch_rmse, batch_mae = dev_step(sess, u_batch, i_batch, uid, iid, reuid, reiid, y_batch, deep, global_step)
                    val_rmse += batch_rmse ** 2 * len(u_batch)
                    val_mae += batch_mae * len(u_batch)

                val_rmse = np.sqrt(val_rmse / data_size_test)
                val_mae /= data_size_test
                print(f"Validation RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}")

                if best_rmse > val_rmse:
                    best_rmse = val_rmse
                if best_mae > val_mae:
                    best_mae = val_mae

            print(f"\nBest Validation RMSE: {best_rmse:.4f}, MAE: {best_mae:.4f}")

if __name__ == "__main__":
    main()

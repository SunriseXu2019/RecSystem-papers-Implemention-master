import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from sklearn.metrics import roc_auc_score
from dataset.Criteo import CriteoDatasetTensorflow
from model.DeepFM.DeepFM_Tensorflow import DeepFM
import argparse
import os
import numpy as np
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


@tf.function
def cross_entropy_loss(gt, pred):
    return tf.reduce_mean(tf.losses.binary_crossentropy(gt, pred))


@tf.function
def train_one_step(model, optimizer, dense_feature, sparse_feature, label):
    with tf.GradientTape() as tape:
        output = model([dense_feature, sparse_feature])
        loss = cross_entropy_loss(gt=label, pred=output)
    grad = tape.gradient(loss, model.trainable_variables)
    grad = [tf.clip_by_norm(g, 100) for g in grad]
    optimizer.apply_gradients(grads_and_vars=zip(grad, model.trainable_variables))
    return loss, output


def train(model, train_dataset, val_dataset, optimizer, epoch, writer):
    gt, pred = [], []
    train_log_loss = tf.keras.metrics.BinaryCrossentropy()
    val_log_loss = tf.keras.metrics.BinaryCrossentropy()

    # train
    for index, (dense_feature, sparse_feature, label) in enumerate(train_dataset):
        loss, output = train_one_step(model, optimizer, dense_feature, sparse_feature, label)
        pred.extend(list(output.numpy()))
        gt.extend(list(label.numpy()))
        train_log_loss.update_state(label, output)

        if index % 1000 == 1:
            print('Train Epoch: {} index:{} Loss:{:.6f}'.format(epoch, index * len(dense_feature), loss.numpy()))
    train_auc = roc_auc_score(np.array(gt), np.array(pred))
    print('train AUC: %.6f' % train_auc)
    print('train LogLoss: %.6f' % train_log_loss.result())
    tf.summary.scalar('train_log_loss', train_log_loss.result(), epoch)
    tf.summary.scalar('train_auc', train_auc, epoch)

    gt.clear()
    pred.clear()
    # val
    for index, (dense_feature, sparse_feature, label) in enumerate(val_dataset):
        output = model([dense_feature, sparse_feature])
        val_log_loss.update_state(label, output)
        pred.extend(list(output.numpy()))
        gt.extend(list(label.numpy()))
    val_auc = roc_auc_score(np.array(gt), np.array(pred))
    print('val AUC: %.6f' % val_auc)
    print('val LogLoss: %.6f' % val_log_loss.result())
    tf.summary.scalar('val_log_loss', val_log_loss.result(), epoch)
    tf.summary.scalar('val_auc', val_auc, epoch)
    writer.flush()


def main(params):
    root_path = '/home/xlm/project/recommendation/dataset/criteo/'
    parser = argparse.ArgumentParser(description='train DeepCTR FM with tensorflow')
    parser.add_argument('--root_path', type=str, default=root_path, help='root path')
    parser.add_argument('--model_save_dir', type=str, default='None', help='FM save dir')
    parser.add_argument('--epochs', type=int, default=200, help='epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer we are using')
    parser.add_argument('--loss', type=str, default='BCE', help='loss_func we are using')
    parser.add_argument('--embed_size', type=int, default=8, help='embedding size of sparse features')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids we are using')
    args = parser.parse_args(params)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    if not os.path.exists(args.model_save_dir):
        os.mkdir(args.model_save_dir)

    # load dense and sparse feature columns
    with open(os.path.join(root_path, 'dense.info'), 'rb') as f:
        dense_feature_columns = pickle.load(f)
    with open(os.path.join(root_path, 'sparse.info'), 'rb') as f:
        sparse_feature_columns = pickle.load(f)
    print('load dense and sparse')

    # load train & test dataset
    train_data = CriteoDatasetTensorflow(root_path, 'train.lmdb', batch_size=args.batch_size)
    val_data = CriteoDatasetTensorflow(root_path, 'val.lmdb', batch_size=args.batch_size)
    print('load_dataset')

    model = DeepFM(dense_feature_columns, sparse_feature_columns, embed_dim=args.embed_size)
    model.summary()

    lr_decay = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=args.lr,
                                                              decay_steps=5,
                                                              decay_rate=0.9)

    if args.optimizer == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    else:
        RuntimeError('No support optimizer')

    writer = tf.summary.create_file_writer("./my_logs")
    with writer.as_default():
        for epoch in range(args.epochs):
            train(model, train_data, val_data, optimizer, epoch, writer)
            model.save_weights(filepath=os.path.join(args.model_save_dir, str(epoch) + '.pth'))


if __name__ == '__main__':
    params = [
        '--epochs',      '200',
        '--batch_size',  '4096',
        '--embed_size',  '8',
        '--optimizer',   'Adam',
        '--loss',        'BCE',   # loss func is Binary CrossEntropy Loss
        '--gpu_ids',     '1, 2, 3, 4',
        '--lr',          '0.001',
        '--model_save_dir', './checkpoints/DeepFM/'
    ]
    main(params)


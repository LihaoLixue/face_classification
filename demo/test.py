import tensorflow as tf

def bn_transform(x):
    batch_mean, batch_var = tf.nn.moments(x,[ 0])
    z_hat = (x - batch_mean) / tf.sqrt(batch_var + epsilon)
    gamma = tf.Variable(tf.ones([ 100]))
    beta = tf.Variable(tf.zeros([ 100]))
    bn = gamma * z_hat + beta
    y = tf.nn.sigmoid(bn)
    return y
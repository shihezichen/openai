import  tensorflow as tf

#hello = tf.constant("Hello Tensorflow")
hello = tf.compat.v1.constant("Hello  Tensorflow")
sess = tf.compat.v1.Session()
print( sess.run( hello ) )
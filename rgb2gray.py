#coding:utf-8
from PIL import Image
from PIL import Image as image
import numpy as np
import tensorflow as tf
import os
import time


imageNum = 1000
BATCH_SIZE = 10
Str2Num = {}
Str2Num["+"] = 11
Str2Num["-"] = 12
Str2Num["*"] = 13
Str2Num["("] = 14
Str2Num[")"] = 15
for i in range(10):
    Str2Num[str(i)] = i+1

Num2Str = {v:k for k,v in Str2Num.items()}
IMAGE_HEIGHT = 60
IMAGE_WIDTH  = 180
MAX_CAPTCHA  = 7
CHAR_SET_LEN = 15

'''
临时测试
'''
#Image_array = []
labels = []

#label = ['(4*8)+8','7+3*0','5+(5+2)','(8-0)-8','0+(0+2)','(2*5)+0','0+(1+8)','2+(4+9)','7*2+4','0-(7*1)','9*(3*9)','2+8-3','(4-8)*5','6-4+4','5+(1*6)','4*(3+8)','6+(5*3)','0+(6+1)','6*(0*2)','(5*8)+4','0+(6+1)','6*(0*2)','(5*8)+4']

def readFile(path):
	with open(path,'rb') as f:
		for line in f:
			content = line.strip().split(' ')[0]
			labels.append(content)
	#return labels


def text2array(string):
	row = MAX_CAPTCHA
	col = CHAR_SET_LEN
	Array = np.zeros([row,col])
	for i in range(len(string)):
		value = Str2Num[string[i]]-1
		Array[i][value] = 1.0
    
	#print(Array)
	return Array

def RGB2Gray(image_origin):
	gray = image_origin.convert('L')
	return gray

def saveImage(image_final,i):
	image_final.save("{0}_gray.png".format(i))

#二值化  图片X
def binaryzation(image_gray):
	image_gray  = image_gray.resize((180,60),image.ANTIALIAS)
	image_array = np.array(image_gray)
	image_array = (image_array/255.0)*(image_array/255.0)
	#image_array = 1-np.rint(image_array) 
	image_array = 1-image_array
	image_array = 1+np.floor(image_array - image_array.mean())
	return image_array

def get_next_batch(startPoint=0,endPoint=100):
	print("[{0},{1}]".format(startPoint,endPoint))
	batch_x = np.zeros([abs(startPoint-endPoint), IMAGE_HEIGHT*IMAGE_WIDTH])
	batch_y = np.zeros([abs(startPoint-endPoint), MAX_CAPTCHA*CHAR_SET_LEN])
	for i in range(startPoint,endPoint):
		batch_x[i-startPoint,:] = Image_array[i].flatten()
		batch_y[i-startPoint,:] = text2array(labels[i]).flatten()
	
	return batch_x, batch_y
	




# 定义CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
	# 将占位符 转换为 按照图片给的新样式
	x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

	#w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH)) #
	#w_c2_alpha = np.sqrt(2.0/(3*3*32))
	#w_c3_alpha = np.sqrt(2.0/(3*3*64))
	#w_d1_alpha = np.sqrt(2.0/(8*32*64))
	#out_alpha = np.sqrt(2.0/1024)

	# 3 conv layer
	w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 32])) # 从正太分布输出随机值
	b_c1 = tf.Variable(b_alpha*tf.random_normal([32]))
	conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
	#print("conv1.0: {0} ".format(conv1.get_shape()))
	conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	#print("conv1.1: {0} ".format(conv1.get_shape()))
	conv1 = tf.nn.dropout(conv1, keep_prob)
	#print("conv1.2: {0} ".format(conv1.get_shape()))

	w_c2 = tf.Variable(w_alpha*tf.random_normal([3, 3, 32, 64]))
	b_c2 = tf.Variable(b_alpha*tf.random_normal([64]))
	conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
	#print("conv2.0: {0} ".format(conv2.get_shape()))
	conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	#print("conv2.1: {0} ".format(conv2.get_shape()))
	conv2 = tf.nn.dropout(conv2, keep_prob)
	#print("conv2.2: {0} ".format(conv2.get_shape()))

	w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 64]))
	b_c3 = tf.Variable(b_alpha*tf.random_normal([64]))
	conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
	#print("conv3.0: {0} ".format(conv3.get_shape()))
	conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	#print("conv3.1: {0} ".format(conv3.get_shape()))
	conv3 = tf.nn.dropout(conv3, keep_prob)
	#print("conv3.2: {0} ".format(conv3.get_shape()))

	# Fully connected layer
	w_d = tf.Variable(w_alpha*tf.random_normal([8*23*64, 1024]))
	b_d = tf.Variable(b_alpha*tf.random_normal([1024]))
	dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
	dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
	dense = tf.nn.dropout(dense, keep_prob)

	w_out = tf.Variable(w_alpha*tf.random_normal([1024, MAX_CAPTCHA*CHAR_SET_LEN]))
	b_out = tf.Variable(b_alpha*tf.random_normal([MAX_CAPTCHA*CHAR_SET_LEN]))
	out = tf.add(tf.matmul(dense, w_out), b_out)
	#out = tf.nn.softmax(out)
	return out

# 训练
def train_crack_captcha_cnn():
	output = crack_captcha_cnn()
	#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))
	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, targets=Y))
        # 最后一层用来分类的softmax和sigmoid有什么不同？
	# optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

	predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
	max_idx_p = tf.argmax(predict, 2)
	max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
	correct_pred = tf.equal(max_idx_p, max_idx_l)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	saver = tf.train.Saver()
	Num   = 0
	
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		step = 0
		while  Num < imageNum:
			t0 = time.time()
			batch_x, batch_y = get_next_batch(Num, Num+BATCH_SIZE)
			_, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
			os.system("clear")
			# 每100 step计算一次准确率
			if step % 10 == 0:
				batch_x_test, batch_y_test = get_next_batch(Num, Num+BATCH_SIZE)
				#print("{0},{1}".format(batch_x_test,batch_y_test))
				acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
				#print("step:{0}, loss: {1}, acc: {2}".format(step, loss_, acc))
				# 如果准确率大于50%,保存模型,完成训练
				if acc > 0.95:
					saver.save(sess, "crack_capcha.model", global_step=step)
					break
			t1 = time.time()
			Time_Letf = (imageNum-Num)/(10*(t1-t0))
			print("step:{0}, loss: {1}, acc: {2}".format(step, loss_, acc))
			print("Time Letf: {0}h {1}m".format(int(Time_Letf/3600),int((Time_Letf-3600*int(Time_Letf/3600))/60)))
			step += 1
			Num  += BATCH_SIZE 

		saver.save(sess, "crack_capcha.model", global_step=step)


if __name__ == "__main__":
	####################################################################
	# 申请占位符 按照图片
	X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH])    #此函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值
	Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN])
	keep_prob = tf.placeholder(tf.float32) # dropout

	Image_array = []
	readFile('../image_contest_level_1/labels.txt')

	for i in range(imageNum):
		os.system("clear")
		print("process:{0}".format(i))
		image_handle = Image.open("../image_contest_level_1/{0}.png".format(i))
		image_handle = RGB2Gray(image_handle)
		image_array = binaryzation(image_handle)
		Image_array.append(image_array)
		#image_array = 255.0*(1-image_array)
		#image_handle = Image.fromarray(np.uint8(image_array))
		#saveImage(image_handle,i)

	train_crack_captcha_cnn()

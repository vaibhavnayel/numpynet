import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split

print 'imports complete'

def logistic(x):
	#sigmoid activation
	#x= vector or matrix
	return 1.0/(1+np.exp(-x))

def dlogistic(x):
	#derivative of sigmoid
	#x= vector or matrix
	return logistic(x)*(1-logistic(x))

def relu(x):
	#rectified linear activation
	#x= vector or matrix
	return np.maximum(x,0,x)

def drelu(x):
	#derivative of relu
	#x=vector or matrix
	return (x>0)*1

def tanh(x):
	#tanh activation
	#x= vector or matrix
	return (2*logistic(2*x))-1

def dtanh(x):
	#derivative of tanh activation
	#x= vector or matrix
	return 1-(tanh(x)*tanh(x))

def softmax(y,axis=0):
	#softmax activation
	#y= vector or matrix
	#axis= 0 for vector y
	#axis= 1 for matrix y
	if axis==0:#vector handling
		s=np.exp(y-max(y))
		return s/np.sum(s)
	
	else:#matrix handling
		s=np.exp(y-(np.max(y,axis=-1)[:,np.newaxis]))
		return s/np.sum(s,axis=-1)[:,np.newaxis]	

def sq(y_pred,y_true):
	#squared error loss
	return np.sum((y_pred-y_true)*(y_pred-y_true),axis=-1)/2
def dsq(y_pred,y_true):
	#derivative of squared error 
	return (y_pred-y_true)

def ce(y_pred,y_true):
	#cross entropy loss
	return -np.sum(y_true*np.log(y_pred),axis=-1)
def dce(y_pred,y_true):
	#derivative of cross entropy 
	return -y_true/y_pred

def onehot(y):
	#y= n-vector , output is nxn matrix
	y_hot=np.zeros((len(y),int(max(y))+1))
	y_hot[np.arange(len(y)),y.astype(int)]=1
	return y_hot

def initialize(sizes,num_input,num_output):
	#initializes weights randomly in Uniform[-0.5,0.5]
	num_hidden=len(sizes)
	sizes=[num_input]+sizes+[num_output]
	W=np.array([(np.random.random([sizes[i+1],sizes[i]])-0.5)*1 for i in range(num_hidden+1)])
	b=np.array([(np.random.random(sizes[i+1])-0.5)*1 for i in range(num_hidden+1)])
	return [W,b]

def activate_batch(X,W,b,activation,loss):
	#returns activations and preactivations of network given
	#vector or matrix X

	num_hidden=W.shape[0]-1
	a=globals()[activation]
	preactivations=[X]
	activations=[X]
	for i in range(num_hidden+1):
		pre_i=(activations[-1].dot(W[i].T)+b[i])
		act_i=a(pre_i)

		if i==num_hidden:
			if loss=='ce':
				act_i=softmax(pre_i,axis=1)
			if loss=='sq':
				act_i=logistic(pre_i)
		
		activations.append(act_i)
		preactivations.append(pre_i)

	return [(preactivations),(activations)]

def backprop_batch(y,W,b,activation,loss,preactivations,activations):
	#returns gradients of F and b given matrix or vector of activations and preactivations
	batch_size=y.shape[0]
	num_hidden=len(W)-1
	d=globals()['d'+activation]
	if loss=='ce':
		deltas=[-y+activations[-1]]
	elif loss=='sq':
		deltas=[dsq(activations[-1],y)*dlogistic(preactivations[-1])]
	deltas[-1]=deltas[-1][:,:,np.newaxis]
	dWs,dbs=[],[]

	for i in range(num_hidden+1):
		dW=np.mean(np.matmul(deltas[-1],activations[-2-i][:,np.newaxis,:]),axis=0)
		dWs=[dW]+dWs
		db=np.mean(deltas[-1],axis=0).flatten()
		dbs=[db]+dbs
		delta=np.matmul(W[-1-i].T,deltas[-1])*d(preactivations[-2-i])[:,:,np.newaxis]
		deltas.append(delta)
	return [np.array(dWs),np.array(dbs)]

def predict(X,W,b,activation,loss):
	#batch predict function
	#X= matrix
	num_hidden=W.shape[0]-1
	a=globals()[activation]
	preactivations=[X]
	activations=[X]
	for i in range(num_hidden+1):
		pre_i=(activations[-1].dot(W[i].T)+b[i])
		act_i=a(pre_i)

		if i==num_hidden:
			if loss=='ce':
				act_i=softmax(pre_i,axis=1)
			if loss=='sq':
				act_i=logistic(pre_i)
		activations.append(act_i)
		preactivations.append(pre_i)

	return np.array(activations[-1])

def avg_loss(y_pred,y_true,loss):
	#Batch avg loss function
	L=globals()[loss]
	return np.mean(L(y_pred,y_true))


class Network:
	def __init__(self, input_size, output_size, hidden_layer_sizes,activation,loss='sq'):
		self.input_size = input_size
		self.output_size = output_size
		self.hidden_layer_sizes = hidden_layer_sizes
		self.activation = activation
		self.W,self.b=initialize(hidden_layer_sizes,input_size,output_size)
		self.loss=loss
		
		#for logging
		self.epoch_val_loss=[]
		self.val_loss=[]
		self.val_acc=[]
		self.train_loss=[]
		self.train_acc=[]
		self.x_axis=[]	

	def train(self,X,y,lr,batch_size,max_iterations=10000,opt='adam',
		loss='sq',anneal=True,momentum=0.0,reg=0,snapshot=100,image_corruption=0,visualise=False):
		
		#y,y_val= one hot matrix
		#X,X_val= matrix
		#opt= {gd, momentum, nag, adam}
		#activation= {logistic, tanh, relu}
		#loss= {sq, ce}
		#lr= learning rate
		#snapshot= interval for reporting metrics
		#anneal= {True,False}
		#reg= regularization parameter
		#momentum= momentum factor

		lr_0=lr
		X,X_val,y,y_val=train_test_split(X,y,stratify=y,test_size=0.05,random_state=42)
		num_input=X.shape[1]
		num_output=y.shape[1]
		num_hidden=len(self.hidden_layer_sizes)


		E=globals()[self.loss]

		#initializing counts
		epoch=1
		count=0
		step=1

		self.hidden_layer_sizes=[num_input]+self.hidden_layer_sizes+[num_output]

		#for momentum
		dW_prev,db_prev=0,0
		#for adam
		mW,vW,mb,vb=0,0,0,0
		b1,b2,eps=0.9,0.999,1e-8
		t=1.0

		if visualise==True:
			plt.ion()
		#if os.path.isfile(expt_dir+"log_train.txt"):os.remove(expt_dir+"log_train.txt")
		#if os.path.isfile(expt_dir+"log_val.txt"):os.remove(expt_dir+"log_val.txt")


		#binary image corruption
		if image_corruption!=0:
			X*=np.random.choice([0,1],size=X.shape,p=[image_corruption,1-image_corruption])
			print('random pixel dropout complete')

		start_time=time.time()

		while t<max_iterations:
			#start_time=time.time()
			reg_W=reg*(self.W)
			reg_b=reg*(self.b)
			if opt=='gd' or opt=='momentum':
				#initialize gradient arrays to zero
				preactivations,activations=activate_batch(X[count:min(count+batch_size,X.shape[0]-1)],self.W,self.b,self.activation,self.loss)
				dW,db=backprop_batch(y[count:min(count+batch_size,X.shape[0]-1)],self.W,self.b,self.activation,self.loss,preactivations,activations)
				#update weights
				if opt=='gd':
					self.W-=lr*dW + reg_W
					self.b-=lr*db+ reg_b

				elif opt=='momentum':
					self.W-=lr*dW + reg_W + momentum*dW_prev
					self.b-=lr*db + reg_b	+ momentum*db_prev	
					dW_prev,db_prev=lr*dW + momentum*dW_prev,lr*db + momentum*db_prev
			elif opt=='nag':
				W_lookahead=self.W-momentum*dW_prev
				b_lookahead=self.b-momentum*db_prev
				preactivations,activations=activate_batch(X[count:min(count+batch_size,X.shape[0]-1)],self.W,self.b,self.activation,self.loss)
				dW_lookahead,db_lookahead=backprop_batch(y[count:min(count+batch_size,X.shape[0]-1)],W_lookahead,b_lookahead,self.ctivation,self.loss,preactivations,activations)
				self.W-=(lr*dW_lookahead + momentum*dW_prev) + reg_W 
				self.b-=(lr*db_lookahead + momentum*db_prev) + reg_b	
				dW_prev,db_prev=lr*dW_lookahead + momentum*dW_prev,lr*db_lookahead + momentum*db_prev	

			elif opt=='adam':
				preactivations,activations=activate_batch(X[count:min(count+batch_size,X.shape[0]-1)],self.W,self.b,self.activation,self.loss)
				dW,db=backprop_batch(y[count:min(count+batch_size,X.shape[0]-1)],self.W,self.b,self.activation,self.loss,preactivations,activations)
				mW=b1*mW + (1-b1)*dW
				mb=b1*mb + (1-b1)*db
				vW=b1*vW + (1-b2)*(dW*dW)
				vb=b1*vb + (1-b2)*(db*db)
				mW_hat,mb_hat,vW_hat,vb_hat=mW/(1-b1**t),mb/(1-b1**t),vW/(1-b2**t),vb/(1-b2**t)
				self.W-=lr*mW_hat/(vW_hat+eps)**0.5 + reg_W
				self.b-=lr*mb_hat/(vb_hat+eps)**0.5 + reg_b

			step+=1
			count+=batch_size
			t+=1
			
			if count>=X.shape[0]-1: 
				epoch+=1
				count=0
				step=0

			lr=lr_0/(t**0.5)

			#display metrics
			if t%snapshot==0:

				pred_val=predict(X_val,self.W,self.b,self.activation,self.loss)
				self.val_loss.append(avg_loss(pred_val,y_val,self.loss))
				self.val_acc.append(np.sum(np.argmax(pred_val,axis=1)==np.argmax(y_val,axis=1))/float(len(X_val)))

				pred_train=predict(X,self.W,self.b,self.activation,self.loss)
				self.train_loss.append(avg_loss(pred_train,y,self.loss))
				self.train_acc.append(np.sum(np.argmax(pred_train,axis=1)==np.argmax(y,axis=1))/float(len(X)))

				self.x_axis.append(t*batch_size/X.shape[0])
				'''
				np.save('W.npy',W)
				np.save('b.npy',b)
				'''

				if visualise==True and step%(20*snapshot)==0:
					plt.close('all')
					images=self.W[0].reshape([400,28,28])

					#plt.imshow(W[0][0].reshape([28,28]),cmap='seismic')
					
					f, axes = plt.subplots(nrows=20, ncols=20, figsize=(10, 10))

					for i in range(20):
					    for j in range(20):

						    axes[i,j].imshow(images[20*i+j], cmap='seismic')
						    axes[i,j].axis('off')


					plt.pause(0.0000001)


				'''
				plt.plot(x_axis,val_loss)

				plt.plot(x_axis,train_loss)
				plt.legend(('validation loss','training loss'))
				plt.xlabel('epochs')
				plt.ylabel('average loss')
				plt.pause(0.000001)
				'''

				print ('epoch= %d | step= %d | learning rate= %f | avg training loss= %f | training accuracy= %f | avg validation loss= %f | validation accuracy= %f| optimizer= %s | time taken= %s seconds'%(epoch,t,lr,self.train_loss[-1],self.train_acc[-1],self.val_loss[-1],self.val_acc[-1],opt,(time.time() - start_time)))
				start_time=time.time()
				#annealing
				if anneal==True and epoch>1:
					if val_loss[-1]>val_loss[-2] and val_loss[-2]>val_loss[-3]:
						print ('halving learning rate due to increase in validation loss over the last 3 snapshots')
						#step=0
						lr_0*=0.5					
			'''
			np.save(save_dir+'weights.npy',np.hstack([W,b]))
			p=np.argmax(predict(X_test,W,b,activation,loss),axis=1)
			r=np.arange(p.shape[0])
			out=np.vstack([r,p]).T
			np.savetxt(save_dir+'sub.csv', out, fmt='%d', delimiter=',', header="id,label",comments='')
			'''
	def predict(self,X):
		#batch predict function
		#X= matrix
		num_hidden=self.W.shape[0]-1
		a=globals()[self.activation]
		preactivations=[X]
		activations=[X]
		for i in range(num_hidden+1):
			pre_i=(activations[-1].dot(self.W[i].T)+self.b[i])
			act_i=a(pre_i)

			if i==num_hidden:
				if self.loss=='ce':
					act_i=softmax(pre_i,axis=1)
				if self.loss=='sq':
					act_i=logistic(pre_i)
			activations.append(act_i)
			preactivations.append(pre_i)

		return np.argmax(np.array(activations[-1]),axis=1)

if __name__='__main__':
	train=pd.read_csv('data/train.csv')
	X=train.as_matrix()[:,1:]/255
	y=train.as_matrix()[:,0]
	y=onehot(y)


	test=pd.read_csv('data/test.csv')
	X_test=test.as_matrix()[:,1:]/255





	print 'data loaded'

	net=Network(500,2,[100,10],'relu')

	net.train(X,y,0.001,16)










from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,ZeroPadding2D,Flatten,Dense,Dropout
from load_data import load_data
from show_plot import *
#读取图片数据
(x_img_train,y_label_train),(x_img_test,y_label_test) = load_data()
#将fuatures标准化
x_img_train_normalize = x_img_train.astype("float32")/255.0
x_img_test_normalize = x_img_test.astype("float32")/255.0
#label以一位有效编码进行转换
y_label_tarin_onehot = np_utils.to_categorical(y_label_train)
y_label_test_onehot = np_utils.to_categorical(y_label_test)
#建立keras的序贯模型
model = Sequential()
#建立卷积池化层1
model.add(ZeroPadding2D((1,1),input_shape=(32,32,3)))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPool2D((2,2),strides=(2,2)))
#建立卷积池化层2
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128,(3,3),activation="relu"))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128,(3,3),activation="relu"))
model.add(MaxPool2D((2,2),strides=(2,2)))
#建立卷积池化层3
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256,(3,3),activation="relu"))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256,(3,3),activation="relu"))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256,(3,3),activation="relu"))
model.add(MaxPool2D((2,2),strides=(2,2)))
#建立卷积池化层4
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512,(3,3),activation="relu"))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512,(3,3),activation="relu"))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512,(3,3),activation="relu"))
model.add(MaxPool2D((2,2),strides=(2,2)))
#建立卷积池化层5
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512,(3,3),activation="relu"))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512,(3,3),activation="relu"))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512,(3,3),activation="relu"))
model.add(MaxPool2D((2,2),strides=(2,2)))
#建立平坦层
model.add(Flatten())
#建立全连接层
model.add(Dense(1024,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1024,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1024,activation="relu"))
model.add(Dropout(0.5))
#建立输出层
model.add(Dense(10,activation="softmax"))
#定义训练方式
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
#开始训练
train_history = model.fit(x_img_train_normalize,y_label_tarin_onehot,validation_split=0.1,
                          epochs=100,batch_size=128,verbose=2)
#画出准确率执行的结果
show_train_history(train_history,"acc","val_acc")
#画出误差的执行结果
show_train_history(train_history,"loss","val_loss")

#评估模型测试集的准确率
scores = model.evaluate(x_img_test_normalize,y_label_test_onehot,verbose=0)
print("评估模型的准确率为：",scores[1])

#进行预测
prediction = model.predict_classes(x_img_test_normalize)
#显示前10项预测结果
plot_images_labels_prediction(x_img_test,y_label_test,prediction,0,10)

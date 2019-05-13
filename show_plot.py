import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title("Train History")
    plt.xlabel("Epoch")
    plt.ylabel(train)
    plt.legend(["train","validation"],loc="upper left")
    plt.savefig(train+".png")
    plt.show()

def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    label_dict = {0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",
                  5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}
    fig = plt.gcf()
    fig.set_size_inches(20,20)
    if num>25 : num = 25
    for i in range(0,num):
        ax = plt.subplot(5,5,1+i)
        ax.imshow(images[idx],cmap="binary")
        title=str(i)+","+label_dict[labels[i][0]]
        if len(prediction)>0:
            title+="=>"+label_dict[prediction[i]]
        ax.set_title(title,fontsize=20)
        ax.set_xticks([])
        ax.set_yticks([])
        idx+=1
    plt.savefig("预测结果.png")
    plt.show()
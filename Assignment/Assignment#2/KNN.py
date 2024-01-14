import numpy as np

class KNeighborsClassifier(object):
    def __init__(self, neighbors): 
        self.neighbors = neighbors

    def fit(self, data_x, data_y): 
        self.x = data_x
        self.y = data_y


    def predict(self, x):

        len_x = x.shape[0]
        for i in range(len_x):
            count = {}
            x_chosen = x[i]           
            distance = np.sqrt(np.sum((self.x - x_chosen)**2, axis=1))  #L2
            # print(x_chosen.shape)
            distance_sorted = distance.argsort() #距离升序排列数组
            for k in range(self.neighbors): #neighbors个点
                value = self.y[distance_sorted[k]]  #得到相应的点的值
                count[value] = count.get(value,0) + 1  #计数并保持在字典中
        result = max(count,key=count.get)
        return result

if __name__ == '__main__':
    N1 = int(input())

    train_data = []
    train_label = []
    for _ in range(N1):
        line = input().split()
        train_label.append(int(line[0]))
        data = []
        for i in range(1,31):
            data.append(float(line[i]))
        train_data.append(data)
    train_data = np.array(train_data)
    train_label = np.array((train_label))

    classifiers = {}
    classifiers[3] = KNeighborsClassifier(3)
    classifiers[5] = KNeighborsClassifier(5)
    classifiers[7] = KNeighborsClassifier(7)
    classifiers[3].fit(train_data,train_label)
    classifiers[5].fit(train_data,train_label)
    classifiers[7].fit(train_data,train_label)


    N2 = int(input())

    for _ in range(N2):
        line = input().split()
        k = int(line[0])
        test_data = []
        for i in range(1,31):
            test_data.append(float(line[i]))
        test_data = np.array(test_data).reshape(1,30)
        print(classifiers[k].predict(test_data))


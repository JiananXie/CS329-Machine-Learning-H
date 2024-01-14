import numpy as np


class SoftmaxRegression():
    def __init__(self, n_features, n_classes, max_epoch, lr) -> None:
        self.w = np.zeros((n_features + 1, n_classes))
        self.max_epoch = max_epoch
        self.lr = lr

    def predict(self, X):
        pass

    def fit(self, X, y):
        # 添加偏置项
        X = np.insert(X, X.shape[1], 1, axis=1)
        # 初始化参数
        n_samples, n_features = X.shape
        n_classes = self.w.shape[1]

        # 迭代训练
        for epoch in range(self.max_epoch):
            scores = np.dot(X, self.w)
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # 计算损失函数梯度
            grad = -(1 / n_samples) * np.dot(X.T, (y - probs))

            # 更新参数
            self.w -= self.lr * grad

        return self.w


if __name__ == "__main__":
    # 从标准输入读取参数
    s = input().split()
    N, D, C, E = map(int, s[:4])
    L = float(s[4])

    # 读取训练样本
    X = []
    for _ in range(N):
        X.append(list(map(float, input().split())))
    X = np.array(X)

    # 读取训练目标
    Y = []
    for _ in range(N):
        Y.append(list(map(float, input().split())))
    Y = np.array(Y)

    model = SoftmaxRegression(D, C, E, L)
    weight = model.fit(X, Y).reshape(-1)
    for w in weight:
        print("%.3f" % w)

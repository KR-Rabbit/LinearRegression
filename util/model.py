import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=1e-4, epochs=50000):
        self.__learning_rate = learning_rate
        self.__epochs = epochs
        self.__weights = None
        self.__w_list = []
        self.__b_list = []
        self.__loss_list = []

    @property
    def w(self) -> float:
        return self.__weights[1].item()

    @property
    def b(self) -> float:
        return self.__weights[0].item()

    @property
    def ws(self):
        """
        训练过程中的w
        """
        return self.__w_list

    @property
    def bs(self):
        """
        训练过程中的b
        """
        return self.__b_list

    @property
    def loss(self):
        """
        训练过程中的loss
        """
        return self.__loss_list

    @staticmethod
    def cal_gradient(x, y, y_pred):
        # 偏导数
        # dJ/dt =  1/m * sum((y-y_pred)*x)
        gradient = np.dot(x.T, y_pred - y.reshape(-1)) / len(x)
        return gradient

    @staticmethod
    def cal_loss(y, y_pred):
        # J(w,b) = 1/2m * sum((y-y_pred)^2)
        return 1 / (2 * len(y)) * np.sum(np.square(y - y_pred))

    def record(self, w_, b_, l_):
        self.__w_list.append(w_)
        self.__b_list.append(b_)
        self.__loss_list.append(l_)

    def fit(self, x: np.ndarray, y: np.ndarray):
        x_ = np.hstack((np.ones((x.shape[0], 1)), x))
        self.__weights = np.random.normal(0, 1, x_.shape[1])
        for i in range(self.__epochs):
            # 计算y_pred
            y_pred = np.dot(x_, self.__weights)
            # 计算梯度
            gradient = self.cal_gradient(x_, y, y_pred)
            # 计算损失
            loss = self.cal_loss(y, y_pred)
            self.record(self.w, self.b, loss)
            # 更新权重
            self.__weights -= self.__learning_rate * gradient

    def predict(self, x: np.ndarray, w: np.ndarray = None) -> np.ndarray:
        if w is None:
            w = self.__weights
        x_ = np.hstack((np.ones((x.shape[0], 1)), x))
        return np.dot(x_, w)

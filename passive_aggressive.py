import numpy as np

class PassiveAggressive:
    def __init__(self, c=1, w=None, iter=1000, loss=1, tau=1):
        self.w = w
        self.c = c
        self.iter = iter
        self.loss = loss
        self.tau = tau

    def calc_eta(self, loss, vec_x):
        if self.tau == 1:
            l2_norm = vec_x.dot(vec_x)
            tau = min(self.c, loss/l2_norm)
        elif self.tau == 2:
            tau = loss / (np.power(np.linalg.norm(vec_x, ord=2), 2) + (1 / (2*self.c)))
        return tau

    def predict(self, vec_feature):
        score = np.sum(np.multiply(vec_feature, self.w), axis=1)
        return score

class PAIC(PassiveAggressive):
    def __init__(self, c=1, w=None, iter=1000, loss=1, tau=1):
        super().__init__(c,w, iter, loss, tau)

    def L_hinge(self, vec_x, y):
        if self.loss == 1:
            return max([0, 1-y*np.dot(self.w,vec_x)])
        elif self.loss == 2:
            return 1/(1+abs(y*np.dot(self.w,vec_x)))

    def update(self, vec_x, y):
        loss = self.L_hinge(vec_x, y)
        eta = self.calc_eta(loss, vec_x)
        self.w += eta*y*vec_x

    def fit_one(self, vec_feature, y):
        if self.w is None:
            weight_dim = len(vec_feature)
            self.w = np.zeros(weight_dim)
        self.update(vec_feature, y)

    def fit(self, vec_feature, y):
        l = len(y)
        for i in range(self.iter):
            self.fit_one(vec_feature[i%l][:], y[i%l])

class PAIC2(PAIC):
    def __init__(self, c=1, w=None, iter=1000, loss=1, tau=2):
        super().__init__(c,w, iter, loss, tau)

class PAIR(PassiveAggressive):
    def __init__(self, c=1, w=None, iter=1000, loss=1, tau=1):
        super().__init__(c, w, iter, loss, tau)

    def L_hinge(self, vec_x):
        if self.loss == 1:
            return max([0, 1-np.dot(self.w,vec_x)])
        elif self.loss == 2:
            return 1/(1+abs(np.dot(self.w,vec_x)))

    def update(self, pos_x, neg_x):
        vec_x = pos_x-neg_x
        loss = self.L_hinge(vec_x)
        eta = self.calc_eta(loss, vec_x)
        self.w += eta*vec_x
        #print("loss ", loss, " eta ", eta, " norma1 ", np.linalg.norm(self.w,ord=1), " norma2 ", np.linalg.norm(self.w,ord=2), " somma ", sum(self.w))

    def fit_one(self, pos_x, neg_x):
        if self.w is None:
            weight_dim = len(pos_x)
            self.w = np.zeros(weight_dim)
        self.update(pos_x, neg_x)

    def fit(self, pos_feature, neg_feature):
        npos = np.shape(pos_feature)[0]
        nneg = np.shape(neg_feature)[0]
        for i in range(self.iter):
            self.fit_one(pos_feature[i%npos][:], neg_feature[i%nneg][:])
        #self.w = self.w/sum(self.w)
        return self.w

class PAIR2(PAIR):
    def __init__(self, c=1, w=None, iter=1000, loss=1, tau=2):
        super().__init__(c, w, iter, loss, tau)

# -*- coding: utf-8 -*-
'''
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: model.py
@time: 2020/1/13 15:16
@desc: model.py
'''
try:
    import numpy as np

    from scipy.special import digamma, gammaln, hyp1f1
    from sklearn.cluster import KMeans
    from numpy.matlib import repmat
    izip = zip

    from utils import cluster_acc, predict, calculate_mix, caculate_pi, log_normalize, console_log, d_hyp1f1, s_kmeans
except ImportError as e:
    print(e)
    raise ImportError


class VIModel_DP:
    """
    Variational Inference Hierarchical Dirichlet process Mixture Models of datas Distributions
    """
    def __init__(self, args):

        self.K = args.K
        self.T = args.T
        self.newJ = self.K
        self.max_k = 150
        self.second_max_iter = args.second_max_iter
        self.max_hy1f1_iter = args.max_hy1f1_iter
        self.args = args
        self.J = 3
        self.N = 300
        self.D = 3
        self.prior = dict()

        self.tau = None
        self.gamma = None
        self.u = None
        self.v = None
        self.zeta = None
        self.xi = None
        self.k = None
        self.pi = None

        self.rho = None
        self.var_theta = None

        self.temp_top_stick = None
        self.temp_xi_ss = None
        self.temp_k_ss = None

        self.det = 1e-10

    def init_top_params(self, data):

        self.J = len(data)
        self.D = data[0].shape[1]

        total_data = np.vstack((i for i in data))
        self.prior = {
            'mu': np.sum(total_data, 0) / np.linalg.norm(np.sum(total_data, 0)),
            'zeta': self.args.zeta,
            'u': self.args.u,
            'v': self.args.v,
            'tau': self.args.tau,
            'gamma': self.args.gamma,
        }

        self.u = np.ones(self.K) * self.prior['u']
        self.v = np.ones(self.K) * self.prior['v']
        self.zeta = np.ones(self.K)
        self.xi = np.ones((self.K, self.D))

        self.xi = self.xi / np.linalg.norm(self.xi, axis=1)[:, np.newaxis]
        self.k = self.u / self.v

        self.temp_top_stick = np.zeros(self.K)
        self.temp_xi_ss = np.zeros((self.K, self.D, self.D))
        self.temp_k_ss = np.zeros(self.K)

        self.init_update(data)

    def set_temp_zero(self):

        self.temp_top_stick.fill(0.0)
        self.temp_xi_ss.fill(0.0)
        self.temp_k_ss.fill(0.0)

    def init_update(self, x):

        for i in range(self.J):
            N = x[i].shape[0]
            # self.var_theta = repmat(caculate_pi(s_kmeans(n_components=self.K, x=x[i])[0], N, self.K), self.T, 1)
            # self.rho = repmat(caculate_pi(s_kmeans(n_components=self.T, x=x[i])[0], N, self.T), N, 1)
            self.var_theta = repmat(caculate_pi(KMeans(n_clusters=self.K).fit(x[i]).labels_, N, self.K), self.T, 1)
            self.rho = repmat(caculate_pi(KMeans(n_clusters=self.T).fit(x[i]).labels_, N, self.T), N, 1)

            self.temp_top_stick += np.sum(self.var_theta, 0)
            self.temp_k_ss += np.sum(self.rho.dot(self.var_theta), 0)

            dot_x = np.einsum("ijk, ilk -> ijl", x[i][:, :, np.newaxis], x[i][:, :, np.newaxis])
            t_r = np.sum(self.rho[:, :, np.newaxis, np.newaxis] * dot_x[:, np.newaxis], 0)

            self.temp_xi_ss += np.sum(self.var_theta[:, :, np.newaxis, np.newaxis] * t_r[:, np.newaxis], 0)

        self.update_zeta_xi()
        self.update_u_v()

    def init_second_params(self, N, x):

        # self.rho = repmat(caculate_pi(s_kmeans(n_components=self.T, x=x)[0], N, self.T), N, 1)
        self.rho = np.ones((N, self.T)) / self.T

    def expect_log_sticks(self, rho, K, gamma):

        E_Nc_minus_n = np.sum(rho, 0, keepdims=True) - rho
        E_Nc_minus_n_cumsum_geq = np.fliplr(np.cumsum(np.fliplr(E_Nc_minus_n), axis=1))
        E_Nc_minus_n_cumsum = E_Nc_minus_n_cumsum_geq - E_Nc_minus_n

        first_tem = np.log(1 + E_Nc_minus_n) - np.log(1 + gamma + E_Nc_minus_n_cumsum_geq)
        first_tem[:, K - 1] = 0
        dummy = np.log(gamma + E_Nc_minus_n_cumsum) - np.log(1 + gamma + E_Nc_minus_n_cumsum_geq)
        second_term = np.cumsum(dummy, axis=1) - dummy
        return first_tem + second_term

    def caclulate_log_lik_x(self, x):

        D = self.D
        E_k = digamma(self.u) - np.log(self.v)
        kdk1 = d_hyp1f1(0.5, D / 2, self.zeta * self.k, iteration=self.max_hy1f1_iter)
        kdk2 = d_hyp1f1(1.5, (D + 2) / 2, self.zeta * self.k, iteration=self.max_hy1f1_iter) * kdk1
        kdk3 = d_hyp1f1(0.5, D / 2, self.k, iteration=self.max_hy1f1_iter)
        temp = (1 / D * kdk1 + self.zeta * self.k * (
                3 / ((D + 2) * D) * kdk2 - (1 / (D ** 2)) * kdk1 * kdk1)) * self.k * (
                       E_k + np.log(self.zeta) - np.log(self.prior['zeta'] * self.k))
        log_like_x = gammaln(D / 2) - (D / 2) * np.log(2 * np.pi) + (D / 2) * E_k - np.log(
            (self.k ** (D / 2)) * hyp1f1(0.5, D / 2, self.k)) - (D / 2 / self.k + kdk3 / D) * (
                      self.u / self.v - self.k) + self.k / D * kdk1 + temp * (
                      x.dot(self.xi.T) ** 2)
        return log_like_x

    def var_inf_2d(self, x, Elogsticks_1nd, ite):

        Elog_phi = self.caclulate_log_lik_x(x)
        second_max_iter = 2000 if self.second_max_iter == -1 else self.second_max_iter
        lambdas = 1
        self.init_second_params(x.shape[0], x)
        Elogsticks_2nd = self.expect_log_sticks(self.rho, self.T, self.prior['gamma'])
        for i in range(second_max_iter):
            # compute var_theta

            if (i + 1) % (second_max_iter // 10) == 0:
                lambdas -= 0.1
            temp_var_theta = self.var_theta
            self.var_theta = self.rho.T.dot(Elog_phi) + Elogsticks_1nd
            log_var_theta, log_n = log_normalize(self.var_theta)
            self.var_theta = (1 - lambdas) * temp_var_theta + lambdas * np.exp(
                log_var_theta)

            temp_rho = self.rho
            self.rho = self.var_theta.dot(Elog_phi.T).T + Elogsticks_2nd
            log_rho, log_n = log_normalize(self.rho)
            self.rho = (1 - lambdas) * temp_rho + lambdas * np.exp(log_rho)
            Elogsticks_2nd = self.expect_log_sticks(self.rho, self.T, self.prior['gamma'])

        self.temp_top_stick += np.sum(self.var_theta, 0)
        self.temp_k_ss += np.sum(self.rho.dot(self.var_theta), 0)
        dot_x = np.einsum("ijk, ilk -> ijl", x[:, :, np.newaxis], x[:, :, np.newaxis])
        t_r = np.sum(self.rho[:, :, np.newaxis, np.newaxis] * dot_x[:, np.newaxis], 0)
        self.temp_xi_ss += np.sum(self.var_theta[:, :, np.newaxis, np.newaxis] * t_r[:, np.newaxis], 0)

        return None

    def var_inf(self, x):

        for ite in range(self.args.max_iter):

            self.set_temp_zero()
            Elogsticks_1nd = self.expect_log_sticks(self.var_theta, self.K, self.prior['tau'])
            for i in range(self.J):
                self.var_inf_2d(x[i], Elogsticks_1nd, ite)

            self.optimal_ordering()
            # compute k
            self.k = self.u / self.v
            self.k[self.k > self.max_k] = self.max_k

            self.update_zeta_xi()
            self.update_u_v()

            print(ite)
            if ite == self.args.max_iter - 1:
                # compute k
                self.k = self.u / self.v
                self.k[self.k > self.max_k] = self.max_k
                if self.args.verbose:
                    print('mu: {}'.format(self.xi))
                    print('kappa: {}'.format(self.k))

    def optimal_ordering(self):
        s = [(a, b) for (a, b) in izip(self.temp_top_stick, range(self.K))]
        x = sorted(s, key=lambda y: y[0], reverse=True)
        idx = [y[1] for y in x]
        self.temp_top_stick[:] = self.temp_top_stick[idx]
        self.temp_k_ss[:] = self.temp_k_ss[idx]
        self.temp_xi_ss[:] = self.temp_xi_ss[idx]

    def update_u_v(self):
        # compute u, v
        D = self.D
        zeta = self.prior['zeta']
        self.u = self.prior['u'] + (D / 2) * (1 + self.temp_k_ss) + self.zeta * self.k * (d_hyp1f1(0.5, D / 2,
                                                                                                      self.zeta * self.k,
                                                                                                      iteration=self.max_hy1f1_iter) / D)
        self.v = self.prior['v'] + self.temp_k_ss * (
                D / (2 * self.k) + (1 / D) * d_hyp1f1(0.5, D / 2, self.k, iteration=self.max_hy1f1_iter)) + \
                 (D / (2 * self.k) + (zeta / D) * d_hyp1f1(0.5, D / 2, zeta * self.k, iteration=self.max_hy1f1_iter))

    def update_zeta_xi(self):

        # compute zeta, xi
        mu = self.prior['mu'][np.newaxis, :]  # 1 * d
        for k in range(self.K):
            A = self.prior['zeta'] * mu.T.dot(mu) + self.temp_xi_ss[k]
            value, vector = np.linalg.eig(A)
            index = np.argmax(value)
            self.zeta[k] = value[index]
            self.xi[k] = vector[:, index]

    def fit(self, data):

        self.init_top_params(data)
        self.var_inf(data)
        return self

    def predict(self, data):
        # predict
        data = np.vstack((i for i in data))
        pi = np.ones(self.K) / self.K
        if 'gene' in self.args:
            c = 4
        else:
            c = 1
            ini = self.xi[0]
            for i in range(1, self.K):
                count = 0
                for j in range(self.xi.shape[1]):
                    if abs(abs(self.xi[i][j]) - abs(ini[j])) < 0.05:
                        count += 1
                if count == self.xi.shape[1]:
                    break
                else:
                    c += 1
                    ini = self.xi[i]
        pred = predict(data, mu=self.xi[:c], k=self.k[:c], pi=pi[:c], n_cluster=c)
        return pred

    def fit_predict(self, data):
        self.fit(data)
        return self.predict(data)


class VIModel_PY:
    """
    Variational Inference Hierarchical Dirichlet process Mixture Models of datas Distributions
    """
    def __init__(self, args):

        self.K = args.K
        self.T = args.T
        self.newJ = self.K
        self.max_k = 150
        self.second_max_iter = args.second_max_iter
        self.max_hy1f1_iter = args.max_hy1f1_iter
        self.args = args
        self.J = 3
        self.N = 300
        self.D = 3
        self.prior = dict()

        self.tau = None
        self.gamma = None
        self.u = None
        self.v = None
        self.zeta = None
        self.xi = None
        self.k = None
        self.pi = None

        self.rho = None
        self.var_theta = None

        self.temp_top_stick = None
        self.temp_xi_ss = None
        self.temp_k_ss = None

        self.det = 1e-10

    def init_top_params(self, data):

        self.J = len(data)
        self.D = data[0].shape[1]

        total_data = np.vstack((i for i in data))
        self.prior = {
            'mu': np.sum(total_data, 0) / np.linalg.norm(np.sum(total_data, 0)),
            'zeta': self.args.zeta,
            'u': self.args.u,
            'v': self.args.v,
            'tau': self.args.tau,
            'gamma': self.args.gamma,
            'omega': self.args.omega,
            'eta': self.args.eta,
        }

        self.u = np.ones(self.K) * self.prior['u']
        self.v = np.ones(self.K) * self.prior['v']
        self.zeta = np.ones(self.K)
        self.xi = np.ones((self.K, self.D))

        self.xi = self.xi / np.linalg.norm(self.xi, axis=1)[:, np.newaxis]
        self.k = self.u / self.v

        self.temp_top_stick = np.zeros(self.K)
        self.temp_xi_ss = np.zeros((self.K, self.D, self.D))
        self.temp_k_ss = np.zeros(self.K)

        self.init_update(data)

    def set_temp_zero(self):

        self.temp_top_stick.fill(0.0)
        self.temp_xi_ss.fill(0.0)
        self.temp_k_ss.fill(0.0)

    def init_update(self, x):

        for i in range(self.J):
            N = x[i].shape[0]
            # self.var_theta = repmat(caculate_pi(s_kmeans(n_components=self.K, x=x[i])[0], N, self.K), self.T, 1)
            # self.rho = repmat(caculate_pi(s_kmeans(n_components=self.T, x=x[i])[0], N, self.T), N, 1)
            self.var_theta = repmat(caculate_pi(KMeans(n_clusters=self.K).fit(x[i]).labels_, N, self.K), self.T, 1)
            self.rho = repmat(caculate_pi(KMeans(n_clusters=self.T).fit(x[i]).labels_, N, self.T), N, 1)

            self.temp_top_stick += np.sum(self.var_theta, 0)
            self.temp_k_ss += np.sum(self.rho.dot(self.var_theta), 0)

            dot_x = np.einsum("ijk, ilk -> ijl", x[i][:, :, np.newaxis], x[i][:, :, np.newaxis])
            t_r = np.sum(self.rho[:, :, np.newaxis, np.newaxis] * dot_x[:, np.newaxis], 0)

            self.temp_xi_ss += np.sum(self.var_theta[:, :, np.newaxis, np.newaxis] * t_r[:, np.newaxis], 0)

        self.update_zeta_xi()
        self.update_u_v()

    def init_second_params(self, N, x):

        # self.rho = repmat(caculate_pi(s_kmeans(n_components=self.T, x=x)[0], N, self.T), N, 1)
        self.rho = np.ones((N, self.T)) / self.T

    def expect_log_sticks(self, rho, K, gamma, eta):

        E_Nc_minus_n = np.sum(rho, 0, keepdims=True) - rho
        E_Nc_minus_n_cumsum_geq = np.fliplr(np.cumsum(np.fliplr(E_Nc_minus_n), axis=1))
        E_Nc_minus_n_cumsum = E_Nc_minus_n_cumsum_geq - E_Nc_minus_n

        k = np.array([i+1 for i in range(K)]).reshape((1, -1))
        first_tem = np.log(1 - eta + E_Nc_minus_n) - np.log(1 - eta + gamma + k * eta + E_Nc_minus_n_cumsum_geq)
        first_tem[:, K - 1] = 0
        dummy = np.log(gamma + k * eta + E_Nc_minus_n_cumsum) - np.log(1 - eta + gamma + k * eta + E_Nc_minus_n_cumsum_geq)
        second_term = np.cumsum(dummy, axis=1) - dummy
        return first_tem + second_term

    def caclulate_log_lik_x(self, x):

        D = self.D
        E_k = digamma(self.u) - np.log(self.v)
        kdk1 = d_hyp1f1(0.5, D / 2, self.zeta * self.k, iteration=self.max_hy1f1_iter)
        kdk2 = d_hyp1f1(1.5, (D + 2) / 2, self.zeta * self.k, iteration=self.max_hy1f1_iter) * kdk1
        kdk3 = d_hyp1f1(0.5, D / 2, self.k, iteration=self.max_hy1f1_iter)
        temp = (1 / D * kdk1 + self.zeta * self.k * (
                3 / ((D + 2) * D) * kdk2 - (1 / (D ** 2)) * kdk1 * kdk1)) * self.k * (
                       E_k + np.log(self.zeta) - np.log(self.prior['zeta'] * self.k))
        log_like_x = gammaln(D / 2) - (D / 2) * np.log(2 * np.pi) + (D / 2) * E_k - np.log(
            (self.k ** (D / 2)) * hyp1f1(0.5, D / 2, self.k)) - (D / 2 / self.k + kdk3 / D) * (
                      self.u / self.v - self.k) + self.k / D * kdk1 + temp * (
                      x.dot(self.xi.T) ** 2)
        return log_like_x

    def var_inf_2d(self, x, Elogsticks_1nd, ite):

        Elog_phi = self.caclulate_log_lik_x(x)

        second_max_iter = 2000 if self.second_max_iter == -1 else self.second_max_iter
        lambdas = 1
        self.init_second_params(x.shape[0], x)
        Elogsticks_2nd = self.expect_log_sticks(self.rho, self.T, self.prior['gamma'], self.prior['eta'])
        for i in range(second_max_iter):
            # compute var_theta

            if (i + 1) % (second_max_iter // 10) == 0:
                lambdas -= 0.1
            temp_var_theta = self.var_theta
            self.var_theta = self.rho.T.dot(Elog_phi) + Elogsticks_1nd
            log_var_theta, log_n = log_normalize(self.var_theta)
            self.var_theta = (1 - lambdas) * temp_var_theta + lambdas * np.exp(
                log_var_theta)

            temp_rho = self.rho
            self.rho = self.var_theta.dot(Elog_phi.T).T + Elogsticks_2nd
            log_rho, log_n = log_normalize(self.rho)
            self.rho = (1 - lambdas) * temp_rho + lambdas * np.exp(log_rho)
            Elogsticks_2nd = self.expect_log_sticks(self.rho, self.T, self.prior['gamma'], self.prior['eta'])

        self.temp_top_stick += np.sum(self.var_theta, 0)
        self.temp_k_ss += np.sum(self.rho.dot(self.var_theta), 0)
        dot_x = np.einsum("ijk, ilk -> ijl", x[:, :, np.newaxis], x[:, :, np.newaxis])
        t_r = np.sum(self.rho[:, :, np.newaxis, np.newaxis] * dot_x[:, np.newaxis], 0)
        self.temp_xi_ss += np.sum(self.var_theta[:, :, np.newaxis, np.newaxis] * t_r[:, np.newaxis], 0)

        return None

    def var_inf(self, x):

        for ite in range(self.args.max_iter):

            self.set_temp_zero()
            Elogsticks_1nd = self.expect_log_sticks(self.var_theta, self.K, self.prior['tau'], self.prior['omega'])
            for i in range(self.J):
                self.var_inf_2d(x[i], Elogsticks_1nd, ite)

            self.optimal_ordering()
            # compute k
            self.k = self.u / self.v
            self.k[self.k > self.max_k] = self.max_k

            self.update_zeta_xi()
            self.update_u_v()

            # if self.args.verbose:

            print(ite)
            if ite == self.args.max_iter - 1:
                # compute k
                self.k = self.u / self.v
                self.k[self.k > self.max_k] = self.max_k
                if self.args.verbose:
                    print('mu: {}'.format(self.xi))
                    print('kappa: {}'.format(self.k))

    def optimal_ordering(self):
        s = [(a, b) for (a, b) in izip(self.temp_top_stick, range(self.K))]
        x = sorted(s, key=lambda y: y[0], reverse=True)
        idx = [y[1] for y in x]
        self.temp_top_stick[:] = self.temp_top_stick[idx]
        self.temp_k_ss[:] = self.temp_k_ss[idx]
        self.temp_xi_ss[:] = self.temp_xi_ss[idx]

    def update_u_v(self):
        # compute u, v
        D = self.D
        zeta = self.prior['zeta']
        self.u = self.prior['u'] + (D / 2) * (1 + self.temp_k_ss) + self.zeta * self.k * (d_hyp1f1(0.5, D / 2,
                                                                                                      self.zeta * self.k,
                                                                                                      iteration=self.max_hy1f1_iter) / D)
        self.v = self.prior['v'] + self.temp_k_ss * (
                D / (2 * self.k) + (1 / D) * d_hyp1f1(0.5, D / 2, self.k, iteration=self.max_hy1f1_iter)) + \
                 (D / (2 * self.k) + (zeta / D) * d_hyp1f1(0.5, D / 2, zeta * self.k, iteration=self.max_hy1f1_iter))

    def update_zeta_xi(self):

        # compute zeta, xi
        mu = self.prior['mu'][np.newaxis, :]  # 1 * d
        for k in range(self.K):
            A = self.prior['zeta'] * mu.T.dot(mu) + self.temp_xi_ss[k]
            value, vector = np.linalg.eig(A)
            index = np.argmax(value)
            self.zeta[k] = value[index]
            self.xi[k] = vector[:, index]

    def fit(self, data):

        self.init_top_params(data)
        self.var_inf(data)
        return self

    def predict(self, data):
        # predict
        data = np.vstack((i for i in data))
        pi = np.ones(self.K) / self.K
        if 'gene' in self.args:
            c = 4
        else:
            c = 1
            ini = self.xi[0]
            for i in range(1, self.K):
                count = 0
                for j in range(self.xi.shape[1]):
                    if abs(abs(self.xi[i][j]) - abs(ini[j])) < 0.05:
                        count += 1
                if count == self.xi.shape[1]:
                    break
                else:
                    c += 1
                    ini = self.xi[i]
        # print(self.xi[:c])
        pred = predict(data, mu=self.xi[:c], k=self.k[:c], pi=pi[:c], n_cluster=c)
        return pred

    def fit_predict(self, data):
        self.fit(data)
        return self.predict(data)


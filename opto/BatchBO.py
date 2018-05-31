# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# -----------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
import scipyplot as spp
from dotmap import DotMap
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm

import opto.regression as rregression
import opto.data as rdata
from opto.CMAES import CMAES
from opto.opto.classes.OptTask import OptTask
from opto.opto.classes.StopCriteria import StopCriteria
from opto.opto.acq_func import *
import opto.utils as rutils
from opto.opto.classes.IterativeOptimizer import IterativeOptimizer

import logging

logger = logging.getLogger(__name__)


class BatchBO(IterativeOptimizer):
    def __init__(self, task, stopCriteria, parameters=DotMap()):
        """
        Bayesian optimization
        :param task:
        :param parameters:
        """
        super(BO, self).__init__(task=task,
                                 stopCriteria=stopCriteria,
                                 parameters=parameters)
        self.name = 'Bayesian Optimization'
        self.order = 0
        # ---
        self.batch_size = 5
        self.acq_func = parameters.get('acq_funcs', default=EI(model=None,
                                                               logs=None))
        self.optimizer = DotMap()
        self.optimizer.optimizer = parameters.get('optimizer', default=CMAES)
        self.optimizer.maxEvals = 20000
        self.model = parameters.get('model', default=rregression.GP)
        self.past_evals = parameters.get('past_evals', default=None)
        self.n_initial_evals = parameters.get('n_initial_evals', default=10)
        self.log_best_mean = False  # optimize mean acq_func at each step

        self.store_model = True  # Should we store all models for logging?
        self._model = None  # Current model
        self._logs.data.m = None
        self._logs.data.v = None
        self._logs.data.model = None

    def _simulate_experiments(self, dataset_initial: rdata.dataset,
                              n: int = 5) -> tuple:
        """
        sample n times from S^k_pi, the set of k experiments resulting from
           running a sequential policy, pi, k iterations

        :param dataset_initial:
        :param n: number of samples of S^k_pi. A hyperparameter
        :return:
        """

        datasets = [dataset_initial.copy()] * n
        acq_funcs = [EI(model=None, logs=None)] * n
        parameters = np.array([])
        objectives = np.array([])
        models = np.array([])
        k = self.batch_size

        for sample in range(n):
            np.append(parameters, np.array([]))
            for iteration in range(k):
                p = DotMap()
                p.verbosity = 0

                # Instantiate the model with given parameters
                self._model = self.model(parameters=p)
                # Train the model with provided dataset
                self._model.train(train_set=datasets[sample])
                # Update acquisition function with the posterior
                acq_funcs[sample].update(model=self._model, logs=self._logs)

                # Optimize acquisition function
                logging.info('Optimizing the acquisition function')
                task = OptTask(f=self.acq_func.evaluate,
                               n_parameters=self.task.get_n_parameters(),
                               n_objectives=1,
                               order=0,
                               bounds=self.task.get_bounds(),
                               name='Acquisition Function',
                               task={'minimize'},
                               labels_param=None,
                               labels_obj=None,
                               vectorized=True)
                stop_criteria = StopCriteria(maxEvals=self.optimizer.maxEvals)

                p = DotMap()
                p.verbosity = 1

                # Calculate the optimizer
                optimizer = self.optimizer.optimizer(parameters=p,
                                                     task=task,
                                                     stopCriteria=stop_criteria)
                x = np.matrix(optimizer.optimize())
                fx = self._model.predict(dataset=x.T)
                dataset_new = rdata.dataset(data_input=x, data_output=fx)
                datasets[sample] = datasets[sample].merge(dataset_new)
                parameters[sample].append(x)
                objectives[sample].append(fx)
            models.append(self._model)

        return parameters, objectives, models

    def _weigh_data_points(self, X: np.array, model) -> np.array:
        assert X.shape[0] == model.shape[0], "Mismatching number of " \
                                             "Xs and models"

        n = X.shape[0]
        k = self.batch_size
        # TODO: check that this k matches Xs
        Ws = np.array([])

        A_master = np.full(X[0].shape, 1)
        np.fill_diagonal(A_master, -1)

        m, v = model._predict(X)

        def prob_a_geq_b(m_normalized):
            def helper(a):
                return 1 - norm.cdf(-a)
            helper_vect = np.vectorize(helper)
            return helper_vect(m_normalized)

        for iteration in range(k):
            A_mask = np.delete(A_master, iteration)
            # computing (A \Sigma_y A.T)^{-1\2}(A\mu_y)
            m_normalized = np.matmul(np.sqrt(np.reciprocal(np.matmul(
                np.matmul(A_mask, v), A_mask.T))), np.matmul(A_mask, m))
            np.append(Ws, np.prod(prob_a_geq_b(m_normalized)))

        return Ws

    def _match_experiments(self, X: np.array, W: np.array,
                           k: int) -> np.array:
        """
        given n experiments weighted by the probability they are the
            maximizer, return k of them that are most representative
        approximates to k_medoids
        """
        B = X

        def obj(X, W, B):
            sum = 0
            for i in range(X.shape[0]):
                nbrs = NearestNeighbors(n_neighbors=1).fit(B)
                dists, _ = nbrs.kneighbors(X)
                sum += W[i] * dists[i]
            return sum

        # greedily remove elements until we have k of them left
        for _ in range(X.shape[0] - k):
            # compute the objective value with one element of batch removed
            for i in range(X.shape[0]):
                objs = obj(X, W, np.delete(B, i))
            np.delete(X, np.argmin(objs))

        return X

    def _select_parameters(self):
        """
        Select the next set of parameters to evaluate on the objective function
        :return: parameters: np.matrix
        """

        # If we don't have any data to start with, randomly pick points
        k = self.batch_size

        if (self._iter == 0) and (self.past_evals is None):
            logging.info('Initializing with %d random evaluations'
                         % self.n_initial_evals)
            self._logs.data.model = [None]
            return self.task.get_bounds() \
                .sample_uniform((self.n_initial_evals,
                                 self.task.get_n_parameters()))
        else:
            # TODO: use past_evals
            logging.info('Fitting response surface')

            dataset = rdata.dataset(data_input=self._logs.get_parameters(),
                                    data_output=self._logs.get_objectives())

            Xs, FXs, GPs = self._simulate_experiments(dataset)
            Xs = Xs.flatten(-1, Xs.shape[1:])
            Ws = np.array([])
            for i in range(k):
                np.append(Ws, self._weigh_data_points(Xs[i], GPs[i]))
            # now Xs and Ws should both be flattened w.r.t samples axis
            Xs = self._match_experiments(Xs, Ws, k)

            # TODO: integrate the different acquisition functions to form one GP
            # # Log the mean and variance
            # if self._logs.data.m is None:
            #     self._logs.data.m = np.matrix(fx[0])
            #     self._logs.data.v = np.matrix(fx[1])
            # else:
            #     self._logs.data.m = np.concatenate((self._logs.data.m, fx[0]),
            #                                        axis=0)
            #     self._logs.data.v = np.concatenate((self._logs.data.v, fx[1]),
            #                                        axis=0)
            #
            # # Store the model
            # if self.store_model:
            #     if self._logs.data.model is None:
            #         self._logs.data.model = [self._model]
            #     else:
            #         self._logs.data.model.append(self._model)
            #
            # # Optimize mean function (for logging purposes)
            # if self.log_best_mean:
            #     logging.info('Optimizing the mean function')
            #     task = OptTask(f=self._model.predict_mean,
            #                    n_parameters=self.task.get_n_parameters(),
            #                    n_objectives=1,
            #                    order=0,
            #                    bounds=self.task.get_bounds(),
            #                    name='Mean Function',
            #                    task={'minimize'},
            #                    labels_param=None, labels_obj=None,
            #                    vectorized=True)
            #     stop_criteria = StopCriteria(maxEvals=self.optimizer.maxEvals)
            #     p = DotMap()
            #     p.verbosity = 1
            #     mean_opt = self.optimizer.optimizer(parameters=p,
            #                                         task=task,
            #                                         stopCriteria=stop_criteria)
            #
            #     best_x = np.matrix(optimizer.optimize())
            #     best_fx = self._model.predict(dataset=best_x.T)
            #
            #     if self._iter == 1:
            #         self._logs.data.best_m = np.matrix(best_fx[0])
            #         self._logs.data.best_v = np.matrix(best_fx[1])
            #     else:
            #         self._logs.data.best_m = np.concatenate(
            #             (self._logs.data.best_m, best_fx[0]), axis=0)
            #         self._logs.data.best_v = np.concatenate(
            #             (self._logs.data.best_v, best_fx[1]), axis=0)

            return Xs

    def f_visualize(self):
        # TODO: plot also model (see plot_optimization_curve)
        if self._iter == 0:
            self._fig = plt.figure()
            self._objectives_curve, _ = plt.plot(
                self.get_logs().get_objectives().T,
                linewidth=2,
                color='blue')
            plt.ylabel('Obj.Func.')
            plt.xlabel('N. Evaluations')
        else:
            self._objectives_curve.set_data(np.arange(self.get_logs()
                                                      .get_n_evals()),
                                            self.get_logs()
                                            .get_objectives().T)
            self._fig.canvas.draw()
            plt.xlim([0, self.get_logs().get_n_evals()])
            plt.ylim([np.min(self.get_logs().get_objectives()),
                      np.max(self.get_logs().get_objectives())])

    def plot_optimization_curve(self, scale='log', plotDelta=True):
        import scipyplot as spp

        logs = self.get_logs()
        plt.figure()
        # logs.plot_optimization_curve()

        if (self.task.opt_obj is None) and (plotDelta is True):
            plt.plot(logs.get_objectives().T, c='red', linewidth=2)
            plt.ylabel('Obj.Func.')
            n_evals = logs.data.m.shape[0]
            x = np.arange(start=logs.get_n_evals() - n_evals,
                          stop=logs.get_n_evals())
            spp.gauss_1D(y=logs.data.m,
                         variance=logs.data.v,
                         x=x,
                         color='blue')
            if self.log_best_mean:
                spp.gauss_1D(y=logs.data.best_m,
                             variance=logs.data.best_v,
                             x=x,
                             color='green')
        else:
            plt.plot(logs.get_objectives().T - self.task.opt_obj,
                     c='red',
                     linewidth=2)
            plt.ylabel('Optimality gap')
            n_evals = logs.data.m.shape[0]
            x = np.arange(start=logs.get_n_evals() - n_evals,
                          stop=logs.get_n_evals())
            spp.gauss_1D(y=logs.data.m - self.task.opt_obj,
                         variance=logs.data.v,
                         x=x,
                         color='blue')
            if self.log_best_mean:
                spp.gauss_1D(y=logs.data.best_m - self.task.opt_obj,
                             variance=logs.data.best_v,
                             x=x,
                             color='green')

        plt.xlabel('N. Evaluations')
        if scale == 'log':
            ax = plt.gca()
            ax.set_yscale('log')

        # TODO: best performance expected
        # if self.log_best_mean:
        #     plt.legend(['Performance evaluated', 'performance expected', 'Best performance expected'])
        # else:
        #     plt.legend(['Performance evaluated', 'performance expected'])
        plt.show()

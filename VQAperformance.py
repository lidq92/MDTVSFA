from ignite.metrics.metric import Metric
import numpy as np
from scipy import stats


class VQAPerformance(Metric):
    """
    Evaluation of VQA methods using SROCC, KROCC, PLCC, RMSE.

    `update` must receive output of the form (y_pred, y).
    """
    def reset(self):
        self._rq = []
        self._mq = []
        self._aq = []
        self._y  = []

    def update(self, output):
        y_pred, y = output
        self._y.append(y[0].item())
        self._rq.append(y_pred[0][0].item())
        self._mq.append(y_pred[1][0].item())
        self._aq.append(y_pred[2][0].item())

    def compute(self):
        sq = np.reshape(np.asarray(self._y), (-1,))
        rq = np.reshape(np.asarray(self._rq), (-1,))
        mq = np.reshape(np.asarray(self._mq), (-1,))
        aq = np.reshape(np.asarray(self._aq), (-1,))

        SROCC = stats.spearmanr(sq, rq)[0]
        KROCC = stats.stats.kendalltau(sq, rq)[0]
        PLCC = stats.pearsonr(sq, mq)[0]
        RMSE = np.sqrt(np.power(sq-aq, 2).mean())
        return {'SROCC': SROCC,
                'KROCC': KROCC,
                'PLCC': PLCC,
                'RMSE': RMSE,
                'sq': sq,
                'rq': rq,
                'mq': mq,
                'aq': aq}

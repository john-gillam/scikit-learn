import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer

def make_bypass_scorer(score_func, filt_func=None, greater_is_better=True, needs_proba=False,
                       needs_threshold=False, **kwargs):
    if filt_func is None:
        #needs to be masked array: create a filt_func based on that
        def filter_func(y, y_pred):
            mask = ~y_pred.mask
            if mask.ndim != 1: mask = mask.all(axis=1)
            return y[mask], y_pred.filled()[mask]
    else:
        #use the filter function directly on the predictions
        def filter_func(y, y_pred):
            mask = filt_func(y_pred)
            if mask.ndim != 1: mask = mask.all(axis=1)
            return y[mask], y_pred[mask]

    def bypass_score_func(y, y_pred, **sfkwargs):
        assert isinstance(y_pred, np.ma.masked_array) != (filt_func is not None)
        #has to either be masked or a filt-func defined
        #assume filt_func takes care of any masking so xor them

        y_sub, y_pred_sub = filter_func(y, y_pred)

        #only score the items that were actually classified
        return score_func(y_sub, y_pred_sub, **sfkwargs)

    return make_scorer(bypass_score_func, greater_is_better=True,
                       needs_proba=False, needs_threshold=False, **kwargs)

class BypassPipeline(Pipeline):
    #The BypassPipeline should behave exactly like a regular pipeline except
    #that it can refuse-to-process marked or specific rows of the data
    #if this pipeline is used in a CV, then the scorer really must be a
    #bypass scorer too!

    def __init__(self, steps, memory=None, bypass_func=None,
                   filled_output=None):
        self.steps = steps
        self._validate_steps()
        self.memory = memory

        #if filled_output is set here, always output filled predictions ect
        #otherwise you need to set the next injected-output to be filled
        if filled_output is None:
            self._always_fill = False
        else:
            self._always_fill = True
        self.filled_output = filled_output

        if bypass_func is None:
            self.bypass_func = lambda x : np.ones(x.shape[0], dtype=np.bool)
        else:
            self.bypass_func = bypass_func

    def _open_bypass(self, X):
        """
        Open the bypass based on some data and the filter func defined on
        this pipeline
        """
        mask = self.bypass_func(X)
        # print("--", mask)
        assert (mask.shape[0] == X.shape[0]) and (mask.ndim == 1)
        return mask

    def set_filled_output(self, val):
        self.filled_output = val

    def unset_filled_output(self):
        self.filled_output = None

    def _fill_this(self, maskedreturner):

        new_maskedreturner = maskedreturner.astype(self.filled_output)
        if new_maskedreturner.dtype.char in np.typecodes['AllFloat']:
            new_maskedreturner.set_fill_value(np.nan)

        if not self._always_fill:
            self.unset_filled_output()
        new_returner = new_maskedreturner.filled()

        return new_returner

    def bypass_inject(self, returns, current_mask):
        """
        Based on the mask injects data into a masked-array set of returns
        """
        assert current_mask is not None
        return_shape = list(returns.shape)
        mask_shape = current_mask.shape[0]
        assert mask_shape >= return_shape[0]

        return_shape[0] = mask_shape
        bypass_return = np.empty(return_shape, dtype=returns.dtype)
        bypass_return[current_mask] = returns


        this_mask = ~current_mask # this is 1d afaics --> for proba need 2d
        if returns.ndim == 2:
            this_mask = np.repeat(this_mask[:, None], 2, axis=1)

        new_returns = np.ma.masked_array(bypass_return,
                                         mask=this_mask)

        if new_returns.dtype.char in np.typecodes['AllFloat']:
            new_returns.set_fill_value(np.nan)

        if self.filled_output is None:
            return new_returns
        else:
            return self._fill_this(new_returns)

    def bypass_filter(self, returns, current_mask):
        """
        Filters out values that do not conform to the bypass mask
        """
        assert current_mask is not None
        assert current_mask.shape[0] == returns.shape[0]

        if returns is not None:
            new_returns = returns[current_mask].copy()
        else:
            new_returns = None

        if isinstance(new_returns, np.ma.masked_array):
            new_returns = new_returns.filled() #don't pass a
        return new_returns

    def fit(self, X, y=None, **fit_params):
        mask = self._open_bypass(X)
        Xt = self.bypass_filter(X, mask)
        yt = self.bypass_filter(y, mask)
        return super(BypassPipeline, self).fit(Xt, y=yt, **fit_params)

    def transform(self, X):
        mask = self._open_bypass(X)
        Xt = self.bypass_filter(X, mask)
        returns = super(BypassPipeline, self).transform(Xt)
        new_returns = bypass_inject(returns, mask)
        return new_returns

    def inverse_transform(self, X):
        mask = self._open_bypass(X)
        Xt = self.bypass_filter(X, mask)
        returns = super(BypassPipeline, self).inverse_transform(Xt)
        new_returns = bypass_inject(returns, mask)
        return new_returns

    def fit_transform(self, X, y=None, **fit_params):#taken from
        mask = self._open_bypass(X)
        Xt = self.bypass_filter(X, mask)
        yt = self.bypass_filter(y, mask)
        returns = super(BypassPipeline, self).fit_transform(Xt, y=yt,**fit_params)
        new_returns = bypass_inject(returns, mask)
        return new_returns

    def fit_predict(self, X, y):
        mask = self._open_bypass(X)
        Xt = self.bypass_filter(X, mask)
        yt = self.bypass_filter(y, mask)
        yt_pred = super(BypassPipeline, self).fit_predict(Xt, yt)
        y_pred = self.bypass_inject(yt_pred, mask)
        return y_pred

    def predict(self, X):
        mask = mask = self._open_bypass(X)
        Xt = self.bypass_filter(X, mask)
        yt_pred = super(BypassPipeline, self).predict(Xt)
        # print("S~~~~>", X.shape)
        # print("S~~~~>", Xt.shape)
        # print("S~~~~>", yt_pred.shape)
        # print("S~~~~>", mask.shape)
        y_pred = self.bypass_inject(yt_pred, mask)
        return y_pred

    def predict_proba(self, X):
        mask = self._open_bypass(X)
        Xt = self.bypass_filter(X, mask)
        yt_pred = super(BypassPipeline, self).predict_proba(Xt)
        # print("P~~~~>", X.shape)
        # print("P~~~~>", Xt.shape)
        # print("P~~~~>", yt_pred.shape)
        # print("P~~~~>", mask.shape)
        y_pred = self.bypass_inject(yt_pred, mask)
        return y_pred

    def predict_log_proba(self, X):
        mask = self._open_bypass(X)
        Xt = self.bypass_filter(X, mask)
        yt_pred = super(BypassPipeline, self).predict_log_proba(Xt)
        y_pred = self.bypass_inject(yt_pred, mask)
        return y_pred

    def decision_function(self, X):
        mask = self._open_bypass(X)
        Xt = self.bypass_filter(X, mask)
        dfunct = super(BypassPipeline, self).decision_function(Xt)
        dfunc = self.bypass_inject(dfunct, mask)
        return dfunc

    def score(self, X, y=None, sample_weight=None):
        mask = self._open_bypass(X)
        Xt = self.bypass_filter(X, mask)
        yt = self.bypass_filter(y, mask)
        score = super(BypassPipeline, self).score(Xt, y = yt,
                                                   sample_weight=sample_weight)
        return score

def test1():
    from sklearn import svm, preprocessing
    X = np.asarray([[0, 0], [1, 1], [0, 0], [1, 1]])
    Xnan = np.asarray([[0, 0], [1, 1], [np.nan, np.nan], [np.nan, np.nan]])
    y = np.asarray([0, 1, 0, 1])

    nofunc = preprocessing.FunctionTransformer(func = lambda x:x)
    clf1 = Pipeline([('trans', nofunc), ('learn', svm.SVC())])
    clf1.fit(X, y)
    print(clf1.predict(X) , [0, 1, 0, 1])

    clf2 = BypassPipeline([('trans', nofunc), ('learn', svm.SVC())])
    clf2.fit(X, y)
    print(clf2.predict(X) , [0, 1, 0, 1])

    func =lambda X: ~(~np.isfinite(X)).all(axis=1)
    clf3 = BypassPipeline([('trans', nofunc), ('learn', svm.SVC())],
                          bypass_func = func)
    clf3.fit(X, y)

    funcma = lambda X: ~(irisd3.mask).all(axis=1)

    print(clf3.predict(X) , [0, 1, 0, 1])
    print(clf3.predict(Xnan) , [0, 1, np.nan, np.nan])
    print(clf3.score(X, y))
    print(clf3.score(Xnan, y))
    clf3.fit(Xnan, y)
    print(clf3.predict(X) , [0, 1, 0, 1])
    print(clf3.predict(Xnan) , [0, 1, np.nan, np.nan])
    print(clf3.score(X, y))
    print(clf3.score(Xnan, y))

if __name__ == "__main__":#def test2():
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    from sklearn import svm
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import cross_val_score
    from sklearn import metrics

    iris = datasets.load_iris()

    irisd = iris.data
    irismask = np.random.binomial(1, 0.5, size=iris.data.shape[0]).astype(np.bool)

    irisd0 = irisd.copy()#iris data
    irisd1 = np.ma.masked_array(irisd0.copy(), mask=~np.isfinite(irisd0))

    irisd2 = irisd.copy()
    irisd2[irismask, :] = np.nan#iris data with nan-vectors

    #nan-vectors masked
    irisd3 = np.ma.masked_array(irisd2.copy(), mask=~np.isfinite(irisd2))
    #regular masked
    irisd4 = np.ma.masked_array(irisd1.copy(), mask=~np.isfinite(irisd2))

    func1 =lambda X: ~(~np.isfinite(X)).all(axis=1) #infinite all rejection
    func2 =lambda X: ~(X.mask).all(axis=1) #masked-all rejection

    clf0 = Pipeline([('learn', svm.SVC(kernel='linear', C=1))])

    clf1 = BypassPipeline([('learn', svm.SVC(kernel='linear', C=1))])

    clf2 = BypassPipeline([('learn', svm.SVC(kernel='linear', C=1))],
                          bypass_func=func1)

    clf3 = BypassPipeline([('learn', svm.SVC(kernel='linear', C=1))],
                          bypass_func=func1)

    clf4 = BypassPipeline([('learn', svm.SVC(kernel='linear', C=1))],
                          bypass_func=func2)

    ssA = make_bypass_scorer(score_func=lambda y1, y2: (y1 == y2).sum())
    ssB = make_scorer(score_func=lambda y1, y2: (y1 == y2).sum())

    for nc, clf in zip(['c0', 'c1', 'c2', 'c3', 'c4'], [clf0, clf1, clf2, clf3, clf4]):
        print("- - - ", nc, " - - -")
        for nd, data in zip(['d0', 'd1', 'd2', 'd3', 'd4'],[irisd0, irisd1, irisd2, irisd3, irisd4]):
            try:
                clf.fit(data, iris.target)
                ss = cross_val_score(clf, data, iris.target, cv=5, scoring=make_bypass_scorer(metrics.accuracy_score))
                print(nd, ": ", ss)
            except ValueError:
                print(nd, ": fail V")
            except AttributeError:
                print(nd, ": fail A1")
            except AssertionError:
                print(nd, ": fail A2")
    # return scores1, scores2a, scores2b, scores3a, scores3b
# if __name__ == "__main__":
#     this = test2()

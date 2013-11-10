import unittest

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    mpl = None

import nose
import numpy as np

import pandas as pd
import pandas.tools.plotting as plotting
import pandas.util.testing as tm


def axis_has_grid(axis, which='any'):
    minor = [tick.gridOn for tick in axis.minorTicks if tick is not None]
    major = [tick.gridOn for tick in axis.majorTicks if tick is not None]
    _which = which.lower()
    if _which == 'any':
        return np.any(major) or np.any(minor)
    if _which == 'both':
        return np.any(major) and np.any(minor)
    if _which == 'major':
        return np.any(major)
    if _which == 'minor':
        return np.any(minor)
    raise ValueError('Invalid argument which={}'.format(repr(which)))


def axes_has_grid(ax, axis='any', which='any'):
    _axis = axis.lower()
    if _axis == 'any':
        return (axis_has_grid(ax.xaxis, which=which) or
                axis_has_grid(ax.yaxis, which=which))
    if _axis == 'both':
        return (axis_has_grid(ax.xaxis, which=which) and
                axis_has_grid(ax.yaxis, which=which))
    if _axis == 'x':
        return axis_has_grid(ax.xaxis, which=which)
    if _axis == 'y':
        return axis_has_grid(ax.yaxis, which=which)
    raise ValueError('Invalid argument axis={}'.format(repr(axis)))


class _rc_context(object):
    def __init__(self, rc=None, fname=None):
        self.rcdict = rc
        self.fname = fname
        self._rcparams = mpl.rcParams.copy()
        if self.fname:
            rc_file(self.fname)
        if self.rcdict:
            mpl.rcParams.update(self.rcdict)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        mpl.rcParams.update(self._rcparams)


if mpl is not None and hasattr(mpl, 'rc_context'):
    rc_context = mpl.rc_context
else:
    rc_context = _rc_context


@tm.mplskip
class TestPlotGrids(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_plot_grids_series(self):
        s = tm.makeFloatSeries()

        with rc_context(rc={'axes.grid': False}):
            ax = s.plot(grid=None)
        self.assertFalse(axes_has_grid(ax))
        plt.close()

        with rc_context(rc={'axes.grid': False}):
            ax = s.plot(grid=True)
        self.assertTrue(axes_has_grid(ax, axis='both'))
        plt.close()

        with rc_context(rc={'axes.grid': False}):
            ax = s.plot(grid=False)
        self.assertFalse(axes_has_grid(ax))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            ax = s.plot(grid=None)
        self.assertTrue(axes_has_grid(ax, axis='both'))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            ax = s.plot(grid=True)
        self.assertTrue(axes_has_grid(ax, axis='both'))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            ax = s.plot(grid=False)
        self.assertFalse(axes_has_grid(ax))
        plt.close()

    def test_plot_grids_dataframe(self):
        df =  tm.makeDataFrame()

        with rc_context(rc={'axes.grid': False}):
            ax = df.plot(grid=None)
        self.assertFalse(axes_has_grid(ax))
        plt.close()

        with rc_context(rc={'axes.grid': False}):
            ax = df.plot(grid=True)
        self.assertTrue(axes_has_grid(ax, axis='both'))
        plt.close()

        with rc_context(rc={'axes.grid': False}):
            ax = df.plot(grid=False)
        self.assertFalse(axes_has_grid(ax))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            ax = df.plot(grid=None)
        self.assertTrue(axes_has_grid(ax, axis='both'))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            ax = df.plot(grid=True)
        self.assertTrue(axes_has_grid(ax, axis='both'))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            ax = df.plot(grid=False)
        self.assertFalse(axes_has_grid(ax))
        plt.close()

    def test_scatter_matrix_grid(self):
        df =  tm.makeDataFrame()

        with rc_context(rc={'axes.grid': False}):
            result = plotting.scatter_matrix(df, grid=None)
        for ax in result.flat:
            self.assertFalse(axes_has_grid(ax))
        plt.close()

        with rc_context(rc={'axes.grid': False}):
            result = plotting.scatter_matrix(df, grid=True)
        for ax in result.flat:
            self.assertTrue(axes_has_grid(ax, axis='both'))
        plt.close()

        with rc_context(rc={'axes.grid': False}):
            result = plotting.scatter_matrix(df, grid=False)
        for ax in result.flat:
            self.assertFalse(axes_has_grid(ax))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            result = plotting.scatter_matrix(df, grid=None)
        for ax in result.flat:
            self.assertTrue(axes_has_grid(ax, axis='both'))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            result = plotting.scatter_matrix(df, grid=True)
        for ax in result.flat:
            self.assertTrue(axes_has_grid(ax, axis='both'))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            result = plotting.scatter_matrix(df, grid=False)
        for ax in result.flat:
            self.assertFalse(axes_has_grid(ax))
        plt.close()

    def test_andrews_curves_grid(self):
        import os
        from pandas import read_csv

        path = os.path.join('../../tests/data', 'iris.csv')
        df = read_csv(path)

        with rc_context(rc={'axes.grid': False}):
            ax = plotting.andrews_curves(df, 'Name')
        self.assertFalse(axes_has_grid(ax))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            ax = plotting.andrews_curves(df, 'Name')
        self.assertTrue(axes_has_grid(ax, axis='both'))
        plt.close()

    def test_parallel_coordinates_grid(self):
        import os
        from pandas import read_csv

        path = os.path.join('../../tests/data', 'iris.csv')
        df = read_csv(path)

        with rc_context(rc={'axes.grid': False}):
            ax = plotting.parallel_coordinates(df, 'Name')
        self.assertFalse(axes_has_grid(ax))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            ax = plotting.parallel_coordinates(df, 'Name')
        self.assertTrue(axes_has_grid(ax, axis='both'))
        plt.close()

    def test_autocorrelation_plot_grid(self):
        ts = tm.makeTimeSeries()

        with rc_context(rc={'axes.grid': False}):
            ax = plotting.autocorrelation_plot(ts)
        self.assertFalse(axes_has_grid(ax))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            ax = plotting.autocorrelation_plot(ts)
        self.assertTrue(axes_has_grid(ax, axis='both'))
        plt.close()

    def test_grouped_hist_grid(self):
        df = pd.DataFrame(np.random.randn(500, 2), columns=['A', 'B'])
        df['C'] = np.random.randint(0, 4, 500)

        with rc_context(rc={'axes.grid': False}):
            axes = plotting.grouped_hist(df.A, by=df.C, grid=None)
        for ax in axes.flat:
            self.assertFalse(axes_has_grid(ax))
        plt.close()

        with rc_context(rc={'axes.grid': False}):
            axes = plotting.grouped_hist(df.A, by=df.C, grid=False)
        for ax in axes.flat:
            self.assertFalse(axes_has_grid(ax))
        plt.close()

        with rc_context(rc={'axes.grid': False}):
            axes = plotting.grouped_hist(df.A, by=df.C, grid=True)
        for ax in axes.flat:
            self.assertTrue(axes_has_grid(ax, axis='both'))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            axes = plotting.grouped_hist(df.A, by=df.C, grid=None)
        for ax in axes.flat:
            self.assertTrue(axes_has_grid(ax, axis='both'))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            axes = plotting.grouped_hist(df.A, by=df.C, grid=False)
        for ax in axes.flat:
            self.assertFalse(axes_has_grid(ax))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            axes = plotting.grouped_hist(df.A, by=df.C, grid=True)
        for ax in axes.flat:
            self.assertTrue(axes_has_grid(ax, axis='both'))
        plt.close()

    def test_boxplot_grid(self):
        import string
        df = pd.DataFrame(np.random.randn(6, 4),
                          index=list(string.ascii_letters[:6]),
                          columns=['one', 'two', 'three', 'four'])
        df['indic'] = ['foo', 'bar'] * 3
        df['indic2'] = ['foo', 'bar', 'foo'] * 2

        with rc_context(rc={'axes.grid': False}):
            plotting.boxplot(df, grid=None)
        self.assertFalse(axes_has_grid(plt.gca()))
        plt.close()

        with rc_context(rc={'axes.grid': False}):
            plotting.boxplot(df, grid=False)
        self.assertFalse(axes_has_grid(plt.gca()))
        plt.close()

        with rc_context(rc={'axes.grid': False}):
            plotting.boxplot(df, grid=True)
        self.assertTrue(axes_has_grid(plt.gca(), axis='both'))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            plotting.boxplot(df, grid=None)
        self.assertTrue(axes_has_grid(plt.gca(), axis='both'))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            plotting.boxplot(df, grid=False)
        self.assertFalse(axes_has_grid(plt.gca()))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            plotting.boxplot(df, grid=True)
        self.assertTrue(axes_has_grid(plt.gca(), axis='both'))
        plt.close()

    def test_boxplot_by_grid(self):
        import string
        df = pd.DataFrame(np.random.randn(6, 4),
                          index=list(string.ascii_letters[:6]),
                          columns=['one', 'two', 'three', 'four'])
        df['indic'] = ['foo', 'bar'] * 3

        with rc_context(rc={'axes.grid': False}):
            axes = plotting.boxplot(df, column=['one', 'two'], by='indic', grid=None)
        for ax in axes.flat:
            self.assertFalse(axes_has_grid(ax))
        plt.close()

        with rc_context(rc={'axes.grid': False}):
            axes = plotting.boxplot(df, column=['one', 'two'], by='indic', grid=False)
        for ax in axes.flat:
            self.assertFalse(axes_has_grid(ax))
        plt.close()

        with rc_context(rc={'axes.grid': False}):
            axes = plotting.boxplot(df, column=['one', 'two'], by='indic', grid=True)
        for ax in axes.flat:
            self.assertTrue(axes_has_grid(ax, axis='both'))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            axes = plotting.boxplot(df, column=['one', 'two'], by='indic', grid=None)
        for ax in axes.flat:
            self.assertTrue(axes_has_grid(ax, axis='both'))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            axes = plotting.boxplot(df, column=['one', 'two'], by='indic', grid=False)
        for ax in axes.flat:
            self.assertFalse(axes_has_grid(ax))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            axes = plotting.boxplot(df, column=['one', 'two'], by='indic', grid=True)
        for ax in axes.flat:
            self.assertTrue(axes_has_grid(ax, axis='both'))
        plt.close()

    def test_scatter_plot_grid(self):
        df = pd.DataFrame(np.random.randn(100, 2))

        with rc_context(rc={'axes.grid': False}):
            plotting.scatter_plot(df, x=0, y=1, grid=None)
        self.assertFalse(axes_has_grid(plt.gca()))
        plt.close()

        with rc_context(rc={'axes.grid': False}):
            plotting.scatter_plot(df, x=0, y=1, grid=False)
        self.assertFalse(axes_has_grid(plt.gca()))
        plt.close()

        with rc_context(rc={'axes.grid': False}):
            plotting.scatter_plot(df, x=0, y=1, grid=True)
        self.assertTrue(axes_has_grid(plt.gca(), axis='both'))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            plotting.scatter_plot(df, x=0, y=1, grid=None)
        self.assertTrue(axes_has_grid(plt.gca(), axis='both'))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            plotting.scatter_plot(df, x=0, y=1, grid=False)
        self.assertFalse(axes_has_grid(plt.gca()))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            plotting.scatter_plot(df, x=0, y=1, grid=True)
        self.assertTrue(axes_has_grid(plt.gca(), axis='both'))
        plt.close()

    def test_scatter_plot_by_grid(self):
        df = pd.DataFrame(np.random.randn(100, 2))
        grouper = pd.Series(np.repeat([1, 2, 3, 4, 5], 20), df.index)

        with rc_context(rc={'axes.grid': False}):
            result = plotting.scatter_plot(df, x=0, y=1, by=grouper, grid=None)
        for ax in result[1].flat:
            self.assertFalse(axes_has_grid(ax))
        plt.close()

        with rc_context(rc={'axes.grid': False}):
            result = plotting.scatter_plot(df, x=0, y=1, by=grouper, grid=False)
        for ax in result[1].flat:
            self.assertFalse(axes_has_grid(ax))
        plt.close()

        with rc_context(rc={'axes.grid': False}):
            result = plotting.scatter_plot(df, x=0, y=1, by=grouper, grid=True)
        for ax in result[1].flat:
            self.assertTrue(axes_has_grid(ax, axis='both'))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            result = plotting.scatter_plot(df, x=0, y=1, by=grouper, grid=None)
        for ax in result[1].flat:
            self.assertTrue(axes_has_grid(ax, axis='both'))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            result = plotting.scatter_plot(df, x=0, y=1, by=grouper, grid=False)
        for ax in result[1].flat:
            self.assertFalse(axes_has_grid(ax))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            result = plotting.scatter_plot(df, x=0, y=1, by=grouper, grid=True)
        for ax in result[1].flat:
            self.assertTrue(axes_has_grid(ax, axis='both'))
        plt.close()

    def test_hist_frame_grid(self):
        df = tm.makeDataFrame()

        with rc_context(rc={'axes.grid': False}):
            result = plotting.hist_frame(df, grid=None)
        for ax in result.flat:
            self.assertFalse(axes_has_grid(ax))
        plt.close()

        with rc_context(rc={'axes.grid': False}):
            result = plotting.hist_frame(df, grid=False)
        for ax in result.flat:
            self.assertFalse(axes_has_grid(ax))
        plt.close()

        with rc_context(rc={'axes.grid': False}):
            result = plotting.hist_frame(df, grid=True)
        for ax in result.flat:
            self.assertTrue(axes_has_grid(ax, axis='both'))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            result = plotting.hist_frame(df, grid=None)
        for ax in result.flat:
            self.assertTrue(axes_has_grid(ax, axis='both'))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            result = plotting.hist_frame(df, grid=False)
        for ax in result.flat:
            self.assertFalse(axes_has_grid(ax))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            result = plotting.hist_frame(df, grid=True)
        for ax in result.flat:
            self.assertTrue(axes_has_grid(ax, axis='both'))
        plt.close()

    def test_hist_series_grid(self):
        s = tm.makeFloatSeries()

        with rc_context(rc={'axes.grid': False}):
            ax = plotting.hist_series(s, grid=None)
        self.assertFalse(axes_has_grid(ax))
        plt.close()

        with rc_context(rc={'axes.grid': False}):
            ax = plotting.hist_series(s, grid=False)
        self.assertFalse(axes_has_grid(ax))
        plt.close()

        with rc_context(rc={'axes.grid': False}):
            ax = plotting.hist_series(s, grid=True)
        self.assertTrue(axes_has_grid(ax, axis='both'))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            ax = plotting.hist_series(s, grid=None)
        self.assertTrue(axes_has_grid(ax, axis='both'))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            ax = plotting.hist_series(s, grid=False)
        self.assertFalse(axes_has_grid(ax))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            ax = plotting.hist_series(s, grid=True)
        self.assertTrue(axes_has_grid(ax, axis='both'))
        plt.close()

    def test_boxplot_frame_groupby_grid(self):
        df = pd.DataFrame(np.random.rand(10, 2), columns=['Col1', 'Col2'])
        df['X'] = pd.Series(['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'])
        grouped = df.groupby(by='X')

        with rc_context(rc={'axes.grid': False}):
            plotting.boxplot_frame_groupby(grouped, grid=None)
        for ax in plt.gcf().axes:
            self.assertFalse(axes_has_grid(ax))
        plt.close()

        with rc_context(rc={'axes.grid': False}):
            plotting.boxplot_frame_groupby(grouped, grid=False)
        for ax in plt.gcf().axes:
            self.assertFalse(axes_has_grid(ax))
        plt.close()

        with rc_context(rc={'axes.grid': False}):
            plotting.boxplot_frame_groupby(grouped, grid=True)
        for ax in plt.gcf().axes:
            self.assertTrue(axes_has_grid(ax, axis='both'))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            plotting.boxplot_frame_groupby(grouped, grid=None)
        for ax in plt.gcf().axes:
            self.assertTrue(axes_has_grid(ax, axis='both'))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            plotting.boxplot_frame_groupby(grouped, grid=False)
        for ax in plt.gcf().axes:
            self.assertFalse(axes_has_grid(ax))
        plt.close()

        with rc_context(rc={'axes.grid': True}):
            plotting.boxplot_frame_groupby(grouped, grid=True)
        for ax in plt.gcf().axes:
            self.assertTrue(axes_has_grid(ax, axis='both'))
        plt.close()

if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)

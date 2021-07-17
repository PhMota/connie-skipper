from xarray import (
    register_dataarray_accessor, 
    register_dataset_accessor
)
from xarray.core.extensions import AccessorRegistrationWarning
import xarray.plot.utils
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
import traceback
import warnings
warnings.simplefilter("ignore", category=AccessorRegistrationWarning)

ORIGINAL_ROBUST_PERCENTILE = xarray.plot.utils.ROBUST_PERCENTILE

xbins = lambda b: (b[1:] + b[:-1])/2
dbins = lambda b: b[1]-b[0]

@register_dataarray_accessor("skipper")
class SkipperDataArrayAccessor:
    """
    Access methods for DataArrays for CONNIE skipper CCDs.
    
    Methods and attributes can be accessed through the `.skipper` attribute.
    """
    def __init__(self, da):
        self.da = da
        plt.rcParams.update({
            "image.origin": "lower",
            "image.aspect": 1,
            "text.usetex": True,
            "grid.alpha": .5,
        })
        
    def mad( self, dim = None, axis = None, skipna = None, keep_attrs = True, **kwargs):
        """Reduce this DataArray’s data by applying median absolute deviation along some dimension(s).

        Parameters
        """
        med = abs( self.da - self.da.median(dim, axis, skipna, keep_attrs = True, **kwargs) )
        return med.median( dim, axis, skipna, keep_attrs = True, **kwargs )/0.6744897501960817
    
    def center( self, dim = None, axis = None, skipna = True, mode = "median", **kwargs ):
        """global image
        
        """
        if dim is None:
            dim = ["row", "col"]
        return self.stats( dim, axis, skipna, mode, **kwargs )
    
    def centralize( self, dim = None, axis = None, skipna = True, mode = "median", **kwargs ):
        da = self.da - self.center( dim, axis, skipna, mode, **kwargs )
        da.attrs = self.da.attrs
        return da
    
    def trim( self, conds = None, na = np.nan ):
        """trim defective columns
        """
        from functools import reduce
        if conds is None:
            conds = reduce( np.logical_or, [
                self.da["col"] <= 8, 
                self.da["row"] <= 1,
                abs(self.da["col"] - self.overscan_coord("col")) <= 1,
                abs(self.da["row"] - self.overscan_coord("row")) <= 1,
            ])
        return xr.where( conds, na, self.da )

    def overscan_coord(self, dim):
        if dim == "col":
            return 349
        elif dim == "row":
            return 512
        else:
            raise Exception(f"not implemented {dim}")

    def overscan(self, dim, keep_size = False):
        if keep_size:
            return xr.where(
                self.da[dim] < self.overscan_coord(dim),
                np.nan,
                self.da
            )
        return self.da.sel( 
            {dim: slice(self.overscan_coord(dim), None)} 
        )

    @property
    def c_os(self):
        return self.col_overscan

    def modulation(self, dim = None, axis = None, skipna = None, mode = "median", **kwargs ):
        """generate a modulation image according to the given dimenstion and statistical reduction
        
        Parameters
        ----------
        dim : str, seq or dict of {dim: selections}
        """
        if isinstance( dim, str ):
            dim = [dim]
        if isinstance( dim, (list, tuple)):
            dim = {d: slice(self.overscan_coord(d), None) for d in dim}

#         print( dim )
        terms = [ self.da.sel({dim:sel}).skipper.stats( dim, axis, skipna, mode, **kwargs ) for dim, sel in dim.items() ]
        ret = xr.zeros_like(self.da)
        for term in terms:
            ret += term
        return ret/len(terms)

    def centralize_os( self, dim = None, axis = None, skipna = True, mode = "median", **kwargs ):
        da = self.da - self.overscan(dim).skipper.center( None, axis, skipna, mode, **kwargs )
        da.attrs = self.da.attrs
        return da

    def demodulate(self, dim = None, axis = None, skipna = None, mode = "median", **kwargs ):
        da = self.da - self.modulation( dim, axis, skipna, **kwargs )
        da.attrs = self.da.attrs
        return da
    
    def gaussianfit(self, energy, distribution, peaks, log = None):
        from scipy.stats import norm
        from scipy.optimize import curve_fit
        
        multinorm = lambda x, mu, sigma, gain, *A: (
            np.sum( [ a*norm.pdf(x, mu + i*gain, sigma)/norm.pdf(0, 0, sigma) for i, a in enumerate(A)], axis=0 )
        )
        p0 = [
            energy[peaks].min(), 
            np.nanmean( energy[peaks[1:]] - energy[peaks[:-1]] )/2 if peaks.size > 1 else 100,
            np.nanmean( energy[peaks[1:]] + energy[peaks[:-1]] )/2 if peaks.size > 1 else 100,
        ]
        p0 = np.concatenate( [p0, distribution[peaks]], axis=-1 )
        if log: log.value += f"p0 = {p0}<br>"

        bounds = (
            [-np.inf, np.inf],
            [10, np.inf],
            [10, np.inf],
            *( [ [1, np.inf] ]*peaks.size )
        )
        if log: log.value += f"bounds = {bounds}<br>"
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            try:
                popt, pcov = curve_fit( 
                    multinorm, 
                    energy, 
                    distribution,
                    p0 = p0,
                    bounds = tuple(zip(*bounds))
                )
                
            except ValueError as e:
                if log: log.value += f"<b>{e}</b><br>"
                raise e
        if log: log.value += f"popt = {popt}<br>"
        ret = multinorm(energy, *popt)
        assert ret.shape == energy.shape
        return (energy[ret>1], ret[ret>1]), popt
        
    
    def stats( self, dim = None, axis = None, skipna = None, mode = "median", **kwargs  ):
        """dispatcher for different statistics functions
        
        Parameters
        ----------
        mode : {median, mean, min, max, std, mad}
            options for the statistical reduction to be used
            
        Returns
        -------
        reduced : DataArray
        """
        if mode == "median":
            return self.da.median( dim, axis, skipna=True, **kwargs )
        elif mode == "mean":
            return self.da.mean( dim, axis, skipna=True, **kwargs )
        elif mode == "min":
            return self.da.min( dim, axis, skipna=True, **kwargs )
        elif mode == "max":
            return self.da.max( dim, axis, skipna=True, **kwargs )
        elif mode == "std":
            return self.da.std( dim, axis, skipna=True, **kwargs )
        elif mode == "mad":
            return self.da.skipper.mad( dim, axis, skipna=True, **kwargs )
        else:
            raise Exception(f"mode {mode} not implemented")

    def plot_projection(
        self, 
        dim = None, 
        axis = None, 
        skipna = None, 
        hue = None,
        mode = "median",
        ax = None,
        **kwargs
    ):
        self.overscan("col").skipper.stats( dim, axis, skipna, mode ).plot( hue = hue, ax=ax, zorder=2, label='os', **kwargs )
        self.da.skipper.stats( dim, axis, skipna, mode ).plot( hue = hue, ax=ax, zorder=1, label='all', **kwargs )
        if ax:
            ax.grid(True)
        plt.legend()
        return ax
    
    def plot_imshow( self, aspect = 1, ax=None, energy_percentile = 0, **kwargs ):
        """generates the heatmap of the data
        
        Parameters
        ----------
        x : string
            the dimension to be used in the x-axis
        y : string
            the dimension to be used in the y-axis
        log : object
            an object with `value` porperty where the output and error will be redirected to
        """
        log = kwargs.pop("log", None)
        x = kwargs.pop("x", "row")
        y = kwargs.pop("y", "col")
        if ax is None:
            fig, ax = plt.subplots(
                figsize = kwargs.pop("figsize", (8,6))
            )
#         if energy_percentile:
#             xarray.plot.utils.ROBUST_PERCENTILE = energy_percentile
#             print( xarray.plot.utils.ROBUST_PERCENTILE )
        emin = np.percentile( self.da.data.flatten(), energy_percentile )
        emax = np.percentile( self.da.data.flatten(), 100 - energy_percentile )
        if log: 
            log.value += f"imgE = [{emin}, {emax}]<br>"
        else:
            print(f"imgE = [{emin}, {emax}]")
        obj = self.da.plot.imshow(
            x = x, 
            y = y, 
            ax = ax, 
            robust = True, 
            vmin = emin, 
            vmax = emax, 
            **kwargs 
        )
        obj.axes.set_aspect( aspect )
        return obj
    
    def plot_spectrum(self, bins=None, ax=None, **kwargs):
        log = kwargs.pop("log", None)
        if ax is None:
            fig, ax = plt.subplots(
                figsize = kwargs.pop("figsize", (8,6))
            )
        os = self.overscan("col")
        energy_percentile = kwargs.pop("energy_percentile", None)
        
        if energy_percentile is None:
            med = os.skipper.stats(["col", "row"], mode="median").data
            mad = os.skipper.stats(["col", "row"], mode="mad").data
            emin = med - energy_percentile*mad
            emax = med + energy_percentile*mad
            if log: log.value += f"E = [{emin},{emax}]<br>"
        else:
            emin = np.nanpercentile(os.data.flatten(), energy_percentile)
            emax = np.nanpercentile(os.data.flatten(), 100 - energy_percentile)

        bins = np.arange( emin, emax, 1. if emax - emin < 5000 else (emax-emin)/5000 )
        if log: log.value += f"dbin = {dbins(bins)}<br>"
        hist, *_ = xr.where(os != 0, os, np.nan).plot.hist( bins = bins, yscale="log", ax = ax, zorder=2, label="os", **kwargs )
        xr.where( self.da != 0, self.da, np.nan).plot.hist( bins = bins, yscale="log", ax = ax, zorder=1, label='all', **kwargs )
        
        from scipy.signal import find_peaks
        peaks, properties = find_peaks( 
            hist, 
            height = 10,
            distance = int(300/dbins(bins)),
        )
        if log: log.value += f"peaks{(peaks.size)} = {peaks}<br>"
        ax.plot( xbins(bins)[peaks], hist[peaks], "ro" )
        try:
            xy, popt = self.gaussianfit( xbins(bins), hist, peaks, log )
            ax.plot( *xy, "r--", label = fr"$\mu={popt[0]:.1f}$"+"\n"+fr"$\sigma={popt[1]:.1f}$"+"\n"+fr"$g={popt[2]:.1f}$" )
        except ValueError:
            if log: log.value += traceback.format_exec()
        ax.grid(True)
        ax.legend(loc="upper right")
        return ax
    
    def plot_full( 
        self, 
        x="col", 
        y="row",
        mode="mean", 
        fig = None,
        **kwargs 
    ):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        log = kwargs.pop("log", None)
        progress = kwargs.pop("progress_bar", None)
        if progress:
            progress_bar, progress_min, progress_max = progress
            progress_bar.value, progress_bar.description = progress_min, "creating fig"
            factor = lambda n: progress_min + n*(progress_max - progress_min)
        if fig is None:
            fig, axImg = plt.subplots(
                figsize = kwargs.pop("figsize", (8,6))
            )
        else:
            fig.clf()
            axImg = fig.add_subplot(111)
        fig.canvas.toolbar_position = 'bottom'
        if "suptitle" in kwargs:
            fig.suptitle(kwargs["suptitle"])
        ### panels
        divider = make_axes_locatable(axImg)
        
        ### image panel
        if progress:
            progress_bar.value, progress_bar.description = factor(.2), "imshow"
        
        energy_percentile = kwargs.pop("energy_percentile", None)
#         os = self.overscan("col")
#         med = os.skipper.stats(["col","row"], mode="median").data
#         mad = os.skipper.stats(["col", "row"], mode="mad").data
#         emin = med - energy_percentile*mad
#         emax = med + energy_percentile*mad
#         im.set_clim((emin, emax))
        im = self.plot_imshow( 
            x = x, 
            y = y, 
            ax = axImg, 
#             vmin = emin,
#             vmax = emax,
#             robust = kwargs.pop("robust", False),
            energy_percentile = energy_percentile,
            log = log
        )
        axImg.set_title("")
        
        ### colorbar
        if progress:
            progress_bar.value, progress_bar.description = factor(.4), "colobar"
        
        axColor_left = divider.append_axes(
            "left", 
            .1, 
            pad = 0.6, 
        )
        extend = im.colorbar.extend
        boundaries = im.colorbar._boundaries
        im.colorbar.remove()
        cbar = fig.colorbar(
            im, 
            cax = axColor_left, 
            extend = extend 
        )
        axColor_left.yaxis.set_label_position("left")
        axColor_left.yaxis.tick_left()
        ### top panel
        axProj_top = None
        if kwargs.pop("yproj", False):
            if progress:
                progress_bar.value, progress_bar.description = factor(.5), "yproj"            
            axProj_top = divider.append_axes(
                "top", 
                1.5, 
                pad = 0.1, 
                sharex = axImg,
            )
            axProj_top.xaxis.set_label_position('top')
            axProj_top.xaxis.tick_top()
            self.plot_projection( 
                ax = axProj_top,
                x = x, 
                dim = y,
            )
            axProj_top.grid(True)
            axProj_top.set_ylabel(f"{mode}(col)")

        ### right panel
        axProj_right = None
        if kwargs.pop("xproj", False):
            if progress:
                progress_bar.value, progress_bar.description = factor(.6), "xproj"
            axProj_right = divider.append_axes(
                "right", 
                1.5, 
                pad = 0.1, 
                sharey = axImg,
            )
            axProj_right.yaxis.set_label_position('right')
            axProj_right.yaxis.tick_right()
            self.plot_projection( 
                ax = axProj_right,
                y = y, 
                dim = x,
                mode = mode,
            )
            axProj_right.grid(True)
            axProj_right.set_title("")
            axProj_right.set_xlabel(f"{mode}(row)")
        
        ### right panel
        if kwargs.pop("spectrum", False) and axProj_top is None:
            if progress:
                progress_bar.value, progress_bar.description = factor(.7), "spectrum"
            axSpectrum = divider.append_axes(
                "top",
                1.5,
                pad = 0.1,
            )
            axSpectrum.yaxis.set_label_position('right')
            axSpectrum.yaxis.tick_right()
            axSpectrum.xaxis.set_label_position('top')
            axSpectrum.xaxis.tick_top()
            self.plot_spectrum( 
                bins = kwargs.pop("bins", 10), 
                ax = axSpectrum, 
                energy_percentile = energy_percentile,
                log = log
            )
#         print( "histogram done img" )
#         ### aux panel for resizing
#         axAux = divider.append_axes(
#             "right",
#             1.5,
#             pad = 0.1,
#             sharey = axProj_top,
#             sharex = axProj_right,
# #             add_to_figure = False
#         )
#         axAux.plot( [0], [0] )
#         axAux.set_aspect(1)
        
        ### finalize
        fig.canvas.layout.width = '100%'
        plt.tight_layout()
        plt.draw()
        if progress:
            progress_bar.value, progress_bar.description = progress_max, "done plot"
        return fig

# attempting to mimmick plt.Axes with all the direction information swapped
# works for the first draw, but does not refresh
class SwapedAxes(Axes):
    def __init__(self, ax):
        self._axis_names = ("y", "x")
        self._ax = ax
        self.xaxis, self.yaxis = ax.yaxis, ax.xaxis
        self._autoscaleXon = self._autoscaleYon = True
        self._animated = ax._animated
        self.dataLim = type(ax.dataLim)( np.array(ax.dataLim.get_points())[:, ::-1].tolist() )
        print(ax.dataLim, "->", self.dataLim)
        self._viewLim = type(ax._viewLim)( np.array(ax._viewLim.get_points())[:, ::-1].tolist() )
        print(ax._viewLim, "->", self._viewLim)
        
        self.spines = type(ax.spines)([ ("left", ax.spines["top"]), ("right", ax.spines["bottom"]), ("top", ax.spines["left"]), ("bottom", ax.spines["bottom"]) ])
        print( ax.spines, "->", self.spines )
        
        print( ax._axes._position, "->", type(ax._axes._position)( np.array(ax._axes._position.get_points())[:, ::-1].tolist() ) )
#         self._axes = type(ax._axes)( 
#             type(ax._axes._position)(
#                 ax.fig,
#                 ax._axes._position.get_points()[:, ::-1].tolist()
#             ) 
#         )
        self._axes = type(
            "_axes",
            (),
            dict( 
                _position = type(ax._axes._position)( np.array(ax._axes._position.get_points())[:, ::-1].tolist() ),
                xaxis = ax._axes.yaxis,
                yaxis = ax._axes.xaxis,
                __repr__ = ax._axes.__repr__,
                __str__ = ax._axes.__str__
            )
        )
        print( ax._axes._position, "->", self._axes._position )
        
        print( ax._axes, "->", self._axes )
        
    
    def get_xlim(self):
        return self._ax.get_ylim()
    
    def get_ylim(self):
        return self._ax.get_xlim()
            
    def __getattr__(self, attr):
        print()
        print( attr, ":" )
        print( getattr(self._ax, attr) )
        return getattr(self._ax, attr)
    
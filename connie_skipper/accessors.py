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
import pandas as pd
# ORIGINAL_ROBUST_PERCENTILE = xarray.plot.utils.ROBUST_PERCENTILE

xbins = lambda b: (b[1:] + b[:-1])/2
dbins = lambda b: b[1]-b[0]

class Plottable:
    def __init__(self):
        pass
    
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
        try:
            flat = self.da.data.flatten()
            emin = np.percentile( flat[~np.isnan(flat)], energy_percentile )
            emax = np.percentile( flat[~np.isnan(flat)], 100 - energy_percentile )
        except IndexError:
            emin = flat[~np.isnan(flat)].min()
            emax = flat[~np.isnan(flat)].max()
            if log: 
                log.value += f"<b style='color:red'>failed percentile</b><br>"

        obj = self.da.plot.imshow(
            x = x, 
            y = y, 
            ax = ax, 
            robust = True, 
            vmin = emin, 
            vmax = emax,
            **kwargs 
        )
        ax.set_xlabel(r"{x} [pix]")
        ax.set_ylabel(r"{y} [pix]")
        obj.axes.set_aspect( aspect )
        return obj
    
    def plot_distribution( self, bins = None, cond = None, ax = None, **kwargs):
        log = kwargs.pop("log", None)
        mu = kwargs.pop("mu", None)
        sigma = kwargs.pop("sigma", None)
        g = kwargs.pop("g", None)
        label = kwargs.pop("label", None)
        zorder = kwargs.pop("zorder", None)

        if cond is not None:
            da = xr.where( cond, self.da, np.nan )
        else:
            da = self.da
        color = kwargs.pop("color", None)
        hist, *_ = plt.hist(
            da.data.flatten(), 
            bins = bins, 
            weights = np.ones_like(da.data.flatten())/dbins(bins), 
            color = color,
            alpha = kwargs.pop("alpha", .5),
            zorder = zorder,
        )
        from scipy.signal import find_peaks
        peaks, properties = find_peaks( 
            hist, 
            height = kwargs.pop("height", 10),
            prominence = 10,
            distance = int( kwargs.pop("distance", 300)/dbins(bins)),
        )
        if log: 
            with np.printoptions(precision=1, suppress=True, threshold=10):
#                 log.value += f"<b>peaks</b>[{peaks.size}] = {peaks}<br>"
                log.value += f"<b>peaks</b>[{peaks.size}] = {xbins(bins)[peaks]}<br>"
        ax.plot( 
            xbins(bins)[peaks], 
            hist[peaks], 
            color = color,
            marker = 'o',
            linestyle = "",
            zorder = zorder,
        )
        try:
            x, y, popt_dict, perr_dict = self.gaussianfit( xbins(bins), hist, peaks, log, mu=mu, sigma=sigma, g=g, label=label, **kwargs )
        except ValueError as e:
            if log: 
                with np.printoptions(precision=1, suppress=True, threshold=10):
                    log.value += f"<b>peaks</b>[{peaks.size}] = {peaks}<br>"
                    log.value += f"<b style='color:red'>failed to fit {e}</b><br>"
            raise e            
        else:
            ax.plot( 
                x, 
                y, 
                color = color,
                linestyle = '-',
                label = (
                    label
                    +"\n" + fr"$\mu={popt_dict['mu']:.1f}$"
                    +"\n" + fr"$\sigma={popt_dict['sigma']:.1f}$"
                    + ("\n" + fr"$g={popt_dict['gain']:.1f}$" if "gain" in popt_dict else "")
                    + ("\n" + fr"$\lambda={popt_dict['lambda']:.2f}$" if "lambda" in popt_dict else "")
                ),
                zorder = zorder,
            )
            return popt_dict, perr_dict
        
        
    def plot_spectrum(self, bins=None, ax=None, **kwargs):
        progress_bar = kwargs.pop( "progress", None )
        log = kwargs.pop( "log", None )
        if ax is None:
            fig, ax = plt.subplots(
                figsize = kwargs.pop("figsize", (8,6))
            )
        os = self.overscan("col")
        ac = self.active("col")
        energy_percentile = kwargs.pop("energy_percentile", None)
        
        flat = self.da.data.flatten()
        emin = np.nanpercentile( flat[~np.isnan(flat)], energy_percentile )
        emax = np.nanpercentile( flat[~np.isnan(flat)], 100 - energy_percentile )
        
        kwargs.pop("bins", None)
        bins = np.arange( emin, emax, 1. if emax - emin < 5000 else (emax-emin)/5000 )
        popt = None
        if log: 
            log.value += f"<b>overscan</b><br>"
        try:
            popt_os, perr_os = os.skipper.plot_distribution(
                cond = os != 0,
                bins = bins, 
                ax = ax, 
                zorder = 2, 
                color = "b",
                alpha = .5,
                label = "os",
                log = log,
                **kwargs 
            )
        except ValueError as e:
            if log: 
                log.value += f"E = [{emin:.1f},{emax:.1f}]<br>"
                log.value += f"<b>dbin</b> = {dbins(bins):.1f}<br>"
                log.value += f"<b style='color:red'>failed to plot the overscan {e}</b><br>"
#                 log.value += traceback.format_exc()
            pass
        
        if log: 
            log.value += f"<b>active</b><br>"            
        try:
            popt_ac, perr_ac = ac.skipper.plot_distribution(
                cond = ac != 0,
                bins = bins,
                ax = ax, 
                zorder = 1, 
                color = "orange",
                alpha = .5,
                label = "ac",
                log = log,
                mu = popt_os[0] if popt is not None else None,
                sigma = popt_os[1] if popt is not None else None,
                g = None,
                **kwargs 
            )
        except ValueError as e:
            if log: 
                log.value += f"E = [{emin:.1f},{emax:.1f}]<br>"
                log.value += f"<b>dbin</b> = {dbins(bins):.1f}<br>"
                log.value += f"<b style='color:red'>failed to plot the overscan</b><br>"
                log.value += traceback.format_exc()
            pass
        
        ax.grid(True)
        ax.set_yscale( "log" )
        ax.set_ylabel( r"$dN/dE$ [counts/ADU]" )
        ax.set_xlabel( r"$E$ [ADU]" )
        if "xaxis" in kwargs:
            for prop, value in kwargs["xaxis"].items():
                if isinstance(value, bool) and value:
                    getattr( ax.xaxis, prop )()
                else:
                    getattr( ax.xaxis, prop )(value)
        ax.set_title("")
        ax.legend( loc = "upper left", frameon = False, bbox_to_anchor = (1, 1) )
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
            fig = plt.figure(
                figsize = kwargs.pop("figsize", (8,6))
            )
            fig.canvas.toolbar_position = 'bottom'
        else:
            fig.clf()
        if "suptitle" in kwargs:
            fig.suptitle(kwargs["suptitle"])

        axImg = fig.add_subplot(111)
      
        ### panels
        divider = make_axes_locatable(axImg)
        
        ### image panel
        if progress:
            progress_bar.value, progress_bar.description = factor(.2), "imshow"
        
        energy_percentile = kwargs.pop("energy_percentile", None)
        im = self.plot_imshow( 
            x = x, 
            y = y, 
            ax = axImg, 
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
        axColor_left.set_ylabel(r"$E$ [ADU]")
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
                mode = mode,
            )
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
            axSpectrum.xaxis.set_label_position('top')
            axSpectrum.xaxis.tick_top()
            if "events" in self.da.attrs:
                axSpectrum.hist( 
                    [ np.sum(events) for events in self.da.attrs["events"]], 
                    bins=100
                )
                axSpectrum.set_yscale("log")
                axSpectrum.set_ylabel(r"$N$ [counts]")
            else:
                self.plot_spectrum( 
                    ax = axSpectrum, 
                    energy_percentile = energy_percentile,
                    log = log,
                    progress = progress_bar,
                )
        
        ### finalize
        fig.canvas.layout.width = '100%'
        plt.tight_layout()
        plt.draw()
        if progress:
            progress_bar.value, progress_bar.description = progress_max, "done plot"
        return fig
    

@register_dataarray_accessor("skipper")
class SkipperDataArrayAccessor(Plottable):
    """
    Access methods for DataArrays for CONNIE skipper CCDs.
    
    Methods and attributes can be accessed through the `.skipper` attribute.
    """
    def __init__(self, da):
        da.attrs["os_col"] = (349, 'first overscan column')
        self.da = da
        plt.rcParams.update({
            "image.origin": "lower",
            "image.aspect": 1,
            "text.usetex": True,
            "grid.alpha": .5,
        })
        
    def isel(self, **indexers_kwargs):
        da = self.da.isel(**indexers_kwargs)
        da.attrs['name'] = f"{self.da.attrs['name']}.isel({indexers_kwargs})"
        return da
        
    def mad( self, dim = None, axis = None, skipna = None, keep_attrs = True, **kwargs):
        """Reduce this DataArray’s data by applying median absolute deviation along some dimension(s).

        Parameters
        """
        med = abs( self.da - self.da.median(dim, axis, skipna, keep_attrs = True, **kwargs) )
        return med.median( dim, axis, skipna, keep_attrs = True, **kwargs )/0.6744897501960817

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
            da = self.da.median( dim, axis, skipna=True, **kwargs )
        elif mode == "mean":
            da = self.da.mean( dim, axis, skipna=True, **kwargs )
        elif mode == "min":
            da = self.da.min( dim, axis, skipna=True, **kwargs )
        elif mode == "max":
            da = self.da.max( dim, axis, skipna=True, **kwargs )
        elif mode == "std":
            da = self.da.std( dim, axis, skipna=True, **kwargs )
        elif mode == "mad":
            da = self.da.skipper.mad( dim, axis, skipna=True, **kwargs )
        else:
            raise Exception(f"mode {mode} not implemented")
        da.attrs['name'] = f"{self.da.attrs['name']}.{mode}({dim})"
        return da
    
    def center( self, dim = None, axis = None, skipna = True, mode = "median", **kwargs ):
        """global image
        
        """
        if dim is None:
            dim = ["row", "col"]
        return self.stats( dim, axis, skipna, mode, **kwargs )
    
    def centralize( self, dim = None, axis = None, skipna = True, mode = "median", **kwargs ):
        da = self.da - self.center( dim, axis, skipna, mode, **kwargs )
        da.attrs = self.da.attrs
        da.attrs["name"] = f"{da.attrs['name']}.centralize({dim})"
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
#                 abs(self.da["row"] - self.overscan_coord("row")) <= 1,
            ])
        da = xr.where( conds, na, self.da )
        da.attrs['name'] = f"{self.da.attrs['name']}.trim(col<8,row<1,abs(col-{self.overscan_coord('col')})<=1)"
        return da

    def overscan_coord(self, dim):
        if dim == "col":
            return self.da.attrs["os_col"][0]
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

    def active(self, dim, keep_size = False):
        if keep_size:
            return xr.where(
                self.da[dim] > self.overscan_coord(dim),
                np.nan,
                self.da
            )
        return self.da.sel( 
            {dim: slice(None, self.overscan_coord(dim))} 
        )

    @property
    def c_os(self):
        return self.col_overscan

    def centralize_os( self, dim = None, axis = None, skipna = True, mode = "median", **kwargs ):
        da = self.da - self.overscan(dim).skipper.center( None, axis, skipna, mode, **kwargs )
        da.attrs = self.da.attrs
        return da

    def modulation(self, dim = None, mode = "median", **kwargs ):
        """generate a modulation image according to the given dimenstion and statistical reduction
        
        Parameters
        ----------
        dim : str, seq or dict of {dim: slice}
            dimension to be consumed by the statistical function defined in `mode`.
            if given as a `dict` it uses the specified dimension and slice, otherwise uses the overscan slice
        mode : str (median)
            statistical function to be used
        """
        if isinstance( dim, str ):
            dim = [dim]
        if isinstance( dim, (list, tuple)):
            dim = {d: slice(self.overscan_coord(d), None) for d in dim}

#         print( dim )
        terms = [ self.da.sel({dim:sel}).skipper.stats( dim, mode = mode, **kwargs ) for dim, sel in dim.items() ]
        ret = xr.zeros_like(self.da)
        for term in terms:
            ret += term
        return ret/len(terms)

    def demodulate(self, dim = None, axis = None, skipna = None, mode = "median", **kwargs ):
        """new skipper with the modulation subtracted
        
        Parameters
        ----------
        dim : str, seq of dict of {dim: slice}
            the dimension to be reduced for the modulation computation
        
        """
        da = self.da - self.modulation( dim, **kwargs )
        da.attrs = self.da.attrs
        return da
    
    def histogram(self, binsize = 1, return_hist = False ):
        flat = self.da.data.flatten()
        emin = np.nanpercentile( flat[~np.isnan(flat)], energy_percentile )
        emax = np.nanpercentile( flat[~np.isnan(flat)], 100 - energy_percentile )

        bins = np.arange( emin, emax, binsize )
        hist, _ = np.histogram(
            flat, 
            bins = bins, 
            weights = np.ones_like(flat)/dbins(bins), 
        )
        self.da.attrs["hist"] = xr.DataArray(
            hist,
            dims = ["E"],
            coords = {"E": xbins(bins)}
        )
        if return_hist:
            return self.da.attrs["hist"]
        return self

#     def peaks()
    
    
    def gaussianfit2(self, cond = None, label = None, log = None, binsize = 1, **kwargs):
        """fits the distribution as a sum of gaussians
        
        Parameters
        ----------
        energy : array
            the energy-axis values (bin centers of the histogram)
        distribution : array
            the histogram-like values
        peaks : array
            indices for the initial guesses of the peak positions
        log : object
            object to redirect the output
        """
        from scipy.stats import norm, poisson
        from scipy.optimize import curve_fit
        from scipy.signal import find_peaks
        
        if cond is not None:
            da = xr.where( cond, self.da, np.nan )
        else:
            da = self.da

        hist, _ = np.histogram(
            self.da.data.flatten(), 
            bins = bins, 
            weights = np.ones_like(da.data.flatten())/dbins(bins), 
        )
        
        peaks, properties = find_peaks( 
            hist, 
            height = kwargs.pop("height", 10),
            prominence = kwargs.pop("prominence", 10),
            distance = int( kwargs.pop("distance", 300)/dbins(bins)),
        )
        bounds = {
            "mu": [-np.inf, np.inf],
            "sigma": [10, np.inf],
            "A": [1, np.inf],
            "gain": [10, np.inf],
            "lambda": [0, np.inf]
        }
        
        if len(peaks) == 1:
            args = ["mu", "sigma", "A"]
            fitfunc = lambda x, mu, sigma, A: (
                A*norm.pdf(x, mu, sigma)
            )
            p0 = [
                kwargs.pop( "mu", None ) or energy[peaks].min(),
                kwargs.pop( "sigma", None ) or 100,
                distribution[peaks[0]],
            ]
            if log: 
                with np.printoptions(precision=1, suppress=True, threshold=10):
                    log.value += f"<b>p0</b>={list(zip(args,p0))}<br>"
            
        elif len(peaks) >= 2:
            args = ["mu", "sigma", "gain", "lambda", "A"]
            fitfunc = lambda x, mu, sigma, gain, lamb_, A: (
                A*np.sum( [ poisson.pmf(i, lamb_) * norm.pdf(x, mu + i*gain, sigma) for i, _ in enumerate(peaks)], axis=0 ) 
            )
            p0 = [
                kwargs.pop( "mu", None ) or energy[peaks].min(),
                kwargs.pop( "sigma", None ) or (np.nanmean( energy[peaks[1:]] - energy[peaks[:-1]] )/2),
                np.nanmean( energy[peaks[1:]] - energy[peaks[:-1]] ) if peaks.size > 1 else 300,
                1,
                distribution[peaks[0]]
            ]
            bounds["gain"] = [p0[2]*.9, p0[2]*1.1]
            if log: 
                with np.printoptions(precision=3, suppress=True, threshold=10):
                    log.value += f"<b>p0</b>={list(zip(args, p0))}<br>"


        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            try:
                popt, pcov = curve_fit( 
                    fitfunc, 
                    energy, 
                    distribution,
                    p0 = p0,
                    bounds = tuple(zip(*[ bounds[arg] for arg in args ])),
                    sigma = np.where(distribution > 0, np.sqrt(distribution), 1)
                )
                perr = np.sqrt(np.diag(pcov))
            except ValueError as e:
                if log: 
                    log.value += f"<b>{e}</b><br>"
                raise e
        
        ret = fitfunc(energy, *popt)
        assert ret.shape == energy.shape
        popt_dict = { arg: val for arg, val in zip(args, popt)}
        perr_dict = { arg: val for arg, val in zip(args, perr)}
        if log:
            with np.printoptions(precision=3, suppress=True, threshold=10):
                log.value += f"<b>popt</b>={list(popt)}<br>"
                log.value += f"<b>perr</b>={list(perr)}<br>"
        return energy[ret>1], ret[ret>1], popt_dict, perr_dict

    def gaussianfit(self, energy, distribution, peaks, log = None, binsize = 1, label=None, **kwargs):
        """fits the distribution as a sum of gaussians
        
        Parameters
        ----------
        energy : array
            the energy-axis values (bin centers of the histogram)
        distribution : array
            the histogram-like values
        peaks : array
            indices for the initial guesses of the peak positions
        log : object
            object to redirect the output
        """
        from scipy.stats import norm, poisson
        from scipy.optimize import curve_fit
        bounds = {
            "mu": [-np.inf, np.inf],
            "sigma": [10, np.inf],
            "A": [1, np.inf],
            "gain": [10, np.inf],
            "lambda": [0, np.inf]
        }
        
        if len(peaks) == 1:
            args = ["mu", "sigma", "A"]
            fitfunc = lambda x, mu, sigma, A: (
                A*norm.pdf(x, mu, sigma)
            )
            p0 = [
                kwargs.pop( "mu", None ) or energy[peaks].min(),
                kwargs.pop( "sigma", None ) or 100,
                distribution[peaks[0]],
            ]
            if log: 
                with np.printoptions(precision=1, suppress=True, threshold=10):
                    log.value += f"<b>p0</b>={list(zip(args,p0))}<br>"
            
        elif len(peaks) >= 2:
            args = ["mu", "sigma", "gain", "lambda", "A"]
            fitfunc = lambda x, mu, sigma, gain, lamb_, A: (
                A*np.sum( [ poisson.pmf(i, lamb_) * norm.pdf(x, mu + i*gain, sigma) for i, _ in enumerate(peaks)], axis=0 ) 
            )
            p0 = [
                kwargs.pop( "mu", None ) or energy[peaks].min(),
                kwargs.pop( "sigma", None ) or (np.nanmean( energy[peaks[1:]] - energy[peaks[:-1]] )/2),
                np.nanmean( energy[peaks[1:]] - energy[peaks[:-1]] ) if peaks.size > 1 else 300,
                1,
                distribution[peaks[0]]
            ]
            bounds["gain"] = [p0[2]*.9, p0[2]*1.1]
            if log: 
                with np.printoptions(precision=3, suppress=True, threshold=10):
                    log.value += f"<b>p0</b>={list(zip(args, p0))}<br>"


        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            try:
                popt, pcov = curve_fit( 
                    fitfunc, 
                    energy, 
                    distribution,
                    p0 = p0,
                    bounds = tuple(zip(*[ bounds[arg] for arg in args ])),
                    sigma = np.where(distribution > 0, np.sqrt(distribution), 1)
                )
                perr = np.sqrt(np.diag(pcov))
            except ValueError as e:
                if log: 
                    log.value += f"<b>{e}</b><br>"
                raise e
        
        ret = fitfunc(energy, *popt)
        assert ret.shape == energy.shape
        popt_dict = { arg: val for arg, val in zip(args, popt)}
        perr_dict = { arg: val for arg, val in zip(args, perr)}
        if log:
            with np.printoptions(precision=3, suppress=True, threshold=10):
                log.value += f"<b>popt</b>={list(popt)},<br>"
                log.value += f"<b>perr</b>={list(perr)},<br>"
                if label:
                    log.value += f"<b>'{label}':</b> {list(popt)},<br>"
                    log.value += f"<b>'{label}err':</b> {list(perr)},<br>"
                
        return energy[ret>1], ret[ret>1], popt_dict, perr_dict
            
    def extract(self, threshold, nborder=0, struct="cross", log=None):
        from scipy import ndimage
        above_threshold = xr.where(self.da >= threshold, 1, 0)
        if struct == "cross":
            struct = ndimage.generate_binary_structure(2, 1)
        elif struct == "square":
            struct = ndimage.generate_binary_structure(2, 2)
        if nborder > 0:
            above_threshold = ndimage.binary_dilation(above_threshold, structure=struct, iterations=nborder)
        if log:
            log.value += f"<b>above</b><br>{above_threshold}<br>"
            log.value += f"<b>struct</b><br>{struct}<br>"
        labels, nclusters = ndimage.label(above_threshold, structure=struct)
        
        events = [ self.da.data[ labels == label ] for label in range(1, nclusters) ]
        
        da = xr.DataArray(
            self.da.data.astype(float),
            dims = self.da.dims,
            coords = self.da.coords,
            attrs = self.da.attrs
        )
        da.data[ labels==0 ] = np.nan
        da.attrs['events'] = events
        if log:
            log.value += f"{da}"
        return da

        
        

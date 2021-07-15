from xarray import (
    register_dataarray_accessor, 
    register_dataset_accessor
)
from xarray.core.extensions import AccessorRegistrationWarning
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes

import warnings
warnings.simplefilter("ignore", category=AccessorRegistrationWarning)

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

        print( dim )
        terms = [ self.da.sel({dim:sel}).skipper.stats( dim, axis, skipna, mode, **kwargs ) for dim, sel in dim.items() ]
        ret = xr.zeros_like(self.da)
        for term in terms:
            ret += term
        return ret/len(terms)
    
    def demodulate(self, dim = None, axis = None, skipna = None, mode = "median", **kwargs ):
        da = self.da - self.modulation( dim, axis, skipna, **kwargs )
        da.attrs = self.da.attrs
        return da
    
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
        **kwargs
    ):
        obj = self.da.skipper.stats( dim, axis, skipna, mode ).plot( hue = hue, **kwargs )
        plt.draw()
        return obj
    
    def plot_imshow( self, aspect = 1, ax=None, **kwargs ):
        x = kwargs.pop("x", "row")
        y = kwargs.pop("y", "col")
        if ax is None:
            fig, ax = plt.subplots(
                figsize = kwargs.pop("figsize", (8,6))
            )
        obj = self.da.plot.imshow(x=x, y=y, ax=ax, **kwargs )
        obj.axes.set_aspect( aspect )
        plt.draw()
        return obj
    
    def plot_full( 
        self, 
        x="col", 
        y="row", 
        mode="median", 
        fig = None,
        **kwargs 
    ):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
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
        im = self.plot_imshow( 
            x = x, 
            y = y, 
            ax = axImg, 
            robust = kwargs.pop("robust", False),
        )
        axImg.set_title("")
        
        ### colorbar
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
#             orientation = "horizontal", 
            extend = extend 
        )
        axColor_left.yaxis.set_label_position("left")
        axColor_left.yaxis.tick_left()

        ### top panel
        if kwargs.pop("yproj", False):
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
        if kwargs.pop("xproj", False):
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
    
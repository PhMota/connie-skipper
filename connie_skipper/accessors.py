from xarray import register_dataarray_accessor, register_dataset_accessor
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

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
        return self.da.skipper.stats( dim, axis, skipna, mode ).plot( hue = hue, **kwargs )
    
    def plot_imshow( self, aspect = 1, **kwargs ):
        x = kwargs.pop("x", "row")
        y = kwargs.pop("y", "col")
        obj = self.da.plot.imshow(x=x, y=y, **kwargs )
        if isinstance(obj, xr.plot.FacetGrid ):
            for ax in obj.axes:
                plt.sca(ax).set_aspect( aspect )
        else:
            obj.axes.set_aspect( aspect )
        return obj
    
    def plot_full( self, mode="median", fig = None, **kwargs ):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        fig, axImg = plt.subplots(
            figsize = kwargs.pop("figsize", (10,5))
        )
        ### panels
        divider = make_axes_locatable(axImg)
        
        ### image panel
        im = self.plot_imshow( 
            x = "row", 
            y = "col", 
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
        axProj_top = divider.append_axes(
            "top", 
            1.5, 
            pad = 0.1, 
            sharex = axImg
        )
        axProj_top.xaxis.set_label_position('top')
        axProj_top.xaxis.tick_top()
        self.plot_projection( 
            ax = axProj_top,
            x = "row", 
            dim = "col",
        )
        axProj_top.grid(True)
        axProj_top.set_ylabel(f"{mode}(col)")        

        ### right panel
        axProj_right = divider.append_axes(
            "right", 
            1.5, 
            pad = 0.1, 
            sharey = axImg,
#             sharex = axProj_top.yaxis
        )
        axProj_right.yaxis.set_label_position('right')
        axProj_right.yaxis.tick_right()
        self.plot_projection( 
            ax = axProj_right,
            y = "col", 
            dim = "row",
            mode = mode,
        )
        axProj_right.grid(True)
        axProj_right.set_title("")
        axProj_right.set_xlabel(f"{mode}(row)")
        
        ### finalize
        plt.tight_layout()
        plt.draw()
        fig.canvas.layout.width = '100%'
        return fig

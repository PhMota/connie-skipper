from xarray import register_dataarray_accessor, register_dataset_accessor

@register_dataarray_accessor("skipper")
class SkipperDataArrayAccessor:
    """
    Access methods for DataArrays for CONNIE skipper CCDs.
    
    Methods and attributes can be accessed through the `.skipper` attribute.
    """
    def __init__(self, da):
        self.da = da
        
    def mad( self, dim = None, axis = None, skipna = None, keep_attrs = True, **kwargs):
        """Reduce this DataArray’s data by applying median absolute deviation along some dimension(s).

        Parameters
        """
        med = abs( self.da - self.da.median(dim, axis, skipna, keep_attrs = True **kwargs) )
        return med.median( dim, axis, skipna, keep_attrs = True, **kwargs )/0.6744897501960817
    
    def center( self, dim = None, axis = None, skipna = None, mode = "median", **kwargs ):
        """subtract global median
        """
        if mode == "median":
            return self.da - self.da.median( dim, axis, skipna, **kwargs )
        elif mode == "mean":
            return self.da - self.da.mean( dim, axis, skipna, **kwargs )
        else:
            raise Exception(f"mode {mode} not implemented")
    
    
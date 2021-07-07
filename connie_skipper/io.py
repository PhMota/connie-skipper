import astropy.io.fits as fits
import xarray as xr

def hdu_to_DataArray( hdu, new_dim = None ):
    """converts a HDU image into a DataArray
    
    Parameters
    ----------
    hdu : HDU object
    
    new_dim : str, optional
        unique key to be picked from the header to be promoted to a dimension
        
    Returns
    -------
    data : DataArray
    """
    dims = ( [new_dim] if new_dim else [] ) + ["row", "col"]
    coords = { new_dim: (new_dim, [hdu.header[new_dim]]) } if new_dim else None
    print( dims, coords )
    return xr.DataArray(
        data = [hdu.data] if new_dim else hdu.data,
        dims = dims,
        coords = coords,
        attrs = { key: (hdu.header[key], hdu.header.comments[key]) for key in hdu.header.keys() if key != new_dim }
    )

def fits_to_DataArray( fpath ):
    """generates a DataArray from a FITS file
    
    Parameters
    ----------
    fpath : str
        the path for a FITS file
    
    Returns
    -------
    data : DataArray
        new DataArray with chid, row, col dimensions

    """
    data = []
    with fits.open( fpath ) as f:
        for hdu in f:
            if hdu.data is not None:
                data.append( hdu_to_DataArray( hdu, new_dim = "chid" ) )
    data = xr.concat( data, dim="chid" )
    return data

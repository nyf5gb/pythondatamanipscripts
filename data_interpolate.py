import os
from tqdm import tqdm

import numpy as np
from scipy import *
import astropy.io.fits as fits
from astropy.wcs import WCS
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator


""""

Variables/strings to change:

directory
ref_file (this one requires you to know your desired file names before running)
filename.startswith
truncation in output_fits and output_filename variables + actual strings


Note: this script takes a good while to finish running. Happy hunting!


"""



directory = "/Users/astravo/Desktop/college_stuff/thesis/hp_tau"
os.chdir(directory) # your directory of fits cubes
csp = 2.99792e8 # speed of light in m/s
nchannels = 132

def find_freq_avg(filename, find_avg):
    hdulist = fits.open(filename) # reads into each fits file

    n_c        = hdulist[0].header['NAXIS3']
    nu         = hdulist[0].header['CRVAL3']
    del_fr     = hdulist[0].header['CDELT3']
    ref_nuind  = hdulist[0].header['CRPIX3']

    hdulist.close()

    frevec = ((np.arange(n_c)+1)-ref_nuind)*del_fr+nu
    del_freq = abs(frevec[0]-frevec[1])

    find_avg.append(del_freq)

def freq_interp(fits_file, output_fits, new_df):

    hdulist = fits.open(fits_file)
    
    data       = hdulist[0].data

    n_c        = hdulist[0].header['NAXIS3']

    nu         = hdulist[0].header['CRVAL3']
    del_fr     = hdulist[0].header['CDELT3']
    ref_nuind  = hdulist[0].header['CRPIX3']
    
    frevec = ((np.arange(n_c)+1)-ref_nuind)*del_fr+nu
    
    # define new uniform frequency grid
    f_min, f_max = np.min(frevec), np.max(frevec)
    new_freq_axis = np.arange(f_min, f_max, new_df)

    # get dimensions
    n_stokes, _, ny, nx = data.shape
    new_shape = (n_stokes, len(new_freq_axis), ny, nx)
    new_data = np.zeros(new_shape)

    # interpolate each spectrum onto the new frequency grid
    for s in range(n_stokes):
        for y in range(ny):
            for x in range(nx):
                spectrum = data[s, :, y, x]
                interp_func = interp1d(frevec, spectrum, kind='linear', bounds_error=False, fill_value=np.nan)
                new_data[s, :, y, x] = interp_func(new_freq_axis)

    # update header with new frequency axis
    new_header = hdulist[0].header.copy()
    new_header['CRVAL3'] = new_freq_axis[0]  # first frequency value
    new_header['CDELT3'] = new_df  # frequency step
    new_header['CRPIX3'] = 1  # reference pixel
    new_header['NAXIS3'] = len(new_freq_axis)  # new number of frequency channels
    new_header['CTYPE3'] = 'FREQ'  # update axis type to frequency

    output_fits = f"interpolated_{fits_file[:-17]}.fits"

    fits.writeto(output_fits, new_data, new_header, overwrite=True)
    print(f"Frequency interpolated -> {output_fits}")

def pixel_interp(fits_file, ref_wcs, ref_shape, x_ref, y_ref):
    with fits.open(fits_file) as hdulist:
        data = hdulist[0].data
        wcs = WCS(hdulist[0].header, naxis=2)

        # generate source cube pixel grid
        n_stokes, freq, ny, nx = data.shape

        # generate source spatial grid   
        y_src = np.arange(ny)
        x_src = np.arange(nx)
        
        ra_ref, dec_ref = ref_wcs.pixel_to_world_values(x_ref, y_ref)
        x_src_coords, y_src_coords = wcs.world_to_pixel_values(ra_ref, dec_ref)
        interp_coords = np.stack([y_src_coords, x_src_coords], axis=-1)
        out_cube = np.zeros((n_stokes, freq, *ref_shape), dtype=np.float32)

        # interpolate each 2D plane
        for s in range(n_stokes):
            for f in tqdm(range(freq), desc=f"{os.path.basename(fits_file)}"):
                plane = data[s, f, :, :]
                interpolator = RegularGridInterpolator((y_src, x_src), plane, bounds_error=False, fill_value=np.nan)
                out_cube[s, f] = interpolator(interp_coords).reshape(ref_shape)


        # save new file
        output_filename = "new_" + fits_file
        hdu = fits.PrimaryHDU(data=out_cube, header=ref_header)
        hdu.writeto(output_filename, overwrite=True)
        
    
        print(f"Reprojected -> {output_filename}")
        
        
        
        
        
        
        

#########
# find the frequency average to interpolate on
#########

find_avg = []

for filename in os.listdir(directory):
    if filename.endswith(".fits") and filename.startswith("HP_Tau"):
        find_freq_avg(filename, find_avg)
freq_avg = sum(find_avg) / len(find_avg)
#print(f"average frequency = {freq_avg}")

#######
# creates list of filenames to interpolate
#######

fitsfiles = []
for filename in os.listdir(directory):
    if filename.startswith("HP_Tau"): 
        if filename.endswith(".fits"):
            fitsfiles.append(filename)

#######
# interpolates velocity slices
#######

for fits_file in fitsfiles:
    freq_interp(fits_file, freq_avg)
print("All cubes have been velocity-interpolated.")


#######
# creates list of velocity-interpolated files
#######

interp_fitsfiles = []
for filename in os.listdir(directory):

    if filename.startswith("interpolated_HP_Tau"): 
        if filename.endswith(".fits"):
            with fits.open(filename) as hdulist:
                data = hdulist[0].data
            interp_fitsfiles.append(filename)


#######
# creates reference frame to interpolate pixels from a file
#######

ref_file = "interpolated_HP_Tau_CO_230.538GHz.fits"

ref_cube = fits.open(ref_file)

with fits.open(ref_file) as hdulist:
    data = hdulist[0].data
    n_stokes, freq, ny, nx = data.shape

ref_header = ref_cube[0].header
ref_wcs = WCS(ref_header, naxis=2)
ref_shape = ref_cube[0].data.squeeze().shape[-2:] 

y_ref, x_ref = np.indices(ref_shape)
ra_ref, dec_ref = ref_wcs.pixel_to_world_values(x_ref, y_ref)
ref_cube.close()



#######
# pixel interpolates velocity-interpolated files
#######

for fits_file in interp_fitsfiles:
    pixel_interp(fits_file, ref_wcs, ref_shape,x_ref, y_ref)

print("All cubes have been pixel-interpolated.")


print("Done!")

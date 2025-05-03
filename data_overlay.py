import os,sys
import matplotlib
import matplotlib.pyplot as plt
import pdb as pdb
from matplotlib import rcParams
import numpy as N
import matplotlib.gridspec as gridspec
from scipy import *
from matplotlib import rc
rc('text',usetex=False)
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import astropy.io.fits as pyfits
from matplotlib.patches import Ellipse


""""

Variables/strings to change:

directory
outputname
svel (used to center the plot on a certain velocity)
fig.suptitle

arrays:

min_array
max_array
files_overlay (the files you want to overlay. the first file in this array is the base)
colorseries (colors to plot the rest of the files in. the first file is plotted viridis)
--> these are inputted as variables at the end



Note: this script takes a good while to finish running depending on the nchannel and Nlevels.

Some of this code is Frankenstein'd from the lovely UVA alumna Claire Thilenius' CASA guide scripts.

"""


directory = "/Users/astravo/Desktop/college_stuff/thesis/hp_tau"
os.chdir(directory)

def plotall(img, min_array, max_array, svel, cmap, window, fitsdata, continuumfits, colors):
    
    ############################################################
    def formax(ax,majx,majy,minx,miny):
        majorxLocator   = MultipleLocator(majx)
        majoryLocator   = MultipleLocator(majy)
        majorxFormatter = FormatStrFormatter('%4.0f')
        majoryFormatter = FormatStrFormatter('%4.0f')
        minorxLocator   = MultipleLocator(minx)
        minoryLocator   = MultipleLocator(miny)
        ax.xaxis.set_major_locator(majorxLocator)
        ax.yaxis.set_major_locator(majoryLocator)
        ax.xaxis.set_major_formatter(majorxFormatter)
        ax.yaxis.set_major_formatter(majoryFormatter)
        ax.xaxis.set_minor_locator(minorxLocator)
        ax.yaxis.set_minor_locator(minoryLocator)
        return

    ############################################################

    def readingrainbowCASA(img,ra_obj,de_obj,nchannels,sourcevel,flip=False,fitsdata=True):

        # ra_obj = right ascenscion in degrees. 11h 40m 30s = ((30/60+40)/60+11)*360/24 RA in hours -> degrees (float)
        # de_obj in degrees (float)
        # sourcevel in km/s
        # img = is fits file name including directory
        hdulist = pyfits.open(img)
        data = hdulist[0].data

        ra0=hdulist[0].header['CRVAL1']
        dra=hdulist[0].header['CDELT1']
        de0=hdulist[0].header['CRVAL2']
        dde=hdulist[0].header['CDELT2']
        pra=hdulist[0].header['CRPIX1']
        pde=hdulist[0].header['CRPIX2']
        nu0=hdulist[0].header['RESTFRQ']
        nra = hdulist[0].header['NAXIS1']
        nde = hdulist[0].header['NAXIS2']
        nc = hdulist[0].header['NAXIS3']

        nu=hdulist[0].header['CRVAL3']
        dfr=hdulist[0].header['CDELT3']
        refnuind = hdulist[0].header['CRPIX3']
        
        if fitsdata == True:
            bmaj = hdulist[0].header['BMAJ']
            bmin = hdulist[0].header['BMIN']
            bpa = hdulist[0].header['BPA']
        else: 
            bmaj = 1.0
            bmin = 1.0
            bpa = 1.0

        
        hdulist.close()

        ravec = (N.arange(nra)+1-pra)*dra+ra0 # in degrees
        devec = (N.arange(nde)[::-1]+1-pde)*dde+de0 # in degrees

        # Convert to relative arcsec:
        ravec_rel = (ravec-ra_obj)*3600.0
        devec_rel = (de_obj-devec)*3600.0
        raarr_out,dearr_out = N.meshgrid(ravec_rel,devec_rel)
        data = data[0,:,:,:]  # Only one stokes parameter
        
        if flip: data = data[::-1,:,:]
        data[N.isnan(data)] = 0.0 # set masked disk to zero (masked b/c has low sigma)
        
        if nc > 1:
            frevec = ((N.arange(nc)+1)-refnuind)*dfr+nu
            velvec = (nu0-frevec)/nu0 * 2.99792e5 # km/s
            velvec = velvec[::-1]

            centerchan = N.argmin(N.abs(velvec-sourcevel))
            cutch = N.round((nc - nchannels) / 2.0)
            datarr = data[int(centerchan-(nchannels-1)/2):int(centerchan-(nchannels-1)/2+nchannels)]
            velvec = velvec[int(centerchan-(nchannels-1)/2):int(centerchan-(nchannels-1)/2+nchannels)]
            velvec = velvec - sourcevel
        else:
            velvec = [1]
            datarr = data

        return datarr,raarr_out,dearr_out,bmaj,bmin,bpa,velvec

    ############################################################
    
    def RADEC(img):
        #Find RA and Dec for Disk based off Header file
        hdulist = pyfits.open(img)
        ra0=hdulist[0].header['CRVAL1']
        de0=hdulist[0].header['CRVAL2']
        return ra0, de0



    ############################################################

    def plotchannels(fts, colormap, colorlab, coltick, bw, pad, barsh,\
                  Nlevels, nchannels, min_array, max_array, window,\
                  img_array, svel, fitsdata, continuumfits):
                  
        flip = False
        
        fig = plt.figure(1,figsize=(9,8)) # Initializes the size of the plot. Subplots are roughly square in this size
        fig.suptitle(f"Channel Maps of CO, 13CO, CS, and HF 1/3 HCN with Continuum")
        fig.supxlabel("\u0394RA''")
        fig.supylabel("\u0394Dec''")
        
        
        ncol = 5 #Feel free to change this to fit the nchannels better
        nrow = int(nchannels/ncol)
        gs0 = gridspec.GridSpec(1,1,left=0.09, right=0.85,top=0.92,wspace=0.3,hspace=0.3)
        gs01 = gridspec.GridSpecFromSubplotSpec(nrow, ncol, subplot_spec=gs0[0])

        
        
        add_details = (nrow*ncol)-1 # subplot to add details in later

        counter = 0 # initializing counter
        
        
        # Reading in continuum file
        print("Reading continuum...")
        rapo, depo = RADEC(continuumfits)
        cont_data_arr,cont_raarr,cont_dearr,bmaj,bmin,bpa,velvec \
            = readingrainbowCASA(continuumfits,rapo,depo,1,svel,flip,fitsdata)
        cont_pltarr = cont_data_arr[0,:,:]
        print("Continuum read.")

        # This goes through each image in the array and overplots
        for img in img_array:
            print(f"Reading image {counter + 1}...")
            # Reads through each file for data
            rapo, depo = RADEC(img)
            data_arr,raarr,dearr,bmaj,bmin,bpa,velvec \
                = readingrainbowCASA(img,rapo,depo,nchannels,svel,flip,fitsdata)
            
            
            contmin = min_array[counter] # This array is an input at the bottom.
            contmax = max_array[counter] # This array is an input at the bottom.

            contvals = N.append(N.arange(contmin, contmax, abs(contmax - contmin)/(Nlevels-1)), contmax)
            contlab = N.append(N.arange(contmin, contmax, abs(contmax - contmin)/(5-1)), contmax)

            print("Plotting...")
            
            for nc in range(nchannels):
                ax = plt.subplot(gs01[nc])
                pltarr = data_arr[nc,:,:]  # slice in velocity space

                if counter == 0:
                    alpha = 1 # Plots the first file opaque
                    colormap = cmap
                    plt.contourf(raarr, dearr, pltarr, levels=contvals, cmap=colormap, extend='both', alpha = alpha)
                    
                else:
                    alpha = 0.8 # Plots the rest of the files as a different colormap and slightly transparent on top of the first file
                    colormap = colors[counter]
                    pltarr[pltarr < contmin]  = N.nan #sets values < contmin to NaN
                    plt.contourf(raarr, dearr, pltarr, levels=contvals, cmap=colormap, extend='both', alpha = alpha)
                
                # Plots the continuum on each channel
                alpha = 0.4
                colormap = 'twilight'
                cont_contmin = 0.01
                cont_contmax = 0.05
                levels = N.append(N.arange(cont_contmin, cont_contmax, abs(cont_contmax - cont_contmin)/(Nlevels-1)), cont_contmax)

                plt.contour(cont_raarr, cont_dearr, cont_pltarr, levels=levels, cmap=colormap, extend='both', alpha = alpha, linewidths = 1)
                    

                plt.annotate("%3.2f km/s"%(velvec[nc]), xy=(0.92,0.03), xycoords='axes fraction', color='white', horizontalalignment='right', verticalalignment='bottom', fontweight='normal', fontsize=fts-2) #annotates the velocity +/- the source
                formax(ax,6,6,3,3) # this sets number of ticks, major x tick, major y tick, minor x tick, minor y tick
                
                if nc == add_details and fitsdata == True:
                
                    ellpo = window-1.0 # ellipse position -- center of where the beam will go
                    ells = Ellipse(xy=[ellpo,-ellpo], width=bmin*3600.0, height=bmaj*3600.0, angle=bpa*-1,zorder=300)
                    ells.set_facecolor('w')
                    ells.set_edgecolor('w')
                    ax.add_artist(ells)
                    ax.set
                    
                else:
                    plt.setp(ax.get_xticklabels(), visible=False)
                    plt.setp(ax.get_yticklabels(), visible=False)
                

                ax.set_xlim(window,-1*window)
                ax.set_ylim(-1*window,window)

            
            counter += 1 # Moves onto next file
            print(f"Image {counter} plotted.")
            
        # Put the colorbar on the last ax
#        pt = ax.get_position().bounds
#        cax = fig.add_axes([pt[0]+pt[2]+pad-bw/2, pt[1]+pt[3]*(1.0-barsh)/2.0, bw, pt[3]*barsh])
#        cbar = plt.colorbar(CS2, cax=cax, ticks=contlab) #,format=ticker.FuncFormatter(fmt))
#            ax.set_ylabel(r'Jy beam$^{-1}$',fontsize=fts-2,color='k',labelpad=15.0,rotation=270)


        return


    #########################################
    #########################################
    #########################################

    # How the font looks
    fts = 11
    rcParams['font.size'] = fts
    rcParams['font.family'] = 'serif'
    rcParams['font.weight'] = 'normal'


    # https://matplotlib.org/stable/users/explain/colors/colormaps.html
    colormap = plt.cm.get_cmap(cmap)
    colorlab= 'k'
    coltick = 'k'

    # Sets position of the colorbar
    
    bw = 0.015
    pad = 0.013
    barsh = 1.0


    # larger numbers = longer runtime
    
    Nlevels = 15 # Number of colors to use in color map
    nchannels = 25 # number of channels to plot around center of line, +/- nchannels/2 ; I recommend square multiples (this might look strange otherwise)

    outputname = "testimage.png"


    plotchannels(fts, colormap, colorlab, coltick, bw, pad, barsh,\
              Nlevels, nchannels, min_array, max_array, window,\
              img, svel, fitsdata, continuumfits)

    
    print(":3 done! Saving and showing plot...")
    
    
    plt.savefig(outputname, format = 'png')
    plt.show()
    return 



# inputs for the plotall function

min_array = [0.02,0.015,0.01,0.015,0.015] # these values are pick-and-choose based on maxs and mins.
max_array = [0.15,0.032,0.03,0.05,0.05]

files_overlay = ["new_interpolated_HP_Tau_CO_230.538GHz.fits","new_interpolated_HP_Tau_13CO_220.399GHz.fits", "new_interpolated_HP_Tau_CS_293.912GHz.fits" ,"new_interpolated_HP_Tau_HCN_hf2_265.886GHz.fits", "new_interpolated_HP_Tau_HCN_hf3_265.889GHz.fits"]
#"new_interpolated_HP_Tau_HCN_hf1_265.885GHz.fits"
svel = 8.1 # this is the value to center on. see: velvec = velvec - sourcevel in the readingrainbowCASA function
# change this to center the interesting data

colorseries = ["Greys","plasma","magma","gnuplot","inferno","cool"]

####




# call the megafunction!
plotall(img = files_overlay , min_array = min_array , max_array = max_array, svel = svel, cmap = "viridis", window = 5.0, fitsdata = True, continuumfits = "HP_Tau_continuum.fits",colors = colorseries)







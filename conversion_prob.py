# This code calculates the axion to photo conversion probability P given a magnetic field and plasma density model.
# All dimensionful quantities are expressed either in eV or eV^-1, unless otherwise specified
import sys
import numpy as np
from pytictoc import TicToc
from scipy.interpolate import interp1d, RegularGridInterpolator

cm = 50684.237202 #  eV^-1
km = 1e5 * cm #  eV^-1
tesla = 195.3 # eV^2
gauss = 1e-4 * tesla # eV^2
Rsun =  69.634e9 * cm #  eV^-1
Rsunkm = 6.9634e5 # km
amu = 1.660538921e-24 # grams
degree = np.pi/180

plasma_prefactor = 4.23674e-4 # eV. Is equal to sqrt(e^2 eV^3 / me). To be multiplied by sqrt(n) in eV^3

helium_fraction  = 10 ** (10.78 - 12) # ratio between He and N number density
step_integration = 1
Zdic = {
 1 : 'H',
 2 : 'He',
 3 : 'Li',
 4 : 'Be',
 5 : 'B',
 6 : 'C',
 7 : 'N',
 8 : 'O',
 9 : 'F',
10 : 'Ne',
11 : 'Na',
12 : 'Mg',
13 : 'Al',
14 : 'Si',
15 : 'P',
16 : 'S',
17 : 'Cl',
18 : 'Ar',
19 : 'K',
20 : 'Ca',
21 : 'Sc',
22 : 'Ti',
23 : 'V',
24 : 'Cr',
25 : 'Mn',
26 : 'Fe',
27 : 'Co',
28 : 'Ni',
29 : 'Cu',
30 : 'Zn'
}

# low resolution vector of altitude above the photosphere for calculating the phase of the conversion amplitude
hmin = 0.9808322598 * cm
hmax = 29 * Rsun
hstmp = np.logspace(np.log10(hmin), np.log10(hmax), num=10001, endpoint=True)  # eV^-1
h_transition = 2100 * km # transition from chromosphere to corona

sigma_thompson = 6.65e-25 * cm**2 # eV^-2

theta_offset = 7.04 * degree # Takes into account inclination of the solar rotation axis wrt the ecliptic at the day of observation. It has positive sign because theta grows away from north pole. Center of the disk is south of the solar equator on Feb 2020


timer = TicToc ()

##############################################################################
npz_dir = 'npzs/'
txt_dir = 'txts/'


###########################################################################
#                       Plasma  and absoption                             #
###########################################################################

def get_H_He_dens_func():
   data = np.loadtxt(txt_dir + 'Hdens_geo.txt', skiprows=1)
   h = data[:, 0] * km  # eV^-1
   dens = data[:, 1] * (1 + 2 * helium_fraction) / cm ** 3  # eV^3. Takes into account 2 electrons for He atoms
   h, idxs = np.unique(h, return_index=True)
   dens = dens[idxs]
   
   idx = np.argsort(h)
   h = h[idx]
   dens = dens[idx]
   
   return interp1d(h,  dens, kind=1, bounds_error=False, fill_value=(dens[0], dens[-1])), h

def get_edens_free_func():
   
   data = np.loadtxt(txt_dir + 'edens_geo.txt', skiprows=4)
   edens_heights = data[:, 0] * km  # eV^-1
   edens_values = data[:, 1] / cm ** 3  # eV^3
   
   edens_heights, idxs = np.unique(edens_heights, return_index=True)
   edens_values = edens_values[idxs]
   
   idx = np.argsort(edens_heights)
   edens_heights = edens_heights[idx]
   edens_values = edens_values[idx]
   
   H_He_dens_func, h = get_H_He_dens_func()
   i = np.where(h > edens_heights[-1])[0][0]
   edens_heights = np.concatenate((edens_heights, h[i:]))
   edens_values = np.concatenate((edens_values, H_He_dens_func(h[i:])))
   
   return interp1d(edens_heights, edens_values, bounds_error=False,
                   fill_value=(edens_values[0], edens_values[-1])), edens_heights

def get_plasma_freq():
   edens_free_func, edens_heights = get_edens_free_func()
   hdens_func, h = get_H_He_dens_func()
   
   i = find_nearest(edens_heights, h_transition)
   n_chromosphere = edens_free_func(edens_heights[:i]) + hdens_func(edens_heights[:i])
   n_corona = edens_free_func(edens_heights[i:])
   n_tot = np.concatenate((n_chromosphere, n_corona))
   
   omegap = plasma_prefactor * np.sqrt(n_tot)  # eV
   omegap_func = interp1d(edens_heights, omegap, kind=1, bounds_error=False, fill_value=(omegap[0], omegap[-1]))  # eV
   
   return omegap_func

def getGamma_XCOM(hs, omega):  # omega must be in eV, hs in eV^-1
   # \Gamma_{TOT} = n_H \sum_Z (n_Z/n_H) \sigma_Z
   # contribution_tot = \sum_Z (n_Z/n_H) \sigma_Z
   
   # Load density profile
   dens, h = get_H_He_dens_func()
   dens = dens(h) / (1 + 2 * helium_fraction)  # divide to get back the density of hydrogen only
   
   # Load coronal abundances
   data = np.loadtxt(txt_dir + 'sun_coronal_2012_schmelz_ext.abund')
   Zs = data[:, 0].astype(int)
   abund = data[:, 1]
   ratio = 10 ** (abund - 12)  # ratio of the density of element to that of H (n_Z/n_H)
   
   # Load atomic weights
   data = np.loadtxt(txt_dir + 'atomic_weight.txt', skiprows=1)
   weight = data[:, 1] * amu  # grams. Weight of one atom.
   
   contribution_tot = 0
   for Z in Zs:
      data = np.loadtxt(txt_dir + 'xcom/' + Zdic[Z] + '.txt')
      energies = data[:, 0] * 1e6  # eV
      sigma_tot_xcom = data[:, 4]  # cm^2 / g
      sigma_tot_func = interp1d(energies, sigma_tot_xcom)
      
      sigma_tot = sigma_tot_func(omega) * weight[Z - 1] * cm ** 2  # eV^-2
      
      contribution = sigma_tot * ratio[int(Z) - 1]
      contribution_tot += contribution
   
   Gamma_func = interp1d(h, dens * contribution_tot, kind=1, bounds_error=False,
                         fill_value=(dens[0] * contribution_tot, dens[-1] * contribution_tot))
   
   return Gamma_func(hs)

def getGamma_thompson_free(hs):
   edens_free_func, edens_heights = get_edens_free_func()
   return sigma_thompson * edens_free_func(hs)  # eV

def getGamma_tot(hs, omega):
   Gamma = getGamma_XCOM(hs, omega)
   Gamma_free = getGamma_thompson_free(hs)
   
   Gamma_tot = Gamma_free + Gamma
   return Gamma_tot

###########################################################################
#                       Magnetic field                                    #
###########################################################################
def get_B_predictive_science(year):
   
   year = str(year)
   
   data = np.load(npz_dir + 'br'+year+'.npz')  # Radial component
   Br = data['b']  # Gauss
   rr = data['r']  # Rsun
   thetar = data['theta']  # rad
   phir = data['phi']  # rad
   Br_func = RegularGridInterpolator((phir, thetar, rr), Br)
   
   
   data = np.load(npz_dir + 'bt'+year+'.npz')  # theta component
   Btheta = data['b']  # Gauss
   rtheta = data['r']  # Rsun
   thetatheta = data['theta']  # rad
   phitheta = data['phi']  # rad
   Btheta_func = RegularGridInterpolator((phitheta, thetatheta, rtheta), Btheta)

   data = np.load(npz_dir + 'bp'+year+'.npz')  # phi component
   Bphi = data['b']  # Gauss
   rphi = data['r']  # Rsun
   thetaphi = data['theta']  # rad
   phiphi = data['phi']  # rad

   Bphi_func = RegularGridInterpolator((phiphi, thetaphi, rphi), Bphi)

   Rlim = 0.1  # Rsun
   N = 30
   Nx = 2000
   Ncenters = 120

   xs = np.logspace(0, np.log10(30), num=Nx)
   ys = np.linspace(-Rlim, Rlim, num=N)
   zs = np.linspace(-Rlim, Rlim, num=N)
   
   phi_centers = np.linspace(0, 2 * np.pi, num=Ncenters)
   Brho = np.empty([Ncenters, Nx])
   
   # helper array to call the interpolating functions.
   arr = np.empty([Nx, 3])
   arr[:, 2] = xs
   
   # Assume the line of sight is the direction x.
   # Initially this is the direction theta=0 and phi=0
   # Then, we rotate this direction about the axis by phi_center to make an average
   for k, phi_center in enumerate(phi_centers):
      # Components of the magnetic field in the plane perpendicular to x (los direction)
      By = np.empty([Nx, N, N])  # x, y, z
      Bz = np.empty([Nx, N, N])  # x, y, z
      # In the perpendicular plane we consider a circle of radius Rlim*Rsun
      for i, y in enumerate(ys):
         for j, z in enumerate(zs):
            if y ** 2 + z ** 2 > Rlim ** 2:
               By[:, i, j] = np.nan
               Bz[:, i, j] = np.nan
               continue
            
            theta = theta_offset +  np.arctan2(np.sqrt(xs ** 2 + y ** 2), z)
            phi = (np.arctan2(y, xs) + phi_center) % (2 * np.pi)
            arr[:, 0] = phi
            arr[:, 1] = theta

            By[:, i, j] = np.sin(theta) * np.sin(phi) * Br_func(arr
                         ) + np.cos(theta) * np.sin(phi) * Btheta_func(arr
                          ) + np.cos(phi) * Bphi_func(arr)
            Bz[:, i, j] = np.cos(theta) * Br_func(arr
                              ) - np.sin(theta) * Btheta_func(arr)
      
      Brho[k] = np.nanmean(np.sqrt(By ** 2 + Bz ** 2), axis=(1, 2))
   np.savez(npz_dir + 'b_rho'+year+'_thetaoffset_%.3f.npz' % theta_offset, B=Brho, xs=xs, phi_centers=phi_centers, theta_offset=theta_offset)

   return Brho, xs

def merge_B_predictive_rempel(year):
   
   # Load B at low altitude from Rempel's model
   data = np.loadtxt(txt_dir + 'Bperp_rempel.txt')
   Rs = data[:, 0] * 1e3 / Rsunkm  # Rsun. Height from photosphere
   idx = np.where((Rs >= 0) & (Rs <= 400 / Rsunkm))
   Rs = Rs[idx]
   Bs = data[:, 1][idx]  # gauss
   
   # Load B in the corona from predictive science simulation
   data = np.load(npz_dir + 'b_rho'+year+'_thetaoffset_%.3f.npz' % theta_offset)
   Brhos = data['B']
   # Brho = np.mean(Brhos, axis=0)  # gauss
   Brho = np.median(Brhos, axis=0)  # gauss

   
   xs = data['xs'] - 1  # Rsun. Height from photosphere
   phi_centers = data['phi_centers']
   # Merge Rempel's B with the predictive science one by interpolating with a straight line in log space between the end of Javier's B and the predictive science B at 0.1 Rsun above photosphere
   x_cut_ps = 0.1  # Rsun
   i = find_nearest(xs, x_cut_ps)
   # Brho = Brho * 0.141 / Brho[i]
   Brho = Brho * 0.210292 / Brho[i]
   
   
   xs_intermediate = np.linspace(np.log10(Rs[-1]), np.log10(x_cut_ps), num=1000)
   B_intermediate_func = interp1d([np.log10(Rs[-1]), np.log10(x_cut_ps)], [np.log10(Bs[-1]), np.log10(Brho[i])],
                                  bounds_error=False, fill_value='extrapolate')
   
   
   # Everything together
   xs_tot = np.concatenate((Rs, 10 ** xs_intermediate, xs[i:]))
   Bperp = np.concatenate((Bs, 10 ** B_intermediate_func(xs_intermediate), Brho[i:]))
   
   Bperp_func = interp1d(xs_tot * Rsun, Bperp * gauss, bounds_error=False, fill_value='extrapolate')
   
   return Bperp_func, Rs[-1], xs_intermediate[-1]

###########################################################################
#                       Helper functions                                  #
###########################################################################
def integrate_steps(xs, integrand, step, type):
   big_xs = xs[::step]
   if type == 'complex':
      result = np.empty_like(big_xs, dtype='complex')
   else:
      result = np.empty_like(big_xs)
   result[0] = 0
   for i, x in enumerate(big_xs):
      if i == 0:
         continue
      imin = step * (i - 1)
      imax = step * i + 1
      result[i] = result[i - 1] + np.trapz(integrand[imin: imax], x=xs[imin: imax])
   
   return result, big_xs

def find_nearest(array, value):
   array = np.asarray(array)
   idx = (np.abs(array - value)).argmin()
   return idx

def create_adaptive_heights(big_hs, bp_idx, omega, ma, omegap_func):
	# Function to create an array of point at which to evaluate the conversion probabilty

   # We start our from the lower end of region 2)
   prec = big_hs[bp_idx]  # big_hs[0] #
   
   # we don't know beforehand how many altitude points we need, so we append points to arr
   arr = [prec]
   
   # We switch to fixed spacing between points for altitudes > finalh.
   # Notice the value of finalh must be tuned based on the B model. For what I'm using 
   # now this is ok, but for B that extends further away, we may need a larger finalh,
   # because P oscillates again after the resonace location mgamma=ma
   i_resonance = find_nearest(omegap_func(big_hs), ma)
   h_resonance = big_hs[i_resonance]

   finalh = big_hs[-1]

   i = 1
   while prec < finalh:
      prec = arr[i - 1]
      # print(prec/km)
   
      # Choose increase in altitude based on the derivative of phi
      phiprime = np.abs((omegap_func(prec) ** 2 - ma ** 2) / (2 * omega))
      wavelength = 2 * np.pi / phiprime
      if 0.9 * h_resonance <= prec <= 1.1 * h_resonance:
         factor2 = 0.005
         # print(prec / Rsun, h_resonance / Rsun, factor2)
      else:
         factor2 = 0.05
      increase = factor2 * np.minimum(wavelength, prec)
      new_h = prec + increase
      if new_h > h_resonance + 50 * wavelength:
         break
      arr.append(new_h)
      i += 1

   hs = np.array(arr)

   # i_avg in the index from which we average over the oscillations
   if ma > 1e-5 and hs[-1] < big_hs[-1]:
      i_avg = find_nearest(hs, h_resonance + 40 * wavelength)
   else:
      i_avg = len(hs)
   
   return hs, i_avg

###########################################################################
#            Get exponents of P_{a\gamma} expression                      #
###########################################################################
def get_phis_func(ma, omega, omegap_func): # everything in eV

   integrand = (omegap_func(hstmp) ** 2 - ma ** 2) / (2 * omega)
   phis, big_hs = integrate_steps(hstmp, integrand, step_integration, '')
   
   phis[0] = 0 
   phis_func = interp1d(big_hs, phis, bounds_error=False, fill_value="extrapolate")

   # There are two regions:
   # 1) increases with an approximately constant non-zero slope
   # 2) stays approximately constant. This corresponds to a drop in plasma density.

   line = (omegap_func(big_hs[0]) ** 2 - ma ** 2) / (2 * omega) * big_hs
   i_resonance = find_nearest(omegap_func(big_hs), ma)
   bp_idx = find_nearest(line, phis[i_resonance])  # find intersection between regions 1) and 2)

   return phis_func, bp_idx, phis, big_hs

def get_psis_func(omega):  # everything in eV
   
   integrand = 0.5 * getGamma_tot(hstmp, omega)
   phis, big_hs = integrate_steps(hstmp, integrand, step_integration, '')
   
   psis, _ = integrate_steps(hstmp, integrand, step_integration, '')
   psis_func = interp1d(big_hs, psis, bounds_error=False, fill_value="extrapolate")

   return psis_func

###########################################################################
#                       Instrument and data                               #
###########################################################################

def get_data():
   data = np.loadtxt(txt_dir + 'src_bkg_spectra.txt', skiprows=1)
   energy = data[:, 0] * 1e3  # eV
   energy = energy
   tracking = data[:, 1]
   background = data[:, 4]
   return energy, tracking, background


def get_data_quiet():
   data = np.loadtxt(txt_dir + 'save_spectrum.txt', skiprows=2)
   energy = 0.5 * (data[:, 0] + data[:, 1]) * 1e3  # eV
   src_counts = data[:, 2]
   bkg_counts = data[:, 3]
   src_cm2_eff = data[:, 4]
   bkg_cm2_eff = data[:, 5]
   
   data = np.loadtxt(txt_dir + 'save_rmf.txt', skiprows=0)
   redistribution_matrix = data[:, 2:]
   
   return energy, src_counts, bkg_counts, src_cm2_eff, bkg_cm2_eff, redistribution_matrix

###########################################################################
#                       Conversion probability                            #
###########################################################################
def get_conversion_probability(omega, ma):
   
   # Get the phase of the convesion amplitude phi = \int dz (mgamma^2 - ma^2) / (2 omega)
   # The general behavior of the phase is:
   # 1) grows ~ linearly where the plasma density is high and ~const. In this region the conversion propability is suppressed. It doesn't grow at all and it oscillates very fast (see also Carlson and Tseng (1995)). So to calculate P in this region is going to be useless and computationally expensive. We are going to exclude this region. The variable bp_idx is the index in the array of altitudes big_hs, corresponding to the location where phi starts to settle to a constant value
   # 2) where the plasma density starts to decrease, the phase becames ~ constant with altitude (decreases slowly). In this region P can grow.
   B_func, Rslim, xs_intermediatelim = merge_B_predictive_rempel(2019)
   omegap_func = get_plasma_freq()  # eV
   
   phis_func, bp_idx, phis, big_hs = get_phis_func(ma, omega, omegap_func) # phase
   # With this method we prepare the array of altitudes to use for integrating the conversion amplitude.
   hs, i_avg = create_adaptive_heights(big_hs, bp_idx, omega, ma, omegap_func)
   # print('hs = ', hs/Rsun)
   
   # Absorption factor.\psi = \int dz Gamma, where gamma is cross-section times number density for the various element
   psis_func = get_psis_func(omega) # \psi = \int dz \Gamma
   
   integrand2 = B_func(hs) * np.exp(1j * phis_func(hs)) * np.exp(psis_func(hs) - psis_func(hs)[-1])
   
   if i_avg != len(hs):
      hs_avg = hs[i_avg:]
      amplitude = np.trapz(integrand2[:i_avg], x=hs[:i_avg])
      amplitude_extra, Ls = integrate_steps(hs_avg, integrand2[i_avg:], 1, 'complex')
      amplitudestot = amplitude + np.mean(amplitude_extra)
   else:
      amplitudestot = np.trapz(integrand2, x=hs)

   P_over_g10_squared = 0.25 * np.abs(amplitudestot)**2 * 1e-38 # the last factor assumes an axion-photon coupling g=10^-10 GeV^-1. We have expressed B in eV^2 and the altitude in eV^-1. So amplitude has units of eV. When we multiply g^2*amplitude^2, we get then the 10^-38 = (10^(-10-9))^2 factor
      
   return P_over_g10_squared

def convert_axions():

   DeltaE = 40 # eV
   E_low = 1820  # eV
   E_high = 2000 #15e3 # eV
   energy = np.arange(E_low, E_high, DeltaE) # eV
   Nomega = len(energy)

   Nm = 3#105
   mas = np.logspace(-8, np.log10(3e-3), num=Nm)# eV
   
   Ps = np.empty([Nomega, len(mas)])
   Ls = np.empty([Nomega, len(mas)], dtype=object)
   
   for i, omega in enumerate(energy):
      print('\n\n', i, omega)
      timer.tic()
      for j, ma in enumerate(mas):
         print(j, end=' ')
         Ps[i,j] = get_conversion_probability(omega, ma)
      timer.toc()

   np.savez(npz_dir+ 'probabilities.npz', Ps=Ps, omegas=energy, mas=mas)



################################################################################
################################################################################
################################################################################
if __name__ == '__main__':

   get_B_predictive_science(2019)
   convert_axions()


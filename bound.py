import numpy as np
from pytictoc import TicToc
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.special import erfc
from scipy import integrate

tint_quiet = 46614.859 / 2 # seconds. Integration time.
DeltaE = 40 # eV

timer = TicToc ()

##############################################################################
npz_dir = 'npzs/'
txt_dir = 'txts/'

###########################################################################
#                       Instrument and data                                  #
###########################################################################

def get_data_quiet():
   
   data = np.loadtxt(txt_dir + 'save_spectrum.txt', skiprows=2)
   energy = 0.5 * (data[:, 0] + data[:, 1]) * 1e3  # eV
   src_counts = data[:, 2]
   bkg_counts = data[:, 3]
   src_cm2_eff = data[:, 4]
   bkg_cm2_eff = data[:, 5]
   
   data = np.loadtxt(txt_dir + 'save_rmf.txt', skiprows=0)
   redistribution_matrix = data[:, 2:]
   
   # print(energy.shape)
   # print(redistribution_matrix.shape)
   
   return energy, src_counts, bkg_counts, src_cm2_eff, bkg_cm2_eff, redistribution_matrix

def get_primakof_flux_quiet():
   
   Rmax_signal = 0.1
   Rmin_annulus = 0.15
   Rmax_annulus = 0.3
   
   data = np.loadtxt(txt_dir + 'fluxy.txt')
   energy = data[:, 0][::100] * 1e3  # eV
   radius = data[:, 1][:100]  # un units of Rsun
   flux = data[:, 2] * 1e10  # Flux for g=10^-10 GeV^-1.
   
   flux = np.reshape(flux, (-1, 100))
   
   idx = np.where(radius >= Rmax_signal)[0][0] + 1
   area_10 = 2 * np.pi * np.trapz(radius[:idx], x=radius[:idx])
   flux_10 = 2 * np.pi * np.trapz((flux * radius)[:, :idx], x=radius[:idx], axis=1) * 1e-3 # per eV
   
   idx = np.where((radius >= Rmin_annulus) & (radius <= Rmax_annulus))[0]
   area_1530 = 2 * np.pi * np.trapz(radius[idx], x=radius[idx]) * 285/360
   flux_1530 = 2 * np.pi * np.trapz((flux * radius)[:, idx], x=radius[idx], axis=1) * 1e-3 * 285/360 # per eV
   

   return interp1d(energy, flux_10), interp1d(energy, flux_1530), area_10, area_1530 # flux in N / cm^2 / s / eV


################################################################################################
#                                   Make bounds                                                #
################################################################################################

def get_Ngammais_quiet(energy, probname):
   
   # energy -> energies at which I want the expected photon count Ngammais
   # omegas -> energies at which I have the conversion probability
   # energy_data -> energy at which I have the data. This would typically contain more values than omega
   
   energy_data, src_counts, bkg_counts, src_cm2_eff, bkg_cm2_eff, redistribution_matrix = get_data_quiet()
   axion_flux_10_func, axion_flux_1530_func, area_10, area_1530 = get_primakof_flux_quiet()

   axion_flux_10 = axion_flux_10_func(energy_data)
   axion_flux_1530 = axion_flux_1530_func(energy_data)

   data = np.load(probname)
   Ps = data['Ps']
   omegas = data['omegas']  # eV
   mas = data['mas']  # eV

   Ps_func = interp1d(omegas, Ps.T, axis=1, bounds_error=False, fill_value="extrapolate", kind='linear')
   
   Ngamma_source = Ps_func(energy_data) * axion_flux_10 * src_cm2_eff * tint_quiet * DeltaE
   Ngamma_bkg = Ps_func(energy_data) * axion_flux_1530 * bkg_cm2_eff * tint_quiet * DeltaE

   Ngamma_source_new = np.dot(Ngamma_source, redistribution_matrix)
   Ngamma_bkg_new = np.dot(Ngamma_bkg, redistribution_matrix)

   Ngammais = Ngamma_source_new - Ngamma_bkg_new * area_10/area_1530
   
   Ngammais_func = RegularGridInterpolator((mas, energy_data), Ngammais, bounds_error=False, fill_value=None)#
   
   Ngammais_return = np.empty([len(mas), len(energy)])
   for i, ma in enumerate(mas):
      for j, omega, in enumerate(energy):
         Ngammais_return[i, j] = Ngammais_func([ma, omega])[0]

   return Ngammais_return, mas

def make_bounds():

   probname = npz_dir + 'probabilities.npz'
   bound_file = txt_dir + 'bounds.txt'

   energy, src_counts, bkg_counts, src_cm2_eff, bkg_cm2_eff, redistribution_matrix = get_data_quiet()
   Ngammais, mas = get_Ngammais_quiet(energy, probname)
   tracking = src_counts
   background = bkg_counts 
   
   # Select range of energies
   idxss = np.where((energy >= 4e3) & (energy <= 11e3))[0]
   energy = energy[idxss]
   tracking = tracking[idxss]
   background = background[idxss]
   Ngammais = Ngammais[:, idxss]
   
   log_g10min = -4
   log_g10max = np.log10(1)
   
   g10s = np.logspace(log_g10min, log_g10max, num=4000, endpoint=True)  
   chi2 = np.zeros([len(mas), len(g10s)])
   bounds = np.empty(len(mas))
   dchis = np.empty(len(mas))

   
   nmany = 50000
   likelihoods = np.empty([len(mas), nmany])
   area_tots = np.empty([len(mas)])
   
   
   for i, ma in enumerate(mas):
      tmp = np.empty([len(g10s), len(energy)])
      for j, g10 in enumerate(g10s):
         lambdas = background + g10 ** 4 * Ngammais[i]
         if len(np.where(lambdas < 0)[0]) != 0:
            chi2[i, j] = 1e6
            print('Negative lambda!!')
            break
         idxs_1 = np.where((tracking != 0) & (lambdas != 0))[0]
         idxs_2 = np.where((tracking == 0) & (lambdas != 0))[0]
         tmp[j, idxs_1] = -2 * (
                 tracking[idxs_1] - lambdas[idxs_1] + tracking[idxs_1] * np.log(lambdas[idxs_1] / tracking[idxs_1]))
         tmp[j, idxs_2] = -2 * (-lambdas[idxs_2])

         chi2[i, j] = -2 * np.sum(
            tracking[idxs_1] - lambdas[idxs_1] + tracking[idxs_1] * np.log(lambdas[idxs_1] / tracking[idxs_1])
         ) - 2 * np.sum(-lambdas[idxs_2])
      
      mychi2 = chi2[i]
      
      chi2best_idx = np.argmin(mychi2)
      chi2_best = mychi2[chi2best_idx]
      
      deltachi2 = interp1d(g10s ** 4, mychi2 - chi2_best)
      chi2_func = interp1d(g10s**4, mychi2)
      
      g10s_many = np.logspace(log_g10min, log_g10max, num=nmany)
      g10s_many = g10s_many ** 4
      
      step = 5
      big_g10s = g10s_many[::step]

      likelihood = np.exp(-chi2_func(g10s_many) / 2)
      area_tot = np.trapz(likelihood, x=g10s_many)
      Pcl = 0.95
      area_95 = Pcl * area_tot
      
      area = 0
      for ii, g10 in enumerate(big_g10s):
         if ii == 0:
            continue
         imin = step * (ii - 1)
         imax = step * ii + 1
         little_gs = g10s_many[imin: imax]
         area += np.trapz(np.exp(-chi2_func(little_gs) / 2), x=little_gs)
         if area >= area_95:
            bounds[i] = np.mean(little_gs ** 0.25) * 1e-10
            dchis[i] = deltachi2(np.mean(little_gs))
            break
      
      print('%.3e \t %.3e \t %d \t %.3f \t %.3f' % (ma, bounds[i], chi2best_idx, dchis[i], chi2_best))

   np.savetxt(bound_file, np.c_[mas, bounds, dchis], fmt=['%.6e', '%.6e', '%.3f'], delimiter='\t\t')



################################################################################
################################################################################
################################################################################
if __name__ == '__main__':

   make_bounds()
   


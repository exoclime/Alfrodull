# =============================================================================
# Module for combining the individual opacity sources
# Copyright (C) 2018 Matej Malik
# =============================================================================
# This file is part of HELIOS.
#
#     HELIOS is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     HELIOS is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You find a copy of the GNU General Public License in the main
#     HELIOS directory under <license.txt>. If not, see
#     <http://www.gnu.org/licenses/>.
# =============================================================================

import sys
import os
import h5py
import numpy as np
import scipy as sp
from source import tools as tls
from source import phys_const as pc
from source.opacity_data import OpacityData
from source.opacity_species import OpacitySpecies
from source.opacity_species_cia import OpacitySpeciesCIA
from source import chemical_composition
from source.rayleigh_scattering import calcRayleighCrossSection
from numpy.polynomial.legendre import leggauss as G
from scipy.interpolate import interp1d

import tracemalloc
import matplotlib.pyplot as plt
import pdb

class OpacityMixing:

  def __init__(self, param):
    #the temperature-pressure grid
    self.temperature = np.empty(0)
    self.pressure = np.empty(0)
    self.pressure_log = np.empty(0)

    self.nb_temperatures, self.nb_pressures = self.setAtmosphereGrid()

    #the spectral grid
    self.resolution = param.resolution
    self.wavelength = np.empty(0)
    self.wavenumber = np.empty(0)
    self.sampling_indices = np.empty(0)

    self.ngauss = 20 #gauss-legendre points for ktabls
    self.nbins = 30 #number of k table bins
    self.kbin_edges = np.empty(0)
    self.kbin_centers = np.empty(0)
    self.kbin_dlam = np.empty(0)
    self.ypoints = np.empty(0)

    self.nb_spectral_points = self.setSpectralGrid(param)

    self.ny = 1

  #sets the temperature-pressure grid the opacities will be mixed for
  def setAtmosphereGrid(self):
    self.temperature = np.arange(50, 3050, 50)

    press_list_p1_log = [p for p in np.arange(0, 10, 1)]
    press_list_p2_log = [p for p in np.arange(0.33333333, 9.33333333, 1)]
    press_list_p3_log = [p for p in np.arange(0.66666666, 9.66666666, 1)]
    press_list_log = np.append(press_list_p1_log, np.append(press_list_p2_log, press_list_p3_log))
    press_list_log.sort()
    self.pressure_log = press_list_log

    self.pressure = 10**self.pressure_log

    return self.temperature.size, self.pressure.size



  def setSpectralGrid(self, param):

    print('Setting up spectral grid....\n')

    top_limit = param.special_limits[1] * 1e-4
    l_point = param.special_limits[0] * 1e-4

    if param.format == 'ktable':
        spacing  = np.int(param.sampling_cadence)
        #sample at the helios-k wavenumber resolution
        self.wavenumber = np.arange(np.round(1./top_limit,2), np.round(1./l_point,2)+\
                            param.helios_wavenumber_step, spacing* param.helios_wavenumber_step)

        self.wavenumber = np.round(self.wavenumber[::-1],2)
        self.wavelength = 1.0/self.wavenumber

    elif param.format == 'sampling':
        #original set up from Daniel
        #construct the wavelength grid, assuming a constant resolution
        #since we don't want to continously append elements to the end of an array,
        #we first, we count the number of wavelengths we need:
        counter = 1
        wavelength = l_point

        while wavelength < top_limit:
          wavelength *= (param.resolution + 1) / param.resolution
          counter += 1

        #now we can construct the wavelength grid
        self.wavelength = np.empty(counter)

        self.wavelength[0] = l_point #this can be removed, it's just here to exactly reproduce the grid from the original ktable

        for i in range (1, self.wavelength.size):
          self.wavelength[i] = self.wavelength[i-1] * (param.resolution + 1) / param.resolution

        self.wavelength = self.wavelength[0:-1]

        #round the wavenumbers to the closest point in the HELIOS-k grid
        self.wavenumber = np.round(1.0/self.wavelength, 2)


    #correlate the chosen grid with the HELIOS-k wavenumber grid
    #first: construct the original HELIOS-k grid with constant wavenumber step
    print('Correlating spectral grid with opacity grid wavenumbers...')
    helios_wavenumbers = np.round(np.arange(0, np.max(self.wavenumber)+param.helios_wavenumber_step, param.helios_wavenumber_step), 2)

    #only works when using the native helios-k resolution
    min_index = np.where(helios_wavenumbers == np.min(self.wavenumber))[0][0]
    max_index = np.where(helios_wavenumbers == np.max(self.wavenumber))[0][0]

    if param.format == 'sampling':
        #original method from Daniel (won't work on very large arrays)
        self.sampling_indices = np.empty(self.wavenumber.size, dtype=int)

        #locate the indices of the sampled wavenumbers within the HELIOS-k grid
        #note: since the full wavenumber grid is ordered, we can continously shrink the array we have
        #to locate the next wavenumbers in
        self.sampling_indices[0] = np.where(helios_wavenumbers == self.wavenumber[0])[0]

        for i in range(1,self.wavenumber.size):
          self.sampling_indices[i] = np.where(helios_wavenumbers[0:self.sampling_indices[i-1]] == self.wavenumber[i])[0]

        print('Final grid:\n  -Number of points:', self.wavelength.size, '\n  -min/max wavelength:', self.wavelength[0], self.wavelength[-1], '\n')

    if param.format == 'ktable':
        #ktable grid
        self.sampling_indices = np.arange(min_index,max_index+1,spacing)[::-1]

        self.nbins = np.int(param.resolution)
        if param.grid_spacing == 'loglambda':  #log spacing in wavelength
            loglam_int = np.linspace(np.log10(l_point),np.log10(top_limit),self.nbins+1)
        else:  #linear spacing in wavenumber
            waven_int = np.linspace(1./top_limit, 1./l_point, self.nbins+1)[::-1]
            loglam_int = np.log10(1.0/waven_int)
        self.kbin_edges = 10**loglam_int

        loglam_cen = 0.5*(loglam_int[:-1]+loglam_int[1:])
        self.kbin_centers = 10**loglam_cen
        self.kbin_dlam = self.kbin_edges[1:]-self.kbin_edges[:-1]
        self.ypoints = 0.5*G(self.ngauss)[0] + 0.5

    return self.wavenumber.size



  #Mixes the absorption and scattering opacties for all species
  def mixOpacities(self, param, species, chemistry):

    #create 2D list of Opacity data objects and initialise their opacities with 0
    mixed_opacity = [[OpacityData('', self.temperature[i], self.pressure_log[j]) for j in range(self.nb_pressures)] for i in range(self.nb_temperatures)]

    for row in mixed_opacity:
      for i in row:
        i.opacity = np.zeros(self.nb_spectral_points)


    #first we do the Rayleigh cross-sections for all scattering species
    print('Start mixing scattering species...')
    mixed_rayleigh_scattering = np.zeros((self.nb_temperatures, self.nb_pressures, self.nb_spectral_points))

    for s in species:
      if s.scattering == True:
        self.mixSingleRayleighSpecies(s, chemistry, mixed_rayleigh_scattering)


    #now we do the absorption species
    print('Start mixing absorbing species...')
    for s in species:
      if s.absorbing == True:
        self.mixSingleSpecies(s, chemistry, mixed_opacity)


    #put all opacity values in a really long array :-?
    combined_opacities = np.zeros(self.ny * self.nb_spectral_points * self.nb_pressures * self.nb_temperatures)
    combined_cross_sections = np.zeros(self.nb_spectral_points * self.nb_pressures * self.nb_temperatures)

    for t in range(self.nb_temperatures):
      for p in range(self.nb_pressures):
        start_index = self.nb_spectral_points*p + self.nb_spectral_points*self.nb_pressures*t

        combined_opacities[start_index:start_index + self.nb_spectral_points] = mixed_opacity[t][p].opacity
        combined_cross_sections[start_index:start_index + self.nb_spectral_points] = mixed_rayleigh_scattering[t, p, :]


    self.saveToFile(param, species, combined_opacities, combined_cross_sections, chemistry)

    return None


  #adds a single absorption species to the mix
  def mixSingleSpecies(self, species, chemistry, mixed_opacity,itemp='all',ipress='all'):

    chem_index = []

    #the indices of our species in the chemistry
    #some of them (e.g. CIA) have multiple indices
    if type(species.fastchem_symbol) == str:
      chem_index.append(np.where(chemistry.species_symbol == species.fastchem_symbol)[0][0])
    else:
      for i in range(len(species.fastchem_symbol)):
        chem_index.append(np.where(chemistry.species_symbol == species.fastchem_symbol[i])[0][0])

    if itemp == 'all' and ipress == 'all':
        print("Current mixing species: ", species.name)
        for t in range(self.nb_temperatures):
          for p in range(self.nb_pressures):

            mass_mixing_ratio = 1.0

            if type(species.fastchem_symbol) == str:
               mass_mixing_ratio = chemistry.mixing_ratio[chem_index[0], t, p] * species.molecular_weight / chemistry.mean_molecular_weight[t,p]
            else:
              for i in range(len(chem_index)):
                if species.molecular_weight[i] != 0.0:
                  mass_mixing_ratio *= chemistry.mixing_ratio[chem_index[i], t, p] * species.molecular_weight[i]  / chemistry.mean_molecular_weight[t,p]
                else:
                  mass_mixing_ratio *= chemistry.mixing_ratio[chem_index[i], t, p]

            if mass_mixing_ratio < 1e-15:
              continue

            species_opacity = species.getOpacity(self.temperature[t], self.pressure_log[p], chemistry.mean_molecular_weight[t,p], self.sampling_indices, self.wavelength)

            mixed_opacity[t][p].opacity += mass_mixing_ratio * 10**species_opacity.opacity

            tls.percent_counter(t, self.nb_temperatures, p, self.nb_pressures)

    elif itemp != 'all' and ipress == 'all':
      for p in range(self.nb_pressures):

        mass_mixing_ratio = 1.0

        if type(species.fastchem_symbol) == str:
           mass_mixing_ratio = chemistry.mixing_ratio[chem_index[0], itemp, p] * species.molecular_weight / chemistry.mean_molecular_weight[itemp,p]
        else:
          for i in range(len(chem_index)):
            if species.molecular_weight[i] != 0.0:
              mass_mixing_ratio *= chemistry.mixing_ratio[chem_index[i], itemp, p] * species.molecular_weight[i]  / chemistry.mean_molecular_weight[itemp,p]
            else:
              mass_mixing_ratio *= chemistry.mixing_ratio[chem_index[i], itemp, p]

        if mass_mixing_ratio < 1e-15:
          continue

        species_opacity = species.getOpacity(self.temperature[itemp], self.pressure_log[p], chemistry.mean_molecular_weight[itemp,p], self.sampling_indices, self.wavelength)

        mixed_opacity[p,:] += mass_mixing_ratio * 10**species_opacity.opacity

    else:
        mass_mixing_ratio = 1.0

        if type(species.fastchem_symbol) == str:
           mass_mixing_ratio = chemistry.mixing_ratio[chem_index[0], itemp, ipress] * species.molecular_weight / chemistry.mean_molecular_weight[itemp,ipress]
        else:
          for i in range(len(chem_index)):
            if species.molecular_weight[i] != 0.0:
              mass_mixing_ratio *= chemistry.mixing_ratio[chem_index[i], itemp, ipress] * species.molecular_weight[i]  / chemistry.mean_molecular_weight[itemp,ipress]
            else:
              mass_mixing_ratio *= chemistry.mixing_ratio[chem_index[i], itemp, ipress]

        if mass_mixing_ratio > 1e-15:

            species_opacity = species.getOpacity(self.temperature[itemp], self.pressure_log[ipress], chemistry.mean_molecular_weight[itemp,ipress], self.sampling_indices, self.wavelength)

            mixed_opacity += mass_mixing_ratio * 10**species_opacity.opacity

        # tls.percent_counter(itemp, self.nb_temperatures, ipress, self.nb_pressures)

    # print("\n")

    return None



  #adds a single specoes to the Rayleigh cross sections
  def mixSingleRayleighSpecies(self, species, chemistry, mixed_cross_sections,itemp='all',ipress='all',ktable=False):

    #our Rayleigh cross sections do not depend on pressure and temperature
    #thus, we only need to compute them once
    if ktable == True:
        species_cross_section = calcRayleighCrossSection(species.name, self.kbin_centers)
    else:
        species_cross_section = calcRayleighCrossSection(species.name, self.wavelength)

    #the index of our species in the chemistry
    chem_index = np.where(chemistry.species_symbol == species.fastchem_symbol)[0][0]

    if itemp == 'all' and ipress == 'all':
        print("Current mixing species: ", species.name)
        for t in range(self.nb_temperatures):
          for p in range(self.nb_pressures):
            mixed_cross_sections[t, p, :] += chemistry.mixing_ratio[chem_index, t, p] * species_cross_section
            tls.percent_counter(t, self.nb_temperatures, p, self.nb_pressures)

    elif itemp != 'all' and ipress == 'all':
        for p in range(self.nb_pressures):
            mixed_cross_sections[p, :] += chemistry.mixing_ratio[chem_index, itemp, p] * species_cross_section

    else:
        mixed_cross_sections += chemistry.mixing_ratio[chem_index, itemp, ipress] * species_cross_section
        #tls.percent_counter(itemp, self.nb_temperatures, ipress, self.nb_pressures)

    # print("\n")

    return None

  #save the output to an HDF5 file
  def saveToFile(self, param, species, combined_opacities, combined_cross_sections, chemistry):
    """ write to hdf5 file """

    # create directory if necessary
    try:
      os.makedirs(param.final_path)
    except OSError:
      if not os.path.isdir(param.final_path):
        raise

    if param.format == 'ktable':
      filename = "mixed_opac_ktable.h5"
    elif param.format == 'sampling':
      filename = "mixed_opac_sampling.h5"


    pressure_factor = 1.0
    opac_factor = 1.0
    cross_sect_factor = 1.0
    wavelength_factor = 1.0

    # change units to MKS if chosen in param file
    if param.units == "MKS":
      pressure_factor = 1e-1
      opac_factor = 1e-1
      cross_sect_factor = 1e-4
      wavelength_factor = 1e-2

      #if param.format == "ktable":
      #  self.k_i = [k * 1e-2 for k in self.k_i]
      #  self.k_w = [k * 1e-2 for k in self.k_w]

    mu = np.empty(self.nb_temperatures*self.nb_pressures)

    for t in range(self.nb_temperatures):
      start_index = self.nb_pressures*t
      mu[start_index:start_index + self.nb_pressures] = chemistry.mean_molecular_weight[t,:]


    # molname_list = np.empty(len(species), dtype=object)

    # for i in range(len(species)):
    #   molname_list[i] = species[i].name

    molname_list = ['' for i in range(len(species))]

    for i in range(len(species)):
      molname_list[i] = species[i].name.encode('utf8')


    with h5py.File(param.final_path + filename, "w") as mixed_file:
            mixed_file.create_dataset("pressures", data=self.pressure*pressure_factor)
            mixed_file.create_dataset("temperatures", data=self.temperature)
            mixed_file.create_dataset("meanmolmass", data=mu)
            mixed_file.create_dataset("kpoints", data=combined_opacities * opac_factor)
            mixed_file.create_dataset("weighted Rayleigh cross-sections", data=combined_cross_sections * cross_sect_factor)
            mixed_file.create_dataset("included molecules", data=molname_list)
            mixed_file.create_dataset("wavelengths", data=self.wavelength*wavelength_factor)
            mixed_file.create_dataset("FastChem path", data=param.fastchem_path)
            mixed_file.create_dataset("units", data=param.units)

    # if param.format == 'ktable':
    #         mixed_file.create_dataset("center wavelengths", data=self.k_x)
    #         mixed_file.create_dataset("interface wavelengths",data=self.k_i)
    #         mixed_file.create_dataset("wavelength width of bins",data=self.k_w)
    #         mixed_file.create_dataset("ypoints",data=self.k_y)

    return None

  def setuph5File(self, param, species, chemistry, overwrite=False, format = 'sampling'):
      """ create hdf5 file, don't yet fill data sets """

      # create directory if necessary
      try:
        os.makedirs(param.final_path)
      except OSError:
        if not os.path.isdir(param.final_path):
          raise

      if format == 'ktable':
        ny = self.ngauss
        filename = "mixed_opac_ktable.h5"
      elif format == 'sampling':
        ny = 1
        filename = "mixed_opac_sampling.h5"


      pressure_factor = 1.0
      opac_factor = 1.0
      cross_sect_factor = 1.0
      wavelength_factor = 1.0

      # change units to MKS if chosen in param file
      if param.units == "MKS":
        pressure_factor = 1e-1
        wavelength_factor = 1e-2

        #if param.format == "ktable":
        #  self.k_i = [k * 1e-2 for k in self.k_i]
        #  self.k_w = [k * 1e-2 for k in self.k_w]

      mu = np.empty(self.nb_temperatures*self.nb_pressures)

      for t in range(self.nb_temperatures):
        start_index = self.nb_pressures*t
        mu[start_index:start_index + self.nb_pressures] = chemistry.mean_molecular_weight[t,:]


      # molname_list = np.empty(len(species), dtype=object)

      # for i in range(len(species)):
      #   molname_list[i] = species[i].name

      molname_list = ['' for i in range(len(species))]

      for i in range(len(species)):
        molname_list[i] = species[i].name.encode('utf8')


      if os.path.exists(param.final_path + filename) and overwrite==False:
          mixed_file = h5py.File(param.final_path + filename,'r+')

      else:
          mixed_file = h5py.File(param.final_path + filename, "w")
          if format == 'sampling':
              mixed_file.create_dataset("pressures", data=self.pressure*pressure_factor,compression='gzip', compression_opts=4)
              mixed_file.create_dataset("temperatures", data=self.temperature,compression='gzip', compression_opts=4)
              mixed_file.create_dataset("meanmolmass", data=mu,compression='gzip', compression_opts=4)
              mixed_file.create_dataset("included molecules", data=molname_list)
              mixed_file.create_dataset("FastChem path", data=param.fastchem_path)
              mixed_file.create_dataset("units", data=param.units)
              mixed_file.create_dataset("kpoints", (ny * self.nb_temperatures*self.nb_pressures*self.nb_spectral_points,), 'f',compression='gzip', compression_opts=4)
              mixed_file.create_dataset("weighted Rayleigh cross-sections",(self.nb_temperatures*self.nb_pressures*self.nb_spectral_points,), 'f',compression='gzip', compression_opts=4)
              mixed_file.create_dataset("wavelengths", data=self.wavelength*wavelength_factor,compression='gzip', compression_opts=4)
          elif format == 'ktable':
              mixed_file.create_dataset("pressures", data=self.pressure*pressure_factor)
              mixed_file.create_dataset("temperatures", data=self.temperature)
              mixed_file.create_dataset("meanmolmass", data=mu)
              mixed_file.create_dataset("included molecules", data=molname_list)
              mixed_file.create_dataset("FastChem path", data=param.fastchem_path)
              mixed_file.create_dataset("units", data=param.units)
              mixed_file.create_dataset("kpoints", (ny * self.nb_temperatures*self.nb_pressures*self.nbins,), 'f')
              mixed_file.create_dataset("weighted Rayleigh cross-sections",(self.nb_temperatures*self.nb_pressures*self.nbins,), 'f')
              mixed_file.create_dataset("wavelengths", data=self.kbin_centers*wavelength_factor)
              mixed_file.create_dataset("center wavelengths", data=self.kbin_centers*wavelength_factor)
              mixed_file.create_dataset("interface wavelengths", data=self.kbin_edges*wavelength_factor)
              mixed_file.create_dataset("wavelength width of bins", data=self.kbin_dlam*wavelength_factor)
              mixed_file.create_dataset("ypoints", data=self.ypoints)

          mixed_file.create_dataset("itemp",data=(0,))
          mixed_file.create_dataset("ipress",data=(0,))

      itemp = mixed_file['itemp'][0]
      ipress = mixed_file['ipress'][0]

      # if param.format == 'ktable':
      #         mixed_file.create_dataset("center wavelengths", data=self.k_x)
      #         mixed_file.create_dataset("interface wavelengths",data=self.k_i)
      #         mixed_file.create_dataset("wavelength width of bins",data=self.k_w)
      #         mixed_file.create_dataset("ypoints",data=self.k_y)

      return mixed_file, itemp, ipress

  def reopenh5(self, param, format = 'sampling'):
      if format == 'ktable':
        filename = "mixed_opac_ktable.h5"
      elif format == 'sampling':
        filename = "mixed_opac_sampling.h5"
      mixed_file = h5py.File(param.final_path + filename,'r+')

      return mixed_file

  def mixOpacitiesChunky(self, param, species, chemistry):

    mixed_file, itemp_start, ipress_start = self.setuph5File(param, species, chemistry,
                                                                overwrite=True)
    mixed_file.close()
    ktable_file, ii, jj = self.setuph5File(param,species,chemistry,overwrite = True,format='ktable')
    ktable_file.close()

    #create 2D list of Opacity data objects and initialise their opacities with 0
    mixed_opacity = [[OpacityData('', self.temperature[i], self.pressure_log[j]) for j in range(self.nb_pressures)] for i in range(self.nb_temperatures)]

    print("Mixing species: ",[s.name for s in species])

    # tracemalloc.start()
    for itemp in np.arange(itemp_start,self.nb_temperatures): #loop over temperatures
        if itemp == itemp_start:
            ipress_start_this_time = ipress_start
        else:
            ipress_start_this_time = 0

        for ipress in np.arange(ipress_start_this_time,self.nb_pressures): #loop over pressures

            mixed_opacity_tmp = np.zeros(self.nb_spectral_points)

            mixed_rayleigh_tmp = np.zeros(self.nb_spectral_points)

            for s in species:
                if s.scattering == True:
                    self.mixSingleRayleighSpecies(s, chemistry, mixed_rayleigh_tmp,itemp=itemp,ipress=ipress)

                if s.absorbing == True:
                    self.mixSingleSpecies(s, chemistry, mixed_opacity_tmp,itemp=itemp,ipress=ipress)

            # do sorting here
            kpoints_tmp = np.zeros(self.ngauss*self.nbins)
            rayleigh_k_tmp = np.zeros(self.nbins)
            for ibin in np.arange(self.nbins):
                indices = np.where(np.logical_and(self.wavelength>=self.kbin_edges[ibin],
                                                    self.wavelength<self.kbin_edges[ibin+1]))
                kbin = np.sort(mixed_opacity_tmp[indices])
                N = len(kbin)
                y = np.linspace(0,1,N)

                kint = 10**(interp1d(y, np.log10(kbin), kind = 'linear')(self.ypoints))
                kint[kint<1e-15] = 1e-15
                if np.isnan(kint).any() or np.isinf(kint).any():
                    pdb.set_trace()

                start_ind_k = self.ngauss*ibin
                kpoints_tmp[start_ind_k:start_ind_k+self.ngauss] = kint

            for s in species:
                if s.scattering == True:
                    self.mixSingleRayleighSpecies(s, chemistry, rayleigh_k_tmp,itemp=itemp,ipress=ipress,ktable=True)

            #report our progress
            tls.percent_counter(itemp, self.nb_temperatures, ipress, self.nb_pressures)

            if param.units == "MKS":
                opac_factor = 1e-1
                cross_sect_factor = 1e-4
            else:
                opac_factor = 1.0
                cross_sect_factor = 1.0

            #save this PT slice to file (ktable)
            start_index = self.nbins*ipress + self.nbins*self.nb_pressures*itemp
            end_index = start_index + self.nbins
            ktable_file = self.reopenh5(param,format='ktable')
            ktable_file['kpoints'][self.ngauss*start_index:self.ngauss*end_index] = kpoints_tmp*opac_factor
            ktable_file['weighted Rayleigh cross-sections'][start_index:end_index] = rayleigh_k_tmp*cross_sect_factor
            ktable_file.close()

            #save this PT slice to file (hires sampling)
            start_index = self.nb_spectral_points*ipress + self.nb_spectral_points*self.nb_pressures*itemp
            end_index = start_index + self.nb_spectral_points
            mixed_file = self.reopenh5(param)
            mixed_file['kpoints'][start_index:end_index] = mixed_opacity_tmp*opac_factor
            mixed_file['weighted Rayleigh cross-sections'][start_index:end_index] = mixed_rayleigh_tmp*cross_sect_factor
            mixed_file['ipress'][0] = ipress + 1
            mixed_file.close()

            # plt.loglog(self.wavelength*1e4,mixed_opacity_tmp)
            # plt.title("(p,T) = (%#.3g,%#.1f)"%(self.pressure[ipress],self.temperature[itemp]))
            # plt.ylim(1e-15,1e5)
            # plt.show()

        mixed_file = self.reopenh5(param)
        mixed_file['itemp'][0] = itemp + 1
        mixed_file['ipress'][0] = 0
        mixed_file.close()

    # mixed_file.close()
    # snapshot = tracemalloc.take_snapshot()
    # import pdb; pdb.set_trace()

  def mixOpacitiesChunky2(self, param, species, chemistry):

    mixed_file, itemp_start, ipress_start = self.setuph5File(param, species, chemistry,
                                                                overwrite=True)
    mixed_file.close()
    ktable_file, ii, jj = self.setuph5File(param,species,chemistry,overwrite = True,format='ktable')
    ktable_file.close()

    #create 2D list of Opacity data objects and initialise their opacities with 0
    mixed_opacity = [[OpacityData('', self.temperature[i], self.pressure_log[j]) for j in range(self.nb_pressures)] for i in range(self.nb_temperatures)]

    print("Mixing species: ",[s.name for s in species])

    # tracemalloc.start()
    for itemp in np.arange(itemp_start,self.nb_temperatures): #loop over temperatures
        mixed_opacity_tmp = np.zeros((self.nb_pressures,self.nb_spectral_points))

        mixed_rayleigh_tmp = np.zeros((self.nb_pressures,self.nb_spectral_points))

        for s in species:
            if s.scattering == True:
                self.mixSingleRayleighSpecies(s, chemistry, mixed_rayleigh_tmp,itemp=itemp,ipress='all')

            if s.absorbing == True:
                self.mixSingleSpecies(s, chemistry, mixed_opacity_tmp,itemp=itemp,ipress='all')

        # do sorting here
        kpoints_tmp = np.zeros((self.nb_pressures,self.ngauss*self.nbins))
        rayleigh_k_tmp = np.zeros((self.nb_pressures,self.nbins))
        for ibin in np.arange(self.nbins):
            indices = np.where(np.logical_and(self.wavelength>=self.kbin_edges[ibin],
                                                self.wavelength<self.kbin_edges[ibin+1]))[0]
            kbin = np.sort(mixed_opacity_tmp[:,indices],axis=1)
            N = len(indices)
            y = np.linspace(0,1,N)

            kint = interp1d(y, kbin, kind = 'linear',axis=1)(self.ypoints)
            kint[kint<1e-15] = 1e-15

            start_ind_k = self.ngauss*ibin
            kpoints_tmp[:,start_ind_k:start_ind_k+self.ngauss] = kint

        for s in species:
            if s.scattering == True:
                self.mixSingleRayleighSpecies(s, chemistry, rayleigh_k_tmp,itemp=itemp,ipress='all',ktable=True)

        #report our progress
        tls.percent_counter(itemp, self.nb_temperatures)

        if param.units == "MKS":
            opac_factor = 1e-1
            cross_sect_factor = 1e-4
        else:
            opac_factor = 1.0
            cross_sect_factor = 1.0

        #save this T slice to file (ktable)
        start_index = self.nbins*self.nb_pressures*itemp
        end_index = start_index + self.nbins*self.nb_pressures
        ktable_file = self.reopenh5(param,format='ktable')
        ktable_file['kpoints'][self.ngauss*start_index:self.ngauss*end_index] = kpoints_tmp.ravel()*opac_factor
        ktable_file['weighted Rayleigh cross-sections'][start_index:end_index] = rayleigh_k_tmp.ravel()*cross_sect_factor
        ktable_file.close()

        #save this T slice to file (hires sampling)
        start_index = self.nb_spectral_points*self.nb_pressures*itemp
        end_index = start_index + self.nb_spectral_points*self.nb_pressures
        mixed_file = self.reopenh5(param)
        try:
            mixed_file['kpoints'][start_index:end_index] = mixed_opacity_tmp.ravel()*opac_factor
        except:
            import pdb; pdb.set_trace()
        mixed_file['weighted Rayleigh cross-sections'][start_index:end_index] = mixed_rayleigh_tmp.ravel()*cross_sect_factor
        mixed_file.close()

        # plt.loglog(self.wavelength*1e4,mixed_opacity_tmp)
        # plt.title("(p,T) = (%#.3g,%#.1f)"%(self.pressure[ipress],self.temperature[itemp]))
        # plt.ylim(1e-15,1e5)
        # plt.show()

    mixed_file = self.reopenh5(param)
    mixed_file['itemp'][0] = itemp + 1
    mixed_file.close()

#,compression='gzip', compression_opts=4

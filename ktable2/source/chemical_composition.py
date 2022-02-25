
import sys
import os
import numpy as np
import scipy as sp
from source import param
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


class ChemicalComposition:

  def __init__(self, param, grid_temperature, grid_pressure, species):
    self.mixing_ratio = np.empty(0)
    self.mean_molecular_weight = np.zeros((grid_temperature.size, grid_pressure.size))
    self.species_symbol = np.empty(0, dtype=object)


    if type(species[0].mixing_ratio) == float:
      self.setFixedChemistry(species, grid_temperature, grid_pressure)
    elif species[0].mixing_ratio == 'FastChem' or species[0].mixing_ratio == 'fastchem':
      self.setFastChemInput(param, species, grid_temperature, grid_pressure)
    else:
      print('Unkown entry', species[0].mixing_ratio, 'for mixing ratio description in parameter file:', species[0].mixing_ratio)




  #set up the mixing ratio using the ones from the parameter file, in case we don't use FastChem
  def setFixedChemistry(self, species, grid_temperature, grid_pressure):

    nb_species = len(species)

    self.mixing_ratio = np.zeros((nb_species,grid_temperature.size,grid_pressure.size))

    #fill in the constant mixing ratios
    for i in range(len(species)):
      if type(species[i].mixing_ratio) == float:
        self.mixing_ratio[i,:,:] = species[i].mixing_ratio
        self.species_symbol = np.append(self.species_symbol,species[i].name)
      else:
        print('Expected a float-value mixing ratio for species', species[i].name, 'but found', species[i].mixing_ratio, 'instead.')
        exit()

    #calculate the mean molecular weight
    for i in range(np.size(self.mean_molecular_weight, 0)):
      for j in range(np.size(self.mean_molecular_weight, 1)):
        mixing_ratio_sum = 0

        for s in species:
          self.mean_molecular_weight[i,j] += s.molecular_weight * s.mixing_ratio
          mixing_ratio_sum += s.mixing_ratio

        if mixing_ratio_sum > 1:
          print("The sum of all mixing ratios exceeds 1. I assume this is an error....")
          exit()


    return None


  #initialises the FastChem output for later use
  def setFastChemInput(self, param, species, grid_temperature, grid_pressure):

    if param.fastchem_path == '':
      print('No fastchem file path found in parameter file')
      exit()

    fastchem_mixing_ratio = np.empty(0)
    fastchem_temperature = np.empty(0)
    fastchem_pressure = np.empty(0)
    fastchem_mean_molecular_weight = np.empty(0)

    fastchem_temperature, fastchem_pressure, fastchem_mixing_ratio, fastchem_mean_molecular_weight = self.readFastChemFile(param.fastchem_path, species)

    print("Interpolating FastChem output\n")

    #create a triagulation of the FastChem grid that we us in the interpolation
    grid_points = np.column_stack((fastchem_temperature, np.log10(fastchem_pressure)))
    triangulation = Delaunay(grid_points)

    self.mixing_ratio = np.zeros((self.species_symbol.size, grid_temperature.size, grid_pressure.size))

    #first, interpolate the mixing ratios (done in log10)
    for i in range(np.size(self.mixing_ratio, 0)):
      interpol_data = self.interpolateChemistryData(np.log10(fastchem_mixing_ratio[i,:]), triangulation, grid_temperature, np.log10(grid_pressure))

      self.mixing_ratio[i,:,:] = 10**interpol_data

    #now interpolate the mean molecular weight
    self.mean_molecular_weight = self.interpolateChemistryData(fastchem_mean_molecular_weight, triangulation, grid_temperature, np.log10(grid_pressure))


    return None




  def readFastChemFile(self, fastchem_path, species):
    """ read in the fastchem mixing ratios"""

    print('Reading chemistry data files\n')

    fastchem_data_low = np.genfromtxt(fastchem_path + 'chem_low.dat',
                                      names=True, dtype=None, skip_header=0, deletechars=" !#$%&'()*,./:;<=>?@[\]^{|}~")

    fastchem_data_high = np.genfromtxt(fastchem_path + 'chem_high.dat',
                                       names=True, dtype=None, skip_header=0, deletechars=" !#$%&'()*,./:;<=>?@[\]^{|}~")

    #temperature and pressure from the chemical grid
    fastchem_pressure = np.concatenate((fastchem_data_low['Pbar'], fastchem_data_high['Pbar']))
    fastchem_temperature = np.concatenate((fastchem_data_low['Tk'], fastchem_data_high['Tk']))

    fastchem_pressure *= 1e6 #convert unit from bar to dyn cm-2


    # mean molecular weight
    fastchem_mean_molecular_weight = np.concatenate((fastchem_data_low['mu'], fastchem_data_high['mu']))

    #find all unique species from the input file
    #note that some opacity sources can have multiple chemical species
    for s in species:
      if type(s.fastchem_symbol) == str:
        if s.fastchem_symbol not in self.species_symbol:
          self.species_symbol = np.append(self.species_symbol, s.fastchem_symbol)
      else:
        for n in s.fastchem_symbol:
          if n not in self.species_symbol:
            self.species_symbol = np.append(self.species_symbol, n)


    #extract the mixing ratios for all species we need
    fastchem_mixing_ratio = np.zeros((self.species_symbol.size, fastchem_pressure.size))

    for i in range(self.species_symbol.size):
      try:
        species_mixing_ratio = np.concatenate((fastchem_data_low[self.species_symbol[i]], fastchem_data_high[self.species_symbol[i]]))
        species_mixing_ratio[species_mixing_ratio < 1e-300] = 1e-300  #set minimum value because we interpolate in log10 later
        fastchem_mixing_ratio[i,:] = species_mixing_ratio
      except ValueError:
        print('Chemical species', self.species_symbol[i], 'not found in FastChem files')
        exit()


    return fastchem_temperature, fastchem_pressure, fastchem_mixing_ratio, fastchem_mean_molecular_weight



  #Interpolation for the chemistry data
  #uses the linear 2D interpolation from scipy
  #for points outside of the calculated grid, it uses the nearest neighbour
  def interpolateChemistryData(self, data, triangulation, interpol_temperature, interpol_pressure):

    interpolator = LinearNDInterpolator(triangulation, data)
    interpolator2 = NearestNDInterpolator(triangulation, data)

    interpol_data = np.zeros((interpol_temperature.size,interpol_pressure.size))

    for i in range(interpol_temperature.size):
      for j in range(interpol_pressure.size):
        interpol_data[i,j] = interpolator([interpol_temperature[i], interpol_pressure[j]])

        #in case we are outside of the tabulated grid, we use the nearest neighbour
        if np.isnan(interpol_data[i,j]) == True:
          interpol_data[i,j] = interpolator2([interpol_temperature[i], interpol_pressure[j]])


    return interpol_data

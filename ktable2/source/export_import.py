
import sys
import os
import numpy as np
from source.opacity_species import OpacitySpecies
from source import opacity_mixing


class DataImportExport:

  def saveSampledData(self, opacity_species, opacity_mixer, file_path):

    print('Saving sampled data to', file_path, '\n')

    data_export = np.array([opacity_species, opacity_mixer], dtype=object)

    np.save(file_path, data_export, allow_pickle=True, fix_imports=True)

    return None


  def readSampledData(self, file_path):

    print('Reading in sampled data from', file_path)
    sampled_data = np.load(file_path, allow_pickle=True)

    opacity_mixer = sampled_data[1]
    opacity_species = sampled_data[0]


    print('Found the following spectral grid:')
    print('  -Resolution:', opacity_mixer.resolution)
    print('  -Number of points:', opacity_mixer.wavelength.size)
    print('  -min/max wavelength:', opacity_mixer.wavelength[0], opacity_mixer.wavelength[-1], '\n')


    print('Found the following opacity grid species:\n')
    for s in opacity_species:
      print(s.name)
      print('opacity grid:')
      print('  scattering: ', s.scattering)
      print('  absorbing:', s.absorbing)
      print('  number of files:',len(s.opacity_data))
      print('  number of temperatures:', s.nb_temperatures, ' number of pressures:', s.nb_pressures)
      print('  temperature min:', s.min_temperature, 'max:', s.max_temperature)
      print('  pressure min:', s.min_pressure, 'max:', s.max_pressure, '\n')

    return opacity_species, opacity_mixer
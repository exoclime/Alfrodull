# ==============================================================================
# This program generates the opacity table used in HELIOS.
# Copyright (C) 2018 Matej Malik
#
# ==============================================================================
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
# ==============================================================================

from source import param as para
from source import information as inf
from source.opacity_species import OpacitySpecies
from source import chemical_composition
from source import opacity_mixing
import numpy as np
from source import export_import
import time

def main():
    """ main function to run k-table generation """

    species = []

    # create objects of classes
    param = para.Param()
    param.parseCommandLineArguments()
    param.readParameterFile(param.param_file, species)

    if param.read_sampled_data == True:
      import_data = export_import.DataImportExport()

      species, opacity_mixer = import_data.readSampledData(param.sampled_file_path)
    else:
      opacity_mixer = opacity_mixing.OpacityMixing(param)

    if param.save_sampled_data == True:
      print('Reading the entire opacity grid now (might take a while)')

      for s in species:
        s.loadAllOpacitiyData(opacity_mixer.sampling_indices)

      export_data = export_import.DataImportExport()
      export_data.saveSampledData(species, opacity_mixer, param.sampled_file_path)


    if param.save_sampled_data == False:
      chemistry = chemical_composition.ChemicalComposition(param, opacity_mixer.temperature, opacity_mixer.pressure, species)

      if param.format == 'ktable':
          opacity_mixer.mixOpacitiesChunky(param, species, chemistry)
      elif param.format == 'sampling':
          opacity_mixer.mixOpacities(param,species,chemistry)


    # write information file
    # this probably does not need to be an object class :-?
    info = inf.Info()
    info.write(param)

    print("\nDone! Production of k-tables went fine :)")

# run the whole thing
first = time.time()

main()

last = time.time()
print(last - first)

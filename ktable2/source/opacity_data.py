
from dataclasses import dataclass, field
import numpy as np
import os



@dataclass
class OpacityData:
    file_path : str = field(init=True)
    temperature : float = field(init=True)
    log_pressure : float = field(init=True)

    opacity : np.array = field(default=np.empty(0), init=False, compare=False)


    #read in a single binary opacity file and extract the data at the sampled wavenumbers
    #the data in the file is stored in single precision
    def readOpacityFile(self, sample_indices, log_opacities):
      if self.file_path == "":
        return None

      #print("Reading ", self.file_path)

      opacity_data = np.fromfile(self.file_path, dtype=np.float32)

      #convert the array to double precision
      opacity_data = np.float64(opacity_data)

      #the largest wavenumber index we need for the sampled data
      nb_wavenumbers = np.max(sample_indices) + 1

      #if we have less data than required, append zeros
      if opacity_data.size < nb_wavenumbers:
        opacity_data = np.append(opacity_data, np.full(nb_wavenumbers - opacity_data.size, 0))

      #pick only opacities at the sampled wavenumbers
      self.opacity = opacity_data[sample_indices]

      #since some of the original opacity values might be 0,
      #we need to replace them with a small number before we use log10
      if log_opacities == False:
        self.opacity[self.opacity == 0] = 1e-300
      else:
        self.opacity[self.opacity == 0] = -300

      #we want to interpolate the opacity in log space later
      if log_opacities == False:
        self.opacity = np.log10(self.opacity)

      return None

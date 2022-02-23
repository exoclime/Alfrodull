
import numpy as np




def calcRayleighCrossSection(species_name, wavelength):

  cross_section = np.zeros(wavelength.size)

  if species_name == 'H2O':
    cross_section = rayleighH2O(wavelength)
  elif species_name == 'H2':
    cross_section = rayleighH2(wavelength)
  elif species_name == 'He':
    cross_section = rayleighHe(wavelength)
  elif species_name == 'H':
    cross_section = rayleighH(wavelength)
  elif species_name == 'CO':
    cross_section = rayleighCO(wavelength)
  elif species_name == 'CO2':
    cross_section = rayleighCO2(wavelength)
  elif species_name == 'CH4':
    cross_section = rayleighCH4(wavelength)
  elif species_name == 'N2':
    cross_section = rayleighN2(wavelength)
  elif species_name == 'O2':
    cross_section = rayleighO2(wavelength)
  else:
    print('No Rayleigh cross-sections for species', species_name, 'found!\n')


  return cross_section



def rayleighH2O(wavelength):
  """ calculates the refractive index of H2O """

  #this is the number density of water at standard temperature and pressure
  #determined from the Avogradro constant and the properties of water at STP
  reference_density = 3.34279671749673e+22

  #values for water at STP
  delta = 1.0
  theta = 1.0

  lambda_uv = 0.229202
  lambda_ir = 5.432937

  a_coeff = np.array([0.244257733, 0.974634476e-2, -0.373234996e-2, 0.268678472e-3, 0.158920570e-2, 0.245934259e-2, 0.900704920, -0.166626219e-1])

  #indices of wavelengths where the fit is valid
  indices = np.where((wavelength <= 2e-4))[0]

  Lambda = wavelength[indices] / 0.589e-4

  A = np.zeros(Lambda.size)
  A = delta * (a_coeff[0] + a_coeff[1]*delta + a_coeff[2]*theta + a_coeff[3]*Lambda**2*theta + a_coeff[4]*Lambda**-2
               + a_coeff[5] / (Lambda**2 - lambda_uv**2) + a_coeff[6] / (Lambda**2 - lambda_ir**2) + a_coeff[7]*delta**2)

  refractive_index = np.zeros(wavelength.size)
  refractive_index[indices] = ((2 * A + 1)/(1 - A))**0.5

  king_correction = (6 + 3 * 3e-4) / (6 - 7 * 3e-4)


  cross_section = generalRayleighCrossSection(refractive_index, king_correction, reference_density, wavelength, 2e-4)

  return cross_section



def rayleighH2(wavelength):
  """ calculates the refractive index of H2 """

  king_correction = 1.0
  reference_density = 2.65163e19

  refractive_index = 13.58e-5 * (1.0 + 7.52e-11 * wavelength**-2) + 1.0

  cross_section = generalRayleighCrossSection(refractive_index, king_correction, reference_density, wavelength, 2e-4)

  return cross_section


def rayleighHe(wavelength):
  """ calculates the refractive index of He """

  king_correction = 1.0
  reference_density = 2.546899e19

  refractive_index = 1e-8 * (2283 + 1.8102e13/(1.5342e10 - wavelength**-2)) + 1.0

  cross_section = generalRayleighCrossSection(refractive_index, king_correction, reference_density, wavelength, 2e-4)

  return cross_section



#Rayleigh scattering of atomic H, taken from Lee & Kim 2004
#Note: this is the low-energy expansion, valid for wavelengths larger than about 0.17 micron
def rayleighH(wavelength):
  """ calculates the refractive index of H """

  lambda_lyman = 0.0912*1e-4  #wavelength of the Lyman limit in cm

  lambda_fraction = lambda_lyman / wavelength

  rayleigh_indices = np.where(wavelength <= 2e-4)[0]

  cross_section = np.zeros(wavelength.size)
  cross_section[rayleigh_indices] = 8.41e-25 * lambda_fraction[rayleigh_indices]**4 + 3.37e-24 * lambda_fraction[rayleigh_indices]**6 + 4.71e-22 * lambda_fraction[rayleigh_indices]**14   #in cm2

  return cross_section



def rayleighN2(wavelength):
  """ calculates the refractive index of N2 """
  wavenumber = 1.0/wavelength

  king_correction = 1.034 + 3.17e-12 * wavenumber
  reference_density = 2.546899e19

  #the two wavenumber ranges for the fit of the refractive index
  lower_indices = np.where(wavenumber <= 21360)[0]
  upper_indices = np.where(wavenumber > 21360)[0]

  refractive_index = np.zeros(wavelength.size)

  refractive_index[lower_indices] = 1e-8 * (6498.2 + 307.4335e12/(14.4e9 - wavenumber[lower_indices]**2)) + 1.0
  refractive_index[upper_indices] = 1e-8 * (5677.465 + 318.81874e12/(14.4e9 - wavenumber[upper_indices]**2)) + 1.0

  cross_section = generalRayleighCrossSection(refractive_index, king_correction, reference_density, wavelength, 2e-4)

  return cross_section



def rayleighCO2(wavelength):
  """ calculates the refractive index of CO2 """

  king_correction = 1.1364 + 25.3e-12 * wavelength**-2
  reference_density = 2.546899e19

  refractive_index = (5799.25 / (128908.9**2 - wavelength**-2) + 120.05 / (89223.8**2 - wavelength**-2)
                   + 5.3334 / (75037.5**2 - wavelength**-2) + 4.3244 / (67837.7**2 - wavelength**-2)
                   + 0.1218145e-6 / (2418.136**2 - wavelength**-2)) * 1.1427e3 + 1.0

  cross_section = generalRayleighCrossSection(refractive_index, king_correction, reference_density, wavelength, 2e-4)

  return cross_section



def rayleighCO(wavelength):
  """ calculates the refractive index of CO """

  king_correction = 1.0
  reference_density = 2.546899e19

  refractive_index = 1e-8 * (22851 + 0.456e14 / (71427**2 - wavelength**-2)) + 1.0

  cross_section = generalRayleighCrossSection(refractive_index, king_correction, reference_density, wavelength, 2e-4)

  return cross_section


def rayleighCH4(wavelength):
  """ calculates the refractive index of CH4 """

  king_correction = 1.0
  reference_density = 2.546899e19

  wavenumber = 1.0/wavelength

  refractive_index = 1e-8 * (46662.0 + 4.02e-6 * wavenumber**2) + 1.0

  cross_section = generalRayleighCrossSection(refractive_index, king_correction, reference_density, wavelength, 2e-4)

  return cross_section


def rayleighO2(wavelength):
  """ calculates the refractive index of O2 """

  king_correction = 1.09 + 1.385e-11 * wavelength**-2 + 1.448e-20 * wavelength**-4
  reference_density = 2.68678e19

  refractive_index = 1e-8 * (20564.8 + 2.480899e13/(4.09e9 - wavelength**-2)) + 1.0

  cross_section = generalRayleighCrossSection(refractive_index, king_correction, reference_density, wavelength, 2e-4)

  return cross_section




#calculates the general Rayleigh scattering cross section
#assumes that the wavelengths are given in units of cm
def generalRayleighCrossSection(refractive_index, king_correction, reference_density, wavelength_full, lambda_limit):

  cross_section = np.zeros(wavelength_full.size)

  #indices and wavelengths where the Rayleigh cross section should be calculated
  indices = np.where((wavelength_full <= lambda_limit))[0]
  wavelength = wavelength_full[indices]

  #the square of the refractive index at the required wavelengths
  r = refractive_index[indices]**2

  #the pre-factor
  if type(king_correction) == float:
    prefactor = 24.0 * np.pi**3 / reference_density**2 * king_correction
  else:
    prefactor = 24.0 * np.pi**3 / reference_density**2 * king_correction[indices]

  cross_section[indices] = prefactor / wavelength**4 * ((r - 1.0) / (r + 2.0))**2


  return cross_section

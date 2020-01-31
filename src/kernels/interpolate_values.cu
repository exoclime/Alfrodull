#include "interpolate_values.h"

#include "physics_constants.h"


// temperature interpolation for the non-isothermal layers
__global__ void interpolate_temperature(double* tlay, double* tint, int numinterfaces) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (0 < i && i < numinterfaces - 1) {
        tint[i] = tlay[i - 1] + 0.5 * (tlay[i] - tlay[i - 1]);
    }
    if (i == 0) {
        tint[i] = tlay[i] - 0.5 * (tlay[i + 1] - tlay[i]);
    }
    if (i == numinterfaces - 1) {
        tint[i] = tlay[i - 1] + 0.5 * (tlay[i - 1] - tlay[i - 2]);
    }
}


// interpolates the Planck function for the layer temperatures from the pre-tabulated values
__global__ void planck_interpol_layer(double* temp,            // in
                                      double* planckband_lay,  // out 
                                      double* planck_grid,     // in
                                      double* starflux,        // in
                                      int     realstar,
                                      int     numlayers,
                                      int     nwave,
                                      double  T_surf,
                                      int     dim,
                                      int     step) {

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < nwave) {

        if (i < numlayers) {

            planckband_lay[i + x * (numlayers + 2)] = 0.0;

            double t = (temp[i] - 1.0) / step;

            t = max(0.001, min(dim - 1.001, t));

            int tdown = floor(t);
            int tup   = ceil(t);

            if (tdown != tup) {
                planckband_lay[i + x * (numlayers + 2)] =
                    planck_grid[x + tdown * nwave] * (tup - t)
                    + planck_grid[x + tup * nwave] * (t - tdown);
            }
            if (tdown == tup) {
                planckband_lay[i + x * (numlayers + 2)] = planck_grid[x + tdown * nwave];
            }
        }
        // taking stellar and internal temperatures
        if (i == numlayers) {
            if (realstar == 1) {
                planckband_lay[i + x * (numlayers + 2)] = starflux[x] / PI;
            }
            else {
                planckband_lay[i + x * (numlayers + 2)] = planck_grid[x + dim * nwave];
            }
        }
    }
}


// interpolates the Planck function for the interface temperatures from the pre-tabulated values
__global__ void planck_interpol_interface(double* temp,
                                          double* planckband_int,
                                          double* planck_grid,
                                          int     numinterfaces,
                                          int     nwave,
                                          int     dim,
                                          int     step) {

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < nwave && i < numinterfaces) {

        planckband_int[i + x * numinterfaces] = 0.0;

        double t = (temp[i] - 1.0) / step;

        t = max(0.001, min(dim - 1.001, t));

        int tdown = floor(t);
        int tup   = ceil(t);

        if (tdown != tup) {
            planckband_int[i + x * numinterfaces] = planck_grid[x + tdown * nwave] * (tup - t)
                                                    + planck_grid[x + tup * nwave] * (t - tdown);
        }
        if (tdown == tup) {
            planckband_int[i + x * numinterfaces] = planck_grid[x + tdown * nwave];
        }
    }
}


// interpolate layer and interface opacities from opacity table
// Note - US: this doesn't care about geometry, only pressure and temperature. works on a list of T and P for each wavelength bin - can be abstracted.
__global__ void interpolate_opacities(
    double* temp,        // in, layer temperature
    double* opactemp,    // in, opac reference table temperatures
    double* press,       // in, layer pressure
    double* opacpress,   // in, opac reference table pressure
    double* ktable,      // in, reference opacity table
    double* opac,        // out, opacities
    double* crosstable,  // in, reference scatter cross-section
    double* scat_cross,  // out, scattering cross-sections
    int     npress,      // in, ref table pressures count
    int     ntemp,       // in, ref table temperatures count
    int     ny,          // in, ref table gaussians y-points count
    int     nbin,        // in, ref wavelength bin count
    double  opaclimit,   // in, opacity limit for max cutoff for low wavelength bin idx
    int     nlay_or_nint // in, number of layer or interfaces (physical position bins count)
) {

    int x = threadIdx.x + blockIdx.x * blockDim.x; // wavelength bin index
    int i = threadIdx.y + blockIdx.y * blockDim.y; // volume element (layer or interface) bin index

    if (x < nbin && i < nlay_or_nint) {

        int x_1micron = lrint(nbin * 2.0 / 3.0);

        double deltaopactemp = (opactemp[ntemp - 1] - opactemp[0]) / (ntemp - 1.0);
        double deltaopacpress =
            (log10(opacpress[npress - 1]) - log10(opacpress[0])) / (npress - 1.0);
        double t = (temp[i] - opactemp[0]) / deltaopactemp;

        t = min(ntemp - 1.001, max(0.001, t));

        int tdown = floor(t);
        int tup   = ceil(t);

        double p = (log10(press[i]) - log10(opacpress[0])) / deltaopacpress;

        // do the cloud deck
        double k_cloud = 0.0; //1e-1 * norm_pdf(log10(press[i]),0,1);

        p = min(npress - 1.001, max(0.001, p));

        int pdown = floor(p);
        int pup   = ceil(p);

        if (pdown != pup && tdown != tup) {
            for (int y = 0; y < ny; y++) {
                double interpolated_opac =
                    ktable[y + ny * x + ny * nbin * pdown + ny * nbin * npress * tdown] * (pup - p)
                        * (tup - t)
                    + ktable[y + ny * x + ny * nbin * pup + ny * nbin * npress * tdown]
                          * (p - pdown) * (tup - t)
                    + ktable[y + ny * x + ny * nbin * pdown + ny * nbin * npress * tup] * (pup - p)
                          * (t - tdown)
                    + ktable[y + ny * x + ny * nbin * pup + ny * nbin * npress * tup] * (p - pdown)
                          * (t - tdown);

                if (x < x_1micron) {
                    opac[y + ny * x + ny * nbin * i] = max(interpolated_opac, opaclimit);
                }
                else {
                    opac[y + ny * x + ny * nbin * i] = interpolated_opac;
                }

                opac[y + ny * x + ny * nbin * i] += k_cloud;
            }

            scat_cross[x + nbin * i] =
                crosstable[x + nbin * pdown + nbin * npress * tdown] * (pup - p) * (tup - t)
                + crosstable[x + nbin * pup + nbin * npress * tdown] * (p - pdown) * (tup - t)
                + crosstable[x + nbin * pdown + nbin * npress * tup] * (pup - p) * (t - tdown)
                + crosstable[x + nbin * pup + nbin * npress * tup] * (p - pdown) * (t - tdown);
        }

        if (tdown == tup && pdown != pup) {
            for (int y = 0; y < ny; y++) {
                double interpolated_opac =
                    ktable[y + ny * x + ny * nbin * pdown + ny * nbin * npress * tdown] * (pup - p)
                    + ktable[y + ny * x + ny * nbin * pup + ny * nbin * npress * tdown]
                          * (p - pdown);
                if (x < x_1micron) {
                    opac[y + ny * x + ny * nbin * i] = max(interpolated_opac, opaclimit);
                }
                else {
                    opac[y + ny * x + ny * nbin * i] = interpolated_opac;
                }

                opac[y + ny * x + ny * nbin * i] += k_cloud;
            }

            scat_cross[x + nbin * i] =
                crosstable[x + nbin * pdown + nbin * npress * tdown] * (pup - p)
                + crosstable[x + nbin * pup + nbin * npress * tdown] * (p - pdown);
        }

        if (pdown == pup && tdown != tup) {
            for (int y = 0; y < ny; y++) {
                double interpolated_opac =
                    ktable[y + ny * x + ny * nbin * pdown + ny * nbin * npress * tdown] * (tup - t)
                    + ktable[y + ny * x + ny * nbin * pdown + ny * nbin * npress * tup]
                          * (t - tdown);
                if (x < x_1micron) {
                    opac[y + ny * x + ny * nbin * i] = max(interpolated_opac, opaclimit);
                }
                else {
                    opac[y + ny * x + ny * nbin * i] = interpolated_opac;
                }

                opac[y + ny * x + ny * nbin * i] += k_cloud;
            }

            scat_cross[x + nbin * i] =
                crosstable[x + nbin * pdown + nbin * npress * tdown] * (tup - t)
                + crosstable[x + nbin * pdown + nbin * npress * tup] * (t - tdown);
        }

        if (tdown == tup && pdown == pup) {
            for (int y = 0; y < ny; y++) {

                double interpolated_opac =
                    ktable[y + ny * x + ny * nbin * pdown + ny * nbin * npress * tdown];

                if (x < x_1micron) {
                    opac[y + ny * x + ny * nbin * i] = max(interpolated_opac, opaclimit);
                }
                else {
                    opac[y + ny * x + ny * nbin * i] = interpolated_opac;
                }

                opac[y + ny * x + ny * nbin * i] += k_cloud;
            }

            scat_cross[x + nbin * i] = crosstable[x + nbin * pdown + nbin * npress * tdown];
        }
    }
}


// interpolate the mean molecular mass for each layer
__global__ void meanmolmass_interpol(double* temp,
                                     double* opactemp,
                                     double* meanmolmass,
                                     double* opac_meanmass,
                                     double* press,
                                     double* opacpress,
                                     int     npress,
                                     int     ntemp,
                                     int     ninterface) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < ninterface) {

        double deltaopactemp = (opactemp[ntemp - 1] - opactemp[0]) / (ntemp - 1.0);
        double deltaopacpress =
            (log10(opacpress[npress - 1]) - log10(opacpress[0])) / (npress - 1.0);
        double t = (temp[i] - opactemp[0]) / deltaopactemp;

        t = min(ntemp - 1.001, max(0.001, t));

        int tdown = floor(t);
        int tup   = ceil(t);

        double p = (log10(press[i]) - log10(opacpress[0])) / deltaopacpress;

        p = min(npress - 1.001, max(0.001, p));

        int pdown = floor(p);
        int pup   = ceil(p);

        if (tdown != tup && pdown != pup) {
            meanmolmass[i] = opac_meanmass[pdown + npress * tdown] * (pup - p) * (tup - t)
                             + opac_meanmass[pup + npress * tdown] * (p - pdown) * (tup - t)
                             + opac_meanmass[pdown + npress * tup] * (pup - p) * (t - tdown)
                             + opac_meanmass[pup + npress * tup] * (p - pdown) * (t - tdown);
        }
        if (tdown != tup && pdown == pup) {
            meanmolmass[i] = opac_meanmass[pdown + npress * tdown] * (tup - t)
                             + opac_meanmass[pdown + npress * tup] * (t - tdown);
        }
        if (tdown == tup && pdown != pup) {
            meanmolmass[i] = opac_meanmass[pdown + npress * tdown] * (pup - p)
                             + opac_meanmass[pup + npress * tdown] * (p - pdown);
        }
        if (tdown == tup && pdown == pup) {
            meanmolmass[i] = opac_meanmass[pdown + npress * tdown];
        }
    }
}



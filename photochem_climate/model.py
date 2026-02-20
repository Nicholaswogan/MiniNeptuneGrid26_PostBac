from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence, Tuple
from tempfile import NamedTemporaryFile
import yaml

import numpy as np
from scipy import special
from astropy import constants

from clima import AdiabatClimate, ClimaException
from photochem.utils import stars
from photochem.utils import settings_dict_for_climate
from photochem.equilibrate import ChemEquiAnalysis
from photochem.extensions import gasgiants

CURRENT_DIR = Path(__file__).resolve().parent

class AdiabatClimateEquilibrium(AdiabatClimate):

    def __init__(
        self, 
        M_planet, 
        R_planet, 
        Teff,
        *,
        P_bottom=1e9,
        P_top=1.0,
        P_ref=1.0e3,
        nz=50, 
        number_of_zeniths=4,
        data_dir='PICASO',
        double_radiative_grid=True
    ) -> None:

        # Prepare settings file
        opacities = {
            'k-distributions': True, 
            'CIA': True, 
            'rayleigh': True, 
            'photolysis-xs': True
        }
        settings_dict = settings_dict_for_climate(
            planet_mass=float(M_planet*constants.M_earth.to('g').value), 
            planet_radius=float(R_planet*constants.R_earth.to('cm').value), 
            surface_albedo=0.0, 
            number_of_layers=int(nz), 
            number_of_zenith_angles=int(number_of_zeniths), 
            photon_scale_factor=1.0, 
            opacities=opacities
        )

        # Prepare stellar spectrum
        stellar_flux = 1360.0
        wv_planet, F_planet = blackbody_spectrum_at_planet(stellar_flux, Teff, nw=5000)
        flux_str = stars.photochem_spectrum_string(wv_planet, F_planet, scale_to_planet=False)

        # Data dir
        if data_dir == 'PICASO':
            data_dir = str(CURRENT_DIR / "picaso_opacities")
        
        # Initialize
        with NamedTemporaryFile('w') as f_settings:
            yaml.safe_dump(settings_dict, f_settings)
            with NamedTemporaryFile('w') as f_flux:
                f_flux.write(flux_str)
                f_flux.flush()
                super().__init__(
                    species_file=str(CURRENT_DIR / "sonora_climate.yaml"), 
                    settings_file=f_settings.name, 
                    flux_file=f_flux.name,
                    data_dir=data_dir,
                    double_radiative_grid=double_radiative_grid
                )

        # Ensure bolometric flux is right
        self.rad.set_bolometric_flux(stellar_flux)

        # Change some settings
        self.reference_pressure = P_ref
        self.rad.has_hard_surface = False
        self.rad.surface_albedo = np.ones(self.rad.surface_albedo.shape[0])*0.0
        self.max_rc_iters = 50
        self.P_top = P_top

        # Equilibrium chemistry solver
        eqsolver = ChemEquiAnalysis(
            thermofile=str(CURRENT_DIR / "sonora_equilibrium.yaml")
        )
        # Change the atomic composition to Lodders (2020)
        molfracs_atoms_sun = eqsolver.molfracs_atoms_sun
        for i,atom in enumerate(eqsolver.atoms_names):
            molfracs_atoms_sun[i] = LODDERS2020_SUN_COMP[atom]
        molfracs_atoms_sun /= np.sum(molfracs_atoms_sun)
        eqsolver.molfracs_atoms_sun = molfracs_atoms_sun
        self.eqsolver = eqsolver

        # Other settings/stuff that is saved
        self.P_bottom = P_bottom
        self.planet_mass = float(M_planet*constants.M_earth.to('g').value)
        self.planet_radius = float(R_planet*constants.R_earth.to('cm').value)
        nz = len(self.T)
        P_grid = np.logspace(np.log10(P_bottom), np.log10(P_top), 2*nz+1)
        P_grid = np.append(P_grid[0], P_grid[1::2])
        self.P_grid = P_grid

    def initial_guess(self, stellar_flux, T_int, T_int_factor):

        Teq = stars.equilibrium_temperature(stellar_flux, 0.0) # in K
        grav = gasgiants.gravity(self.planet_radius, self.planet_mass, 0.0) # CGS units
        # self.P_grid is in dyn/cm^2; guillot_pt expects pressure in bar.
        P_grid_bar = self.P_grid/1.0e6
        T_guillot = guillot_pt(
            grav=grav,
            Teq=Teq,
            T_int=T_int*T_int_factor, # Better guess
            P_in=P_grid_bar,
        )
        return np.clip(T_guillot, a_min=50.0, a_max=5000.0)
        
    def g_eval(self, T, stellar_flux, T_int, metallicity, CtoO, convecting_with_below=None):

        # Compute equilibrium chemistry
        gases, condensates = equilibrate_atmosphere(self.eqsolver, self.P_grid/1e6, T, np.log10(metallicity), CtoO)

        # Copy to climate model
        P_i = np.ones(len(self.species_names))*1.0e-30
        custom_dry_mix = {'pressure': self.P_grid}
        for i,sp in enumerate(self.species_names):
            custom_dry_mix[sp] = np.maximum(gases[sp],1.0e-30)
            P_i[i] = np.maximum(gases[sp][0], 1.0e-30)*self.P_grid[0]
        assert np.isclose(np.sum(P_i), self.P_grid[0])

        # Set energy fluxes
        self.rad.set_bolometric_flux(stellar_flux)
        self.surface_heat_flow = stars.stefan_boltzmann(T_int)*1e3

        if convecting_with_below is None:
            convecting_with_below = self.convecting_with_below.copy()

        # Call RCE
        T_surf_guess = float(T[0])
        T_guess = T[1:].copy()

        self.convective_max_boundary_shift = 1
        self.max_rc_iters_convection = 5

        converged = False
        try:
            converged = self.RCE(P_i, T_surf_guess, T_guess, convecting_with_below, custom_dry_mix=custom_dry_mix)
        except ClimaException:
            converged = False

        if not converged:
            dt_increment_save = self.dt_increment
            self.dt_increment = 1.1
            try:
                converged = self.RCE(P_i, T_surf_guess, T_guess, convecting_with_below, custom_dry_mix=custom_dry_mix)
            except ClimaException:
                converged = False
            self.dt_increment = dt_increment_save
        
        # if not converged:
        #     self.dt_increment = 1.1
        #     convecting_with_below[:] = False
        #     convecting_with_below[:10] = True
        #     converged = self.RCE(P_i, T_surf_guess, T_guess, convecting_with_below, custom_dry_mix=custom_dry_mix)
        #     assert converged
        
        difference = self.lapse_rate - self.lapse_rate_intended
        if np.any(difference > np.maximum(1e-3, 2.0e-2*np.abs(self.lapse_rate_intended))):
            self.max_rc_iters_convection = -1
            self.convective_max_boundary_shift = 1
            converged = self.RCE(P_i, self.T_surf, self.T, self.convecting_with_below, custom_dry_mix=custom_dry_mix)
            assert converged

        return np.append(self.T_surf, self.T)


    def solve(self, stellar_flux, T_int, metallicity, CtoO, *, T_int_factor=1.0, tol=2.0, max_tol=5.0, **kwargs):

        convecting_with_below_init = self.convecting_with_below.copy()
        convecting_with_below_init[:] = False
        if T_int < 1:
            pass
        elif 1 <= T_int < 100:
            convecting_with_below_init[0] = True
        elif 100 <= T_int < 200:
            convecting_with_below_init[:3] = True
        else:
            convecting_with_below_init[:10] = True

        first_call = True
        def g(T):
            nonlocal first_call
            convecting_with_below = None
            if first_call:
                convecting_with_below = convecting_with_below_init
            T_result = self.g_eval(T, stellar_flux, T_int, metallicity, CtoO, convecting_with_below)
            first_call = False
            return T_result
        
        solver = RobustFixedPointSolver(
            g=g,
            x0=self.initial_guess(stellar_flux, T_int, T_int_factor),
            tol=tol,
            max_tol=max_tol,
            **kwargs
        )
        result = solver.solve()

        return result
    
    def return_atmosphere(self):
        f_i = np.concatenate((self.f_i_surf.reshape((1,len(self.f_i_surf))),self.f_i))
        P = np.append(self.P_surf, self.P)
        T = np.append(self.T_surf, self.T)

        res = {
            'pressure': P/1e6, # bar
            'temperature': T, # K
        }
        for i,sp in enumerate(self.species_names):
            res[sp] = f_i[:,i]

        return res

def guillot_pt(grav, Teq, T_int, P_in, logg1=-1, logKir=-1.5, alpha=0.5):

    kv1, kv2 =10.**(logg1+logKir),10.**(logg1+logKir)
    kth=10.**logKir

    Teff = T_int
    f = 1.0  # solar re-radiation factor
    A = 0.0  # planetary albedo
    g0 = grav/100.0 #cm/s2 to m/s2

    # Compute equilibrium temperature and set up gamma's
    T0 = Teq
    gamma1 = kv1/kth #Eqn. 25
    gamma2 = kv2/kth

    # Initialize arrays
    logtau =np.arange(-10,20,.1)
    tau =10**logtau

    #computing temperature
    T4ir = 0.75*(Teff**(4.))*(tau+(2.0/3.0))
    f1 = 2.0/3.0 + 2.0/(3.0*gamma1)*(1.+(gamma1*tau/2.0-1.0)*np.exp(-gamma1*tau))+2.0*gamma1/3.0*(1.0-tau**2.0/2.0)*special.expn(2.0,gamma1*tau)
    f2 = 2.0/3.0 + 2.0/(3.0*gamma2)*(1.+(gamma2*tau/2.0-1.0)*np.exp(-gamma2*tau))+2.0*gamma2/3.0*(1.0-tau**2.0/2.0)*special.expn(2.0,gamma2*tau)
    T4v1=f*0.75*T0**4.0*(1.0-alpha)*f1
    T4v2=f*0.75*T0**4.0*alpha*f2
    T=(T4ir+T4v1+T4v2)**(0.25)
    P=tau*g0/(kth*0.1)/1.E5

    T = np.interp(np.log10(P_in), np.log10(P), T)

    return T

def equilibrate_atmosphere(eqsolver, P, T, log10mh, CtoO_relative):
    """Solve for equilibrium chemistry across a vertical profile.

    Parameters
    ----------
    P : ndarray
        Pressures in bar.
    T : ndarray
        Temperatures in Kelvin aligned with `P`.
    log10mh : float
        log10 metallicity relative to solar.
    CtoO_relative : float
        The C/O ratio relative to solar.

    Returns
    -------
    tuple(dict, dict)
        Gas and condensate mole fraction dictionaries keyed by species
        name.
    """

    # Check inputs
    if not isinstance(P, np.ndarray):
        raise ValueError('`P` should be a numpy array.')
    if not isinstance(P, np.ndarray):
        raise ValueError('`T` should be a numpy array.')

    # Some conversions and copies
    P_cgs = P*1e6
    metallicity = 10.0**log10mh
    gas_names = eqsolver.gas_names
    condensate_names = eqsolver.condensate_names

    gases = {}
    for key in gas_names:
        gases[key] = np.empty(len(P))
    condensates = {}
    for key in condensate_names:
        condensates[key] = np.empty(len(P))

    for i in range(len(P)):
        if i > 0:
            eqsolver.use_prev_guess = True
        # Try many perturbations on T to try to get convergence
        for eps in [0.0, 1.0e-12, -1.0e-12, 1.0e-8, -1.0e-8, 1.0e-6, -1.0e-6, 1.0e-4, -1.0e-4]:
            converged = eqsolver.solve_metallicity(P_cgs[i], T[i] + T[i]*eps, metallicity, CtoO_relative)
            if converged:
                break
        if not converged:
            # We will not enforce convergence.
            pass
        molfracs_species_gas = eqsolver.molfracs_species_gas
        molfracs_species_condensate = eqsolver.molfracs_species_condensate
        for j,key in enumerate(gas_names):
            gases[key][i] = molfracs_species_gas[j]
        for j,key in enumerate(condensate_names):
            condensates[key][i] = molfracs_species_condensate[j]
    eqsolver.use_prev_guess = False

    return gases, condensates


# The composition of the Sun from Lodders (2020)
LODDERS2020_SUN_COMP = {
    'H':  9.082387E-01,
    'He': 9.046346E-02,
    'Li': 2.050745E-09,
    'C':  3.286959E-04,
    'N':  7.893027E-05,
    'O':  5.982842E-04,
    'F':  4.577235E-08,
    'Na': 2.083182E-06,
    'Mg': 3.712245E-05,
    'Si': 3.604122E-05,
    'P':  2.977005E-07,
    'S':  1.575001E-05,
    'Cl': 1.906580E-07,
    'K':  1.301448E-07,
    'Ti': 8.862536E-08,
    'V':  9.911335E-09,
    'Cr': 4.732212E-07,
    'Fe': 3.142794E-05,
    'Rb': 2.584155E-10,
    'Cs': 1.326317E-11,
}

def blackbody_spectrum_at_planet(stellar_flux, Teff, nw):

    # Blackbody
    wv_planet = np.logspace(np.log10(0.1), np.log10(100), nw)*1e3 # nm
    F_planet = stars.blackbody(Teff, wv_planet)*np.pi

    # Rescale so that it has the proper stellar flux for the planet
    factor = stellar_flux/stars.energy_in_spectrum(wv_planet, F_planet)
    F_planet *= factor

    return wv_planet, F_planet

@dataclass
class SolveResult:
    x: np.ndarray
    converged: bool
    iters: int
    func_evals: int
    # (k, x_k, r_k, ||r_k||_scaled, omega_k, beta_k)
    history: List[Tuple[int, np.ndarray, np.ndarray, float, float, float]]


class RobustFixedPointSolver:
    """
    Robust Anderson-accelerated fixed-point solver.

    This class solves

    .. math::
        x = g(x)

    by iterating on the residual

    .. math::
        r(x) = g(x) - x.

    Notes
    -----
    Per iteration, the algorithm:
    1. Evaluates ``g(x)`` once.
    2. Forms a relaxed fixed-point proposal ``x_plain = x + omega * r``.
    3. Builds an Anderson candidate from recent ``(x, r)`` history.
    4. Mixes using ``beta`` and applies a directional safeguard.
    5. Applies optional step limits and updates adaptive ``omega``/``beta``.
    """

    def __init__(
        self,
        g: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        *,
        m: int = 6,
        omega: float = 0.5,
        omega_min: float = 0.05,
        omega_max: float = 1.0,
        omega_shrink: float = 0.5,
        omega_grow: float = 1.2,
        beta: float = 1.0,
        beta_min: float = 0.1,
        beta_shrink: float = 0.5,
        beta_grow: float = 1.1,
        ridge: float = 1e-6,
        max_step: float | None = None,
        max_norm_step: float | None = None,
        growth_threshold: float = 2.0,
        improve_threshold: float = 0.5,
        safeguard_factor: float = 1.2,
        scale: float | Sequence[float] = 1.0,
        tol: float = 1e-8,
        max_tol: float | None = None,
        max_iter: int = 80,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the robust Anderson fixed-point solver.

        Parameters
        ----------
        g : callable
            Fixed-point map. Must take and return a 1D NumPy array of the same
            shape.
        x0 : ndarray
            Initial guess as a 1D NumPy array. Use a length-1 array for scalar
            problems.
        m : int, optional
            Anderson memory length.
        omega : float, optional
            Initial under-relaxation for the plain fixed-point step.
        omega_min, omega_max : float, optional
            Bounds for adaptive ``omega``.
        omega_shrink, omega_grow : float, optional
            Multipliers used when residual worsens/improves.
        beta : float, optional
            Initial blend between plain and Anderson proposals.
        beta_min : float, optional
            Lower bound for adaptive ``beta``.
        beta_shrink, beta_grow : float, optional
            Multipliers used when residual worsens/improves.
        ridge : float, optional
            Ridge regularization in the Anderson least-squares system.
        max_step : float or None, optional
            Optional per-component cap on ``x_{k+1}-x_k``.
        max_norm_step : float or None, optional
            Optional cap on scaled RMS step norm.
        growth_threshold : float, optional
            If residual grows above this ratio, shrink damping and restart
            Anderson history.
        improve_threshold : float, optional
            If residual shrinks below this ratio, cautiously grow damping.
        safeguard_factor : float, optional
            Directional safeguard aggressiveness; values near 1 are more
            conservative.
        scale : float or array_like, optional
            Scaling used in residual/step RMS norms.
        tol : float, optional
            Convergence tolerance on scaled RMS residual norm.
        max_tol : float or None, optional
            Optional max per-component scaled residual tolerance.
        max_iter : int, optional
            Maximum number of iterations.
        verbose : bool, optional
            If True, prints per-iteration diagnostics.
        """
        if m < 0:
            raise ValueError("m must be >= 0")
        if not (0.0 < omega_min <= omega <= omega_max <= 1.0):
            raise ValueError("require 0 < omega_min <= omega <= omega_max <= 1")
        if beta <= 0.0:
            raise ValueError("beta must be > 0")
        if beta_min <= 0.0:
            raise ValueError("beta_min must be > 0")
        if ridge < 0.0:
            raise ValueError("ridge must be >= 0")
        if max_step is not None and max_step <= 0.0:
            raise ValueError("max_step must be > 0 or None")
        if max_norm_step is not None and max_norm_step <= 0.0:
            raise ValueError("max_norm_step must be > 0 or None")
        if safeguard_factor < 1.0:
            raise ValueError("safeguard_factor must be >= 1.0")
        if max_tol is not None and max_tol <= 0.0:
            raise ValueError("max_tol must be > 0 or None")
        if not isinstance(x0, np.ndarray):
            raise TypeError("x0 must be a numpy.ndarray (use a length-1 array for scalars)")

        x = np.asarray(x0, dtype=float)
        if x.ndim != 1:
            raise ValueError("x0 must be a 1D numpy array")
        sc = (
            np.full_like(x, float(scale), dtype=float)
            if isinstance(scale, (int, float))
            else np.asarray(scale, dtype=float)
        )
        if sc.shape != x.shape:
            raise ValueError("scale must be a scalar or have the same shape as x0")
        if np.any(sc <= 0.0):
            raise ValueError("scale entries must be > 0")

        self.g = g
        self.x = x
        self.sc = sc
        self.m = int(m)
        self.omega = float(omega)
        self.omega_min = float(omega_min)
        self.omega_max = float(omega_max)
        self.omega_shrink = float(omega_shrink)
        self.omega_grow = float(omega_grow)
        self.beta = float(beta)
        self.beta_min = float(beta_min)
        self.beta_shrink = float(beta_shrink)
        self.beta_grow = float(beta_grow)
        self.ridge = float(ridge)
        self.max_step = max_step
        self.max_norm_step = max_norm_step
        self.growth_threshold = float(growth_threshold)
        self.improve_threshold = float(improve_threshold)
        self.safeguard_factor = float(safeguard_factor)
        self.tol = float(tol)
        self.max_tol = max_tol
        self.max_iter = int(max_iter)
        self.verbose = bool(verbose)

        self.history: List[Tuple[int, np.ndarray, np.ndarray, float, float, float]] = []
        self.func_evals = 0
        self.xs: List[np.ndarray] = []
        self.rs: List[np.ndarray] = []
        self.prev_norm: float | None = None
        self.prev_x: np.ndarray | None = None
        self.prev_r: np.ndarray | None = None
        self.k = 0
        self.converged = False

    def _rms_scaled(self, v: np.ndarray) -> float:
        vs = v / self.sc
        return float(np.linalg.norm(vs) / (max(1, vs.size) ** 0.5))

    def _max_scaled(self, v: np.ndarray) -> float:
        return float(np.max(np.abs(v / self.sc)))

    def step(self) -> bool:
        """
        Execute one solver iteration.

        Returns
        -------
        bool
            True if the solver is converged after this call, else False.
        """
        if self.converged or self.k >= self.max_iter:
            return self.converged

        k = self.k
        # One expensive model call: evaluate the fixed-point map at the current state.
        gx = np.asarray(self.g(self.x), dtype=float)
        if gx.shape != self.x.shape:
            raise ValueError("g(x) must return a numpy array with the same shape as x0")
        self.func_evals += 1

        # Residual for the root-equivalent problem r(x) = g(x) - x.
        xk = self.x.copy()
        r = gx - self.x
        rnorm = self._rms_scaled(r)
        rmax = self._max_scaled(r)

        # Convergence test: scaled RMS criterion plus optional per-component max criterion.
        if rnorm < self.tol and (self.max_tol is None or rmax < self.max_tol):
            self.history.append((k, xk, r.copy(), rnorm, self.omega, self.beta))
            if self.verbose:
                print(
                    f"[AA] k={k:3d}  rnorm={rnorm: .3e}  rmax={rmax: .3e}  "
                    f"omega={self.omega: .3f}  beta={self.beta: .3f}  (converged)"
                )
            self.converged = True
            return True

        # Adapt relaxation/mixing using residual progress relative to last iteration.
        # If things get worse, damp and restart AA history; if much better, grow cautiously.
        did_restart = False
        if self.prev_norm is not None and self.prev_norm > 0.0:
            ratio = rnorm / self.prev_norm
            if ratio > self.growth_threshold:
                self.omega = max(self.omega_min, self.omega * self.omega_shrink)
                self.beta = max(self.beta_min, self.beta * self.beta_shrink)
                self.xs.clear()
                self.rs.clear()
                self.prev_x = None
                self.prev_r = None
                did_restart = True
            elif ratio < self.improve_threshold:
                self.omega = min(self.omega_max, self.omega * self.omega_grow)
                self.beta = min(1.0, self.beta * self.beta_grow)
        self.prev_norm = rnorm

        # Append current point to limited-memory buffers used by Anderson mixing.
        self.xs.append(xk)
        self.rs.append(r.copy())

        # Baseline robust step: under-relaxed fixed-point update.
        omega_used = self.omega
        x_plain = self.x + self.omega * r

        # Accelerated step: Anderson type-I on recent residual/iterate differences.
        # Falls back to plain g(x) if there is not enough history.
        x_acc = gx.copy()
        mk_used = 0
        if self.m > 0 and len(self.xs) >= 2:
            mk = min(self.m, len(self.xs) - 1)
            mk_used = mk
            i0 = len(self.xs) - (mk + 1)
            x_win = self.xs[i0:]
            r_win = self.rs[i0:]

            dR = np.column_stack([r_win[j + 1] - r_win[j] for j in range(mk)])
            dX = np.column_stack([x_win[j + 1] - x_win[j] for j in range(mk)])
            dR_scaled = dR / self.sc[:, None]
            r_scaled = r / self.sc
            A = dR_scaled.T @ dR_scaled + self.ridge * np.eye(mk)
            b = dR_scaled.T @ r_scaled
            try:
                gamma = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                gamma = np.linalg.lstsq(A, b, rcond=None)[0]
            x_acc = gx - dX @ gamma

        # Blend baseline and accelerated candidates.
        x_next = x_plain + self.beta * (x_acc - x_plain)

        # Directional safeguard (no extra g-eval):
        # use a local secant slope estimate to reject overly aggressive AA directions.
        did_safeguard = False
        beta_before_safeguard = self.beta
        if self.prev_x is not None and self.prev_r is not None:
            last_dx = self.x - self.prev_x
            last_dr = r - self.prev_r
            dxs = last_dx / self.sc
            drs = last_dr / self.sc
            denom = float(dxs @ dxs)
            if denom > 0.0:
                alpha = float(drs @ dxs) / denom
                base = self._rms_scaled(r + alpha * (x_plain - self.x))
                beta_try = self.beta
                while beta_try >= self.beta_min:
                    x_try = x_plain + beta_try * (x_acc - x_plain)
                    pred = self._rms_scaled(r + alpha * (x_try - self.x))
                    if pred <= self.safeguard_factor * base:
                        x_next = x_try
                        self.beta = beta_try
                        break
                    beta_try *= self.beta_shrink
                else:
                    self.beta = self.beta_min
                    x_next = x_plain
                    self.xs.clear()
                    self.rs.clear()
                did_safeguard = self.beta != beta_before_safeguard

        # Optional hard step limits (componentwise and/or scaled RMS norm).
        beta_used = self.beta
        dx = x_next - self.x
        dx_before_clip = dx.copy()
        if self.max_step is not None:
            dx = np.clip(dx, -float(self.max_step), float(self.max_step))
        if self.max_norm_step is not None:
            nrm = self._rms_scaled(dx)
            cap = float(self.max_norm_step)
            if nrm > cap and nrm > 0.0:
                dx *= cap / nrm
        did_clip = not np.allclose(dx, dx_before_clip)

        # Commit the step and keep state needed for next-iteration safeguards.
        x_new = self.x + dx
        self.prev_x = self.x
        self.prev_r = r
        self.x = x_new

        # Store iteration diagnostics/history for analysis and plotting.
        self.history.append((k, xk, r.copy(), rnorm, omega_used, beta_used))
        dxnorm = self._rms_scaled(dx)
        if self.verbose:
            flags = []
            if mk_used == 0:
                flags.append("noAA")
            if did_restart:
                flags.append("restart")
            if did_safeguard:
                flags.append("safeguard")
            if did_clip:
                flags.append("clip")
            flag_str = ("  [" + ",".join(flags) + "]") if flags else ""
            print(
                f"[AA] k={k:3d}  rnorm={rnorm: .3e}  rmax={rmax: .3e}  dxnorm={dxnorm: .3e}  "
                f"omega={omega_used: .3f}  beta={beta_used: .3f}  mk={mk_used}{flag_str}"
            )

        self.k += 1
        return False

    def solve(self) -> SolveResult:
        """
        Run iterations until convergence or ``max_iter``.

        Returns
        -------
        SolveResult
            Final solver result and iteration history.
        """
        while not self.converged and self.k < self.max_iter:
            self.step()

        return SolveResult(
            x=self.x.copy(),
            converged=self.converged,
            iters=self.k,
            func_evals=self.func_evals,
            history=self.history,
        )

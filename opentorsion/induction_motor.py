import numpy as np
from scipy import linalg as LA

# from opentorsion.shaft_element import Shaft
# from opentorsion.disk_element import Disk
# from opentorsion.gear_element import Gear


class Induction_motor:
    """Induction motor object
    Arguments:
    ----------
    n: int
        Node position of the motor in the global coordinates
    omega: float
        Motor nominal speed [RPM]
    f: float
        Frequency of the stator current supply [Hz]
    p: int
        Number of pole-pairs in the motor
    voltage: float
        Stator voltage [V]
    circuit_parameters: list
        T-equivalent circuit parameters of the motor
    """

    def __init__(
        self,
        n,
        omega,
        f,
        p,
        voltage=4000,
        small_signal=False,
        circuit_parameters_nonlinear=[0.023457, 0.019480, 0.030470, 0.030030, 0.028904],
        circuit_parameters_linear=[0.023457, 0.019480, 0.030470, 0.030030, 0.028904],
    ):
        self.n = n
        self.omega = omega
        self.f = f
        self.p = p
        self.small_signal = small_signal

        # Calculations
        self.voltage = voltage * np.sqrt(2 / 3)
        self.omega_M = self.omega * 2 * np.pi / 60  # Mechanical rotational speed
        self.omega_m = p * self.omega_M  # Electrical rotation speed
        self.omega_s = 2 * np.pi * f  # Synchronous speed
        self.omega_r = (
            self.omega_s - self.omega_m
        )  # Rotor speed relative to synchronous coordinates
        self.s = (self.omega_s - self.omega_m) / self.omega_s  # Slip

        # nonlinear parameters
        self.circuit_parameters_nonlinear = circuit_parameters_nonlinear

        if small_signal:  # linear parameters
            self.circuit_parameters_linear = circuit_parameters_linear

        # Calculate steady-state currents
        self.i_s, self.i_r = self.get_currents()

    def get_currents(self):
        """Calculates the steady-state currents at an operating point. Calculation is based on the T-equivalent circuit model of the induction motor."""
        R_s, R_r, L_s, L_r, L_m = self.circuit_parameters_nonlinear
        p, f, omega_s, omega_0 = self.p, self.f, self.omega_s, self.omega_m
        u_s = self.voltage
        S = (omega_s - omega_0) / omega_s

        denominator = np.complex(
            -S * L_r * L_s * omega_s ** 2 + L_m ** 2 * omega_s ** 2 * S + R_r * R_s,
            S * L_r * R_s * omega_s + L_s * R_r * omega_s,
        )

        i_s = np.complex(R_r, L_r * S * omega_s) * u_s / denominator
        i_r = np.complex(0, -S * L_m * omega_s) * u_s / denominator

        i_sx, i_sy = i_s.real, i_s.imag
        i_rx, i_ry = i_r.real, i_r.imag

        return (i_sx, i_sy), (i_rx, i_ry)

    def R(self):
        """Resistance matrix"""

        R_s, R_r, L_s, L_r, L_m = self.circuit_parameters_nonlinear
        i_sx, i_sy = self.i_s
        i_rx, i_ry = self.i_r
        omega_s, omega, p = self.omega_s, self.omega_m, self.p

        # fmt: off
        R = [
            [                R_s,         -omega_s*L_s,                   0,         -omega_s*L_m, 0],
            [        omega_s*L_s,                  R_s,         omega_s*L_m,                    0, 0],
            [                  0, -(omega_s-omega)*L_m,                 R_r, -(omega_s-omega)*L_r, 0],
            [(omega_s-omega)*L_m,                    0, (omega_s-omega)*L_r,                  R_r, 0],
            [                  0,     -3/2*p*L_m*i_rx,                   0,        3/2*p*L_m*i_sx, 0],
        ]
        # fmt: on

        return R

    def L(self):
        """Inductance matrix"""
        R_s, R_r, L_s, L_r, L_m = self.circuit_parameters_nonlinear

        # fmt: off
        L = [
            [L_s,   0, L_m,   0, 0],
            [  0, L_s,   0, L_m, 0],
            [L_m,   0, L_r,   0, 0],
            [  0, L_m,   0, L_r, 0],
            [  0,   0,   0,   0, 0]
        ]
        # fmt: on

        return L

    def R_linear(self):
        """Linearized resistance matrix"""
        R_s, R_r, L_s, L_r, L_m = self.circuit_parameters_linear
        R_s0, R_r0, L_s0, L_r0, L_m0 = self.circuit_parameters_nonlinear
        i_sx0, i_sy0 = self.i_s
        i_rx0, i_ry0 = self.i_r
        omega_s, omega, p = self.omega_s, self.omega_m, self.p

        # fmt: off
        R = [
            [                R_s,         -omega_s*L_s,                   0,         -omega_s*L_m,                          0],
            [        omega_s*L_s,                  R_s,         omega_s*L_m,                    0,                          0],
            [                  0, -(omega_s-omega)*L_m,                 R_r, -(omega_s-omega)*L_r,  p*(L_m0*i_sy0+L_r0*i_ry0)],
            [(omega_s-omega)*L_m,                    0, (omega_s-omega)*L_r,                  R_r, -p*(L_m0*i_sx0+L_r0*i_rx0)],
            [   3/2*p*L_m0*i_ry0,    -3/2*p*L_m0*i_rx0,   -3/2*p*L_m0*i_sy0,     3/2*p*L_m0*i_sx0,                          0]
        ]
        # fmt: on

        return R

    def L_linear(self):
        """Linearized inductance matrix"""
        R_s, R_r, L_s, L_r, L_m = self.circuit_parameters_linear

        # fmt: off
        L = [
            [L_s,   0, L_m,   0, 0],
            [  0, L_s,   0, L_m, 0],
            [L_m,   0, L_r,   0, 0],
            [  0, L_m,   0, L_r, 0],
            [  0,   0,   0,   0, 0]
        ]
        # fmt: on

        return L


if __name__ == "__main__":
    pass

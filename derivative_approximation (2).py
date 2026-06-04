"""
Author: Alexander Huhn, Fedir Deineko
Date: 24.10.2025
"""

import matplotlib.pyplot as plt
import numpy as np

class FiniteDifference:
    """
    Represents the first and second order finite difference approximation
    of a function and allows for computation of the error compared to
    the exact analytical derivatives.

    Parameters
    ----------
    h : float
        Step size for finite difference approximation.
    f : callable
        Function to approximate the derivatives of. Callable as f(x).
    d_f : callable, optional
        Analytical first derivative of f.
    dd_f : callable, optional
        Analytical second derivative of f.
    """

    def __init__(self, h: float, f, d_f=None, dd_f=None):
        self.h = h
        self.f = f
        self.d_f = d_f
        self.dd_f = dd_f

    def compute_dh_right_f(self):
        """First derivative using the forward (right) finite difference."""
        def d_f_approx(x):
            return (self.f(x + self.h) - self.f(x)) / self.h
        return d_f_approx

    def compute_dh_left_f(self):
        """First derivative using the backward (left) finite difference."""
        def d_f_approx(x):
            return (self.f(x) - self.f(x - self.h)) / self.h
        return d_f_approx

    def compute_dh_central_f(self):
        """First derivative using the central finite difference."""
        def d_f_approx(x):
            return (self.f(x + self.h) - self.f(x - self.h)) / (2 * self.h)
        return d_f_approx

    def compute_ddh_f(self):
        """Second derivative using central finite difference."""
        def dd_f_approx(x):
            return (self.f(x + self.h) - 2 * self.f(x) + self.f(x - self.h)) / (self.h ** 2)
        return dd_f_approx

    def compute_r_interval(self, array, b: float):
        """Computes the accepted interval if h is added to x"""
        r_array = []
        for i in array:
            if i + self.h <= b:
                r_array.append(i)
        return r_array

    def compute_l_interval(self, array, a: float):
        """Computes the accepted interval if h is subtracted from x"""
        l_array = []
        for i in array:
            if i - self.h >= a:
                l_array.append(i)
        return l_array

    def compute_c_interval(self, array, a: float, b: float):
        """Computes the accepted interval if h is subtracted from x and is added to x"""
        c_array = []
        l_array = self.compute_l_interval(array, a)
        r_array = self.compute_r_interval(array, b)
        for i in r_array:
            if i in l_array:
                c_array.append(i)
        return c_array

    def compute_errors(self, a: float, b: float, p: int):
        """
        Computes the infinity norm of the error between numerical and analytical
        derivatives for a given interval [a, b].

        Parameters
        ----------
        a, b : float
            Start and end of the interval.
        p : int
            Number of subdivisions in the interval.

        Returns
        -------
        tuple of floats
            (error_d_right, error_d_central, error_d_left, error_dd)
        """
        if self.d_f is None or self.dd_f is None:
            raise ValueError("Analytical derivatives d_f and dd_f must be provided.")

        x_right = np.array(self.compute_r_interval([a + i * (b - a) / p for i in range(p + 1)], b))
        x_left = np.array(self.compute_l_interval([a + i * (b - a) / p for i in range(p + 1)], a))
        x_central = np.array(self.compute_c_interval([a+i * (b-a) / p for i in range(p + 1)], a, b))

        # Compute approximations
        d_right = self.compute_dh_right_f()(x_right)
        d_central = self.compute_dh_central_f()(x_central)
        d_left = self.compute_dh_left_f()(x_left)
        dd = self.compute_ddh_f()(x_central)

        # Compute errors
        err_d_right = np.max(np.abs(self.d_f(x_right) - d_right))
        err_d_central = np.max(np.abs(self.d_f(x_central) - d_central))
        err_d_left = np.max(np.abs(self.d_f(x_left) - d_left))
        err_dd = np.max(np.abs(self.dd_f(x_central) - dd))

        return err_d_right, err_d_central, err_d_left, err_dd

    def plot_functions(self, a, b, p):
        """Plots the function, its derivatives and their approximated derivatives"""
        df_approx_list = [self.compute_dh_right_f(), self.compute_dh_left_f(),
                          self.compute_dh_central_f(), self.compute_ddh_f()]

        x0_points = [a + i * (b - a) / p for i in range(p+1)]
        x_points = np.array(x0_points)

        f_points = np.array([self.f(i) for i in x0_points])
        x_points_list = [self.compute_r_interval(x0_points, b),
                         self.compute_l_interval(x0_points, a),
                         self.compute_c_interval(x0_points, a, b)]

        df_points_list = [np.array([df_approx_list[0](i) for i in x_points_list[0]]),
                          np.array([df_approx_list[1](i) for i in x_points_list[1]]),
                          np.array([df_approx_list[2](i) for i in x_points_list[2]]),
                          np.array([df_approx_list[3](i) for i in x_points_list[2]])
                          ]

        plt.plot(x_points, f_points, 'b--', label="f(x)")

        if self.d_f is not None:
            d_f_points = np.array([self.d_f(i) for i in x0_points])
            plt.plot(x_points, d_f_points, 'm--',label="f '(x)")

        if self.dd_f is not None:
            dd_f_points = np.array([self.dd_f(i) for i in x0_points])
            plt.plot(x_points, dd_f_points, 'r--',label="f ''(x)")

        plt.plot(x_points_list[0], df_points_list[0], 'k-',label="D1R f '(x)")
        plt.plot(x_points_list[1], df_points_list[1], 'y-',label="D1L f '(x)")
        plt.plot(x_points_list[2], df_points_list[2], 'c-',label="D1Z f '(x)")
        plt.plot(x_points_list[2], df_points_list[3], 'g-',label="D2 f ''(x)")
        plt.xlabel('x')
        plt.title(f"Die Ableitungen und ihre Approximation mit h ={self.h:.1e}")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_errors(self, a, b, p, h_array=None):
        """Plots the approximation errors for different step sizes h"""
        if h_array is None:
            h_array = [self.h]
        errors_1 = [[],[],[],[]]
        for h in h_array:
            self.h = h
            errors = self.compute_errors(a, b, p)
            errors_1[0].append(errors[0])
            errors_1[1].append(errors[1])
            errors_1[2].append(errors[2])
            errors_1[3].append(errors[3])

        err_d_right_points = np.array(list(reversed(errors_1[0])))
        err_d_central_points = np.array(list(reversed(errors_1[1])))
        err_d_left_points = np.array(list(reversed(errors_1[2])))
        err_dd_points = np.array(list(reversed(errors_1[3])))
        plt.xscale('log')
        plt.yscale('log')
        x = np.array(list(reversed(h_array)))
        plt.plot(x, err_d_left_points, 'Pk',label="Fehler von D1L f '(x)")
        plt.plot(x, err_d_central_points, '*g',label="Fehler von D1Z f '(x)")
        plt.plot(x, err_d_right_points, 'xr',label="Fehler von D1R f '(x)")
        plt.plot(x, err_dd_points, 'xb',label="Fehler von D2 f ''(x)")
        plt.plot(x, x, '-b', label="O(h)")
        plt.plot(x, np.array([h**2 for h in list(reversed(h_array))]), '-k',label="O(h^2)")
        plt.plot(x, np.array([h**3 for h in list(reversed(h_array))]), '-r',label="O(h^3)")
        plt.xlabel('h')
        plt.gca().invert_xaxis()
        plt.ylabel('Error')
        plt.title("Approximationsfehler")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_functions_list_1r(self, a, b, p, h_array):
        """Plots the right derivative approximations for multiple h values"""
        x_points = [a + i * (b - a) / p for i in range(p+1)]
        colors = ['r', 'g', 'c', 'm', 'y', 'k']

        # Plot first derivative approximations
        if self.d_f is not None:
            plt.plot(x_points, [self.d_f(x) for x in x_points], 'b--', label="f'(x)")
        for h, color in zip(h_array, colors):
            self.h = h
            d_f_approx = self.compute_dh_right_f()
            x_right = self.compute_r_interval(x_points, b)
            plt.plot(x_right, [d_f_approx(x) for x in x_right], f'{color}-', label=f"D1R,h={h:.1e}")
        plt.xlabel("x")
        plt.ylabel("f '(x)")
        plt.title("Erste rechtsseitige finite Differenz")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_functions_list_1l(self, a, b, p, h_array):
        """Plots the left derivative approximations for multiple h values"""
        x_points = [a + i * (b - a) / p for i in range(p+1)]
        colors = ['r', 'g', 'c', 'm', 'y', 'k']

        # Plot first derivative approximations
        if self.d_f is not None:
            plt.plot(x_points, [self.d_f(x) for x in x_points], 'b--', label="f'(x)")
        for h, color in zip(h_array, colors):
            self.h = h
            d_f_approx = self.compute_dh_left_f()
            x_left = self.compute_l_interval(x_points, a)
            plt.plot(x_left, [d_f_approx(x) for x in x_left], f'{color}-', label=f"D1L,h={h:.1e}")
        plt.xlabel("x")
        plt.ylabel("f '(x)")
        plt.title("Erste linksseitige finite Differenz")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_functions_list_1c(self, a, b, p, h_array):
        """Plots the central derivative approximations for multiple h values"""
        x_points = [a + i * (b - a) / p for i in range(p+1)]
        colors = ['r', 'g', 'c', 'm', 'y', 'k']

        # Plot first derivative approximations
        if self.d_f is not None:
            plt.plot(x_points, [self.d_f(x) for x in x_points], 'b--', label="f'(x)")
        for h, color in zip(h_array, colors):
            self.h = h
            d_f_approx = self.compute_dh_central_f()
            x_central = self.compute_c_interval(x_points, a,  b)
            text = f"D1Z,h={h:.1e}"
            plt.plot(x_central, [d_f_approx(x) for x in x_central], f'{color}-', label=text)
        plt.xlabel("x")
        plt.ylabel("f '(x)")
        plt.title("Erste zentrale finite Differenz")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_functions_list_2(self, a, b, p, h_array):
        """Plots the second derivative approximations for multiple h values"""
        x_points = [a + i * (b - a) / p for i in range(p+1)]
        colors = ['r', 'g', 'c', 'm', 'y', 'k']

        # Plot second derivative approximations
        if self.dd_f is not None:
            plt.plot(x_points, [self.dd_f(x) for x in x_points], 'b--', label="f''(x)")
        for h, color in zip(h_array, colors):
            self.h = h
            dd_f_approx = self.compute_ddh_f()
            x_central = self.compute_c_interval(x_points, a, b)
            text = f"D2,h={h:.1e}"
            plt.plot(x_central, [dd_f_approx(x) for x in x_central], f'{color}-', label=text)
        plt.xlabel("x")
        plt.ylabel("f ''(x)")
        plt.title("Zweite finite Differenz")
        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    """ Demonstration of the FiniteDifference class functionality. """

    def f(x):
        """ Example function: f(x) = sin(x) / x """

        return np.sin(x) / x

    def d_f(x):
        """ First derivative of f(x) """

        return (x * np.cos(x) - np.sin(x)) / (x ** 2)

    def dd_f(x):
        """ Second derivative of f(x) """

        return (2 * np.sin(x) - 2 * x * np.cos(x) - x**2 * np.sin(x)) / (x**3)

    #Parameters

    a, b = np.pi, 3 * np.pi
    p = 500
    h = np.pi / 5
    h_arr = [np.pi / 3, np.pi / 4, np.pi / 5, np.pi / 10]

    # Create FiniteDifference instance
    fd = FiniteDifference(h, f, d_f, dd_f)

    #User menu

    print("\nWählen Sie aus, welches Diagramm angezeigt werden soll:")
    print("1 — Funktion und ihre approximierten Ableitungen "
    "(plot_functions)")
    print("2 — Approximationsfehler für verschiedene Schrittweiten h "
    "(plot_errors)")
    print("3 — Rechtsseitige Ableitungsapproximationen für f' für mehrere h-Werte "
    "(plot_functions_list_1r)")
    print("4 — Linksseitige Ableitungsapproximationen für f' für mehrere h-Werte "
    "(plot_functions_list_1l)")
    print("5 — Zentrale Ableitungsapproximationen für f' für mehrere h-Werte "
    "(plot_functions_list_1c)")
    print("6 — Ableitungsapproximationen für f'' für mehrere h-Werte "
    "(plot_functions_list_2)")
    print("0 — Beenden")

    choice = input("Your choice: ").strip()

    if choice == "1":
        fd.plot_functions(a, b, p)
    elif choice == "2":
        h_arr = [10 ** (-i) for i in range(11)]
        fd.plot_errors(a, b, p, h_arr)
    elif choice == "3":
        fd.plot_functions_list_1r(a, b, p, h_arr)
    elif choice == "4":
        fd.plot_functions_list_1l(a, b, p, h_arr)
    elif choice == "5":
        fd.plot_functions_list_1c(a, b, p, h_arr)
    elif choice == "6":
        fd.plot_functions_list_2(a, b, p, h_arr)
    elif choice == "0":
        print("Exiting program.")
    else:
        print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()

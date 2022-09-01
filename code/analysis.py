import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm 
from matplotlib.colors import ListedColormap

from scipy.optimize import curve_fit

""" All the numbers are in MeVs """

# Constants (masses of proton, neutron, and atomic constant)
Mp = 938.2720882
Mn = 939.5654205
u = 931.4941024

def read_data():
    """ Reads binding energies and masses from the data file """
    nuclides = []

    with open(r"code/mass_1.mas20.txt") as file:
        # We skip the header
        for _ in range(37):
            file.readline()

        for line in file:
            nuclide = {}

            nuclide["N"] = int(line[4:9])
            nuclide["Z"] = int(line[10:14])
            nuclide["A"] = int(line[15:19])

            if nuclide["N"] + nuclide["Z"] != nuclide["A"]:
                print(f"Inconsistency at N={nuclide['N']}, Z={nuclide['Z']}")

            nuclide["symbol"] = line[20:22].strip()

            b = line[54:66]

            # We skip nonexperimental values marked by # sign
            if b.find('#') >= 0:
                continue

            # We convert keVs into MeVs
            nuclide["b"] = float(b) / 1000
            nuclide["error"] = float(line[68:77]) / 1000

            # We calculate the mass (can be also used the last column from the data file)
            nuclide["M"] = Mn * nuclide["N"] + Mp * nuclide["Z"] - nuclide["b"] * nuclide["A"]

            nuclides.append(nuclide)

    return nuclides


def max_nza(nuclides):
    """ Returns maximum values of the neutron number, proton number and mass number """
    max_n = 0
    max_z = 0
    max_a = 0

    for nuclide in nuclides:
        max_n = max(nuclide["N"], max_n)
        max_z = max(nuclide["Z"], max_z)
        max_a = max(nuclide["A"], max_a)

    return max_n, max_z, max_a


def colormap(elements=1024):
    """ An auxiliary function to change the colormap - zero values in white, large values in black """
    cmap = cm.get_cmap("jet", elements)
    cmap = cmap(np.linspace(0, 1, elements))
    cmap[:1, :] = [1, 1, 1, 1]        # Zero values (usually not measured values)
    cmap[-1:, :] = [0, 0, 0, 1]       # Large values
    cmap = ListedColormap(cmap)

    return cmap


def convert_2d(nuclides, name="b"):
    """ Returns the element with the given name as a 2D array """
    max_n, max_z, _ = max_nza(nuclides)

    result = np.empty((max_n + 1, max_z + 1), dtype=np.string_) if name == "symbol" else np.zeros((max_n + 1, max_z + 1))

    for nuclide in nuclides:
        result[nuclide["N"], nuclide["Z"]] = nuclide[name]

    return result


def plot_b(nuclides):
    """ Plots the binding energy """
    b = convert_2d(nuclides, "b")

    plt.imshow(b.transpose(), cmap=colormap(), origin="lower", interpolation="none")
    
    # Line N = Z
    plt.plot([0, min(b.shape)], [0, min(b.shape)], "black")

    plt.colorbar(label="MeV")                          
    plt.title("Binding energy per nucleon")
    plt.xlabel("N")
    plt.ylabel("Z")  

    plt.show()
    

def sort_b(nuclides):
    """ Sorts the nuclides array according to the binding energy per nucleon """
    bs = [nuclide["b"] for nuclide in nuclides]
    return [nuclide for _, nuclide in sorted(zip(bs, nuclides))]


def alpha(nuclides):
    """ Finds all possible nuclides that can decay via alpha process """

    masses = convert_2d(nuclides, "M")

    # Hellium - alpha particle
    M_alpha = masses[2, 2]

    alpha = np.zeros_like(masses)

    for nuclide in nuclides:
        n = nuclide["N"]
        z = nuclide["Z"]

        # We don't have data
        if masses[n, z] == 0:
            continue
        if n <= 2 or z <= 2 or masses[n - 2, z - 2] == 0:
            continue

        if masses[n, z] - M_alpha > masses[n - 2, z - 2]:
            alpha[n, z] = 1
        else:
            alpha[n, z] = 2

    plt.imshow(alpha.transpose(), cmap=colormap(), origin="lower", interpolation="none")
    plt.title("Possible alpha decay (in green)")
    plt.xlabel("N")
    plt.ylabel("Z")  

    plt.show()


def plot_magic(max_n, max_z):
    """ Plots the magic numbers """
    magic = [2, 8, 20, 28, 40, 50, 82, 126]

    for m in magic:
        if m < max_n:
            plt.plot([m, m], [0, max_z], "gray", lw=1)
        if m < max_z:
            plt.plot([0, max_n], [m, m], "gray", lw=1)            


def beta(nuclides):
    """ Finds all possible nuclides that can decay via beta or beta+ process """
    max_n, max_z, _ = max_nza(nuclides)
    masses = convert_2d(nuclides, "M")
    beta = np.zeros_like(masses)

    for nuclide in nuclides:
        n = nuclide["N"]
        z = nuclide["Z"]

        # We don't have data
        if masses[n, z] == 0:
            continue
        if z >= max_z or masses[n - 1, z + 1] == 0:
            continue
        if n >= max_n or masses[n + 1, z - 1] == 0:
            continue

        if n > 1 and masses[n, z] - Mn + Mp > masses[n - 1, z + 1]:
            beta[n, z] = 1
        elif z > 1 and masses[n, z] - Mp + Mn > masses[n + 1, z - 1]:
            beta[n, z] = 2
        else:
            beta[n, z] = 3

    plt.imshow(beta.transpose(), cmap=colormap(), origin="lower", interpolation="none")
    plt.title("Possible beta (in blue) or beta+ (in yellow) decay.")
    plt.xlabel("N")
    plt.ylabel("Z")  

    plt.show()


def s2n(nuclides):
    """ Two-neutron separation energy """
    masses = convert_2d(nuclides, "M")
    s = np.zeros_like(masses)

    for nuclide in nuclides:
        n = nuclide["N"]
        z = nuclide["Z"]

        # We don't have data
        if masses[n, z] == 0:
            continue
        if n < 2 or masses[n - 2, z] == 0:
            continue

        s[n, z] = masses[n - 2, z] + 2 * Mn - masses[n, z]

    plt.imshow(s.transpose(), cmap=colormap(), origin="lower", interpolation="none", vmin=0, vmax=20)
    plt.colorbar(label="keV")                          
    plt.title("Two-neutron separation energy")
    plt.xlabel("N")
    plt.ylabel("Z")  

    plt.show()


def s2p(nuclides):
    """ Two-proton separation energy """
    masses = convert_2d(nuclides, "M")
    s = np.zeros_like(masses)

    for nuclide in nuclides:
        n = nuclide["N"]
        z = nuclide["Z"]

        # We don't have data
        if masses[n, z] == 0:
            continue
        if z < 2 or masses[n, z - 2] == 0:
            continue

        s[n, z] = masses[n, z - 2] + 2 * Mp - masses[n, z]

    plt.imshow(s.transpose(), cmap=colormap(), origin="lower", interpolation="none", vmin=0, vmax=20)
    plt.colorbar(label="keV")                          
    plt.title("Two-proton separation energy")
    plt.xlabel("N")
    plt.ylabel("Z")  

    plt.show()


def b_theory(x, a_v, a_s, a_c, a_a, a_p):
    """ Bethe-Weizsäcker formula for the binding energy per nucleon; x = (n, z) """
    z, n = x
    a = n + z

    b = a_v - a_s * a**(-1/3) - a_c * z * z * a**(-4/3) - a_a * ((n - z) / a)**2
    delta = a_p * a**(-3/2)

    for i in range(len(z)):
        if z[i] % 2 == 1 and n[i] % 2 == 1:
            delta[i] = -delta[i]
        if a[i] % 2 == 1:
            delta[i] = 0

    b += delta

    return b


def bw_fit(nuclides):
    """ Fitting the Bethe-Weizsäcker formula """
    z = []
    n = []
    b = []

    for nuclide in nuclides:
        # We skip proton, neutron and deuteron
        if nuclide["A"] < 3:
            continue

        z.append(nuclide["Z"])
        n.append(nuclide["N"])
        b.append(nuclide["b"])

    z = np.array(z)
    n = np.array(n)
    b = np.array(b)

    cf = curve_fit(b_theory, (z,n), b)
    
    # Found parameters
    p = cf[0]
    print(f"aV = {p[0]}, aS = {p[1]}, aC = {p[2]}, aA = {p[3]}, aP = {p[4]}")

    d = b - b_theory((z, n), *p)
    print("chi^2 =", sum(d * d))

    max_n, max_z, _ = max_nza(nuclides)
    difference = np.zeros((max_n + 1, max_z + 1))
    for i in range(len(z)):
        difference[n[i], z[i]] = d[i]

    plt.imshow(difference.transpose(), cmap=cm.seismic, origin="lower", interpolation="none", vmin=-0.2, vmax=0.2)
    plt.colorbar(label="MeV")                          
    plt.title("Difference b_exp - b_theory")    
    plt.xlabel("N")
    plt.ylabel("Z")  

    plot_magic(max_n, max_z)

    plt.show()

if __name__ == "__main__":
    nuclides = read_data()
    plot_b(nuclides)

    sorted_nuclides = sort_b(nuclides)

    # Five elements with the highest binding energy
    for i in range(5):
        print(sorted_nuclides[-i - 1])

    alpha(nuclides)
    beta(nuclides)
    s2n(nuclides)
    s2p(nuclides)

    bw_fit(nuclides)
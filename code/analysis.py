from asyncore import read
import csv
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm 
from matplotlib.colors import ListedColormap

Mp = 938.2720882
Mn = 939.5654205
u = 931.4941024

def read_data():
    nuclides = []

    with open(r"mass_1.mas20.txt") as file:
        # Hlavička nás nezajímá
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

            # Hledáme jen experimentální hodnoty
            if b.find('#') >= 0:
                continue

            nuclide["b"] = float(b)
            nuclide["error"] = float(line[68:77])

            # Hmotnost dopočteme
            nuclide["M"] = Mn * nuclide["N"] + Mp * nuclide["Z"] - nuclide["b"] * nuclide["A"] / 1000

            nuclides.append(nuclide)

    return nuclides

def max_nza(nuclides):
    max_n = 0
    max_z = 0
    max_a = 0

    for nuclide in nuclides:
        max_n = max(nuclide["N"], max_n)
        max_z = max(nuclide["Z"], max_z)
        max_a = max(nuclide["A"], max_a)

    return max_n, max_z, max_a

def colormap():
    cmap = cm.get_cmap("jet", 1024)
    cmap = cmap(np.linspace(0, 1, 1024))
    cmap[:1, :] = [1, 1, 1, 1]        # Neměřené hodnoty
    cmap[-1:, :] = [0, 0, 0, 1]       # Velké hodnoty
    cmap = ListedColormap(cmap)

    return cmap

def b_array(nuclides):
    max_n, max_z, _ = max_nza(nuclides)

    b = np.zeros((max_n + 1, max_z + 1))
    for nuclide in nuclides:
        b[nuclide["N"], nuclide["Z"]] = nuclide["b"]

    return b

def plot_b(nuclides):
    b = b_array(nuclides)
    plt.imshow(b.transpose(), cmap=colormap(), origin="lower", interpolation="none")
    plt.plot([0, min(b.shape)], [0, min(b.shape)], "black")

    plt.colorbar(label="keV")                          
    plt.title("Binding energy per nucleon")
    plt.xlabel("N")
    plt.ylabel("Z")  

    plt.show()
    
def sort_b(nuclides):
    bs = [nuclide["b"] for nuclide in nuclides]
    return [nuclide for _, nuclide in sorted(zip(bs, nuclides), reverse=True)]

def alpha(nuclides):
    b = b_array(nuclides)

    max_n, max_z, _ = max_nza(nuclides)
    masses = np.zeros((max_n + 1, max_z + 1))
    alpha = np.zeros((max_n + 1, max_z + 1))

    for nuclide in nuclides:
        masses[nuclide["N"], nuclide["Z"]] = nuclide["M"]

    Ma = masses[2, 2]

    for nuclide in nuclides:
        n = nuclide["N"]
        z = nuclide["Z"]

        if masses[n, z] == 0:
            continue

        if n > 2 and z > 2 and masses[n - 2, z - 2] > 0 and masses[n, z] - Ma > masses[n - 2, z - 2]:
            alpha[n, z] = 1
        else:
            alpha[n, z] = 2

    plt.imshow(alpha.transpose(), cmap=colormap(), origin="lower", interpolation="none")
    plt.title("Alpha decay")
    plt.xlabel("N")
    plt.ylabel("Z")  

    plt.show()

def beta(nuclides):
    b = b_array(nuclides)

    max_n, max_z, _ = max_nza(nuclides)
    masses = np.zeros((max_n + 1, max_z + 1))
    beta = np.zeros((max_n + 1, max_z + 1))

    for nuclide in nuclides:
        masses[nuclide["N"], nuclide["Z"]] = nuclide["M"]

    Ma = masses[2, 2]

    for nuclide in nuclides:
        n = nuclide["N"]
        z = nuclide["Z"]

        # Nemáme data
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
    plt.title("Beta decay")
    plt.xlabel("N")
    plt.ylabel("Z")  

    plt.show()

nuclides = read_data()
plot_b(nuclides)

sorted_nuclides = sort_b(nuclides)

# Tři prvky s nejvyšší vazebnou energií
for i in range(5):
    print(sorted_nuclides[i])

alpha(nuclides)
beta(nuclides)

b = b_array(nuclides)

plt.plot(b[:, 82])
plt.ylim(7700, 7900)
plt.show()
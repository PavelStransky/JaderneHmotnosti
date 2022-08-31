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
            nuclide["M"] = Mn * nuclide["N"] + Mp * nuclide["Z"] - nuclide["b"] * nuclide["A"]

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

def plot_b(nuclides):
    max_n, max_z, _ = max_nza(nuclides)

    b = np.zeros((max_n + 1, max_z + 1))
    for nuclide in nuclides:
        b[nuclide["N"], nuclide["Z"]] = nuclide["b"]

    cmap = colormap()

    plt.imshow(b.transpose(), cmap=cmap, origin="lower", interpolation="none")
    plt.plot([0, max_z], [0, max_z], "black")

    plt.colorbar(label="keV")                          
    plt.title("Binding energy per nucleon")
    plt.xlabel("N")
    plt.ylabel("Z")  

    plt.show()
    
plot_b(read_data())


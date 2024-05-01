##### ASSIGNMENT 5

#### Student 23R51556 - Jorge Luri Vañó

### Imports and global definitions

## Hola Laura !!!!


import scipy.constants as sp
import numpy as np
from scipy.integrate import quad

T = 6504

mat_rgb_conv = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])

### Question 1

# Planck's law to acquire the spectral power distribution of the D65 illuminant

def S(lamb):

    lamb *= 10**(-9)
    first = (2*sp.pi*(sp.c**2)*sp.h) / (lamb**5)
    second = 1 / (np.exp((sp.h*sp.c)/(lamb*sp.k*T)) - 1)

    return first*second

# Let's get all the data from the files

plot1 = None
plot6 = None
plot15 = None
plot33 = None
plot41 = None
plot46 = None
plot51 = None
plot58 = None
plot64 = None
plot72 = None
plot74 = None
plot84 = None
plot92 = None
rgb = None

## Esto lo añado


with open("6plot", "r", encoding="utf-8") as f:
    f.readline()
    f.readline()
    plot6 = f.readlines()
    plot6 = [i.strip().split() for i in plot6]
    for i in range(len(plot6)):
        plot6[i] = [float(j) for j in plot6[i]]

with open("15plot", "r", encoding="utf-8") as f:
    f.readline()
    f.readline()
    plot15 = f.readlines()
    plot15 = [i.strip().split() for i in plot15]
    for i in range(len(plot15)):
        plot15[i] = [float(j) for j in plot15[i]]

with open("33plot", "r", encoding="utf-8") as f:
    f.readline()
    f.readline()
    plot33 = f.readlines()
    plot33 = [i.strip().split() for i in plot33]
    for i in range(len(plot33)):
        plot33[i] = [float(j) for j in plot33[i]]


with open("41plot", "r", encoding="utf-8") as f:
    f.readline()
    f.readline()
    plot41 = f.readlines()
    plot41 = [i.strip().split() for i in plot41]
    for i in range(len(plot41)):
        plot41[i] = [float(j) for j in plot41[i]]

with open("46plot", "r", encoding="utf-8") as f:
    f.readline()
    f.readline()
    plot46 = f.readlines()
    plot46 = [i.strip().split() for i in plot46]
    for i in range(len(plot46)):
        plot46[i] = [float(j) for j in plot46[i]]

with open("51plot", "r", encoding="utf-8") as f:
    f.readline()
    f.readline()
    plot51 = f.readlines()
    plot51 = [i.strip().split() for i in plot51]
    for i in range(len(plot51)):
        plot51[i] = [float(j) for j in plot51[i]]

with open("58plot", "r", encoding="utf-8") as f:
    f.readline()
    f.readline()
    plot58 = f.readlines()
    plot58 = [i.strip().split() for i in plot58]
    for i in range(len(plot58)):
        plot58[i] = [float(j) for j in plot58[i]]

with open("64plot", "r", encoding="utf-8") as f:
    f.readline()
    f.readline()
    plot64 = f.readlines()
    plot64 = [i.strip().split() for i in plot64]
    for i in range(len(plot64)):
        plot64[i] = [float(j) for j in plot64[i]]

with open("72plot", "r", encoding="utf-8") as f:
    f.readline()
    f.readline()
    plot72 = f.readlines()
    plot72 = [i.strip().split() for i in plot72]
    for i in range(len(plot72)):
        plot72[i] = [float(j) for j in plot72[i]]

with open("74plot", "r", encoding="utf-8") as f:
    f.readline()
    f.readline()
    plot74 = f.readlines()
    plot74 = [i.strip().split() for i in plot74]
    for i in range(len(plot74)):
        plot74[i] = [float(j) for j in plot74[i]]

with open("84plot", "r", encoding="utf-8") as f:
    f.readline()
    f.readline()
    plot84 = f.readlines()
    plot84 = [i.strip().split() for i in plot84]
    for i in range(len(plot84)):
        plot84[i] = [float(j) for j in plot84[i]]

with open("92plot", "r", encoding="utf-8") as f:
    f.readline()
    f.readline()
    plot92 = f.readlines()
    plot92 = [i.strip().split() for i in plot92]
    for i in range(len(plot92)):
        plot92[i] = [float(j) for j in plot92[i]]

with open("1plot", "r", encoding="utf-8") as f:
    f.readline()
    f.readline()
    plot1 = f.readlines()
    plot1 = [i.strip().split() for i in plot1]
    for i in range(len(plot1)):
        plot1[i] = [float(j) for j in plot1[i]]

with open("rgb", "r", encoding="utf-8") as f:
    rgb = f.readlines()
    rgb = [i.strip().split() for i in rgb]
    for i in range(len(rgb)):
        rgb[i] = [float(j) for j in rgb[i]]

# Application of reflectance to D65
        
def reflectance(plot):
    
    reflectance = []

    for i in range(len(plot)):
        reflectance.append([plot[i][0], plot[i][1] * S(plot[i][0])])

    return reflectance

plot1 = reflectance(plot1)
plot6 = reflectance(plot6)
plot15 = reflectance(plot15)
plot33 = reflectance(plot33)
plot41 = reflectance(plot41)
plot46 = reflectance(plot46)
plot51 = reflectance(plot51)
plot58 = reflectance(plot58)
plot64 = reflectance(plot64)
plot72 = reflectance(plot72)
plot74 = reflectance(plot74)
plot84 = reflectance(plot84)
plot92 = reflectance(plot92)

# Application of color matching function value

def find_values(lamb):
    
    for i in range(len(rgb)):
        if rgb[i][0] == lamb:
            return rgb[i][1], rgb[i][2], rgb[i][3]

def color_matching(plot):

    xyz = np.array([0.0, 0.0, 0.0])
    xyz_ref = np.array([0.0, 0.0, 0.0])

    for i in range(len(plot)):

        x, y, z = find_values(np.round(plot[i][0]))

        xyz += np.array([x, y, z]) * plot[i][1]
        xyz_ref += np.array([x, y, z]) * S(plot[i][0])

    xyz /= xyz_ref # Normalization

    return xyz[0], xyz[1], xyz[2]


x1, y1, z1 = color_matching(plot1)
x6, y6, z6 = color_matching(plot6)
x15, y15, z15 = color_matching(plot15)
x33, y33, z33 = color_matching(plot33)
x41, y41, z41 = color_matching(plot41)
x46, y46, z46 = color_matching(plot46)
x51, y51, z51 = color_matching(plot51)
x58, y58, z58 = color_matching(plot58)
x64, y64, z64 = color_matching(plot64)
x72, y72, z72 = color_matching(plot72)
x74, y74, z74 = color_matching(plot74)
x84, y84, z84 = color_matching(plot84)
x92, y92, z92 = color_matching(plot92)

print("For the pigment 1, our coordinates are :")
print(x1, y1, z1)
print()

print("For the pigment 6, our coordinates are :")
print(x6, y6, z6)
print()

print("For the pigment 15, our coordinates are :")
print(x15, y15, z15)
print()


print("For the pigment 33, our coordinates are :")
print(x33, y33, z33)
print()

print("For the pigment 41, our coordinates are :")
print(x41, y41, z41)
print()

print("For the pigment 46, our coordinates are :")
print(x46, y46, z46)
print()

print("For the pigment 51, our coordinates are :")
print(x51, y51, z51)
print()

print("For the pigment 58, our coordinates are :")
print(x58, y58, z58)
print()

print("For the pigment 64, our coordinates are :")
print(x64, y64, z64)
print()

print("For the pigment 72, our coordinates are :")
print(x72, y72, z72)
print()

print("For the pigment 74, our coordinates are :")
print(x74, y74, z74)
print()

print("For the pigment 84, our coordinates are :")
print(x84, y84, z84)
print()

print("For the pigment 92, our coordinates are :")
print(x92, y92, z92)
print()

print("\n--- --- ---\n")

#########

### Question 2


def xyz_to_srgb(xyz):

    mat_rgb_conv_inv = np.linalg.inv(mat_rgb_conv)

    rgb_out = np.dot(mat_rgb_conv_inv, xyz)
    rgb_out *= 255
    return np.clip(np.round(rgb_out), 0, 255)

sumsOfSrgb = {}

def question2(xyz, number):
    srgb_values = xyz_to_srgb(xyz)

    sumsOfSrgb[number] = srgb_values[0] + srgb_values[1] + srgb_values[2]

    print("sRGB Color Coordinates of " + str(number) + "plot:")
    print("Red =", int(srgb_values[0]))
    print("Green =", int(srgb_values[1]))
    print("Blue =", int(srgb_values[2]))
    print()

question2(np.array([x1, y1, z1]), 1)
question2(np.array([x6, y6, z6]), 6)
question2(np.array([x15, y15, z15]), 15)
question2(np.array([x33, y33, z33]), 33)
question2(np.array([x41, y41, z41]), 41)
question2(np.array([x46, y46, z46]), 46)
question2(np.array([x51, y51, z51]), 51)
question2(np.array([x58, y58, z58]), 58)
question2(np.array([x64, y64, z64]), 64)
question2(np.array([x72, y72, z72]), 72)
question2(np.array([x74, y74, z74]), 74)
question2(np.array([x84, y84, z84]), 84)
question2(np.array([x92, y92, z92]), 92)

print("\n--- --- ---\n")

#########

### Question 3

max_value = max(sumsOfSrgb.values())
min_value = min(sumsOfSrgb.values())

for i in sumsOfSrgb.keys():
    if sumsOfSrgb[i] == max_value:
        print("The pigment with the highest luminance is " + str(i) + "plot")
        print()
        break


for i in sumsOfSrgb.keys():
    if sumsOfSrgb[i] == min_value:
        print("The pigment with the lower luminance is " + str(i) + "plot")
        print()
        break


# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 17:20:05 2022
@author: Jonas Peter
"""
import sys
import os
import time
#insert path of parent folder to import helpers
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import torch
import torch.nn as nn
from torch.autograd import Variable
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
from helpers import *
import warnings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train = True
class Net(nn.Module):
    def __init__(self, num_layers, layers_size):
        super(Net, self).__init__()
        assert num_layers == len(layers_size)
        self.linears = nn.ModuleList([nn.Linear(1, layers_size[0])])
        self.linears.extend([nn.Linear(layers_size[i-1], layers_size[i])
                            for i in range(1, num_layers)])
        self.linears.append(nn.Linear(layers_size[-1], 1))

    def forward(self, x):  # ,p,px):
        # torch.cat([x,p,px],axis=1) # combined two arrays of 1 columns each to one array of 2 columns
        #x = torch.unsqueeze(x, 1)
        for i in range(len(self.linears)-1):
            x = torch.tanh(self.linears[i](x))
        output = self.linears[-1](x)
        return output

##
# Hyperparameter
learning_rate = 0.01
mse_cost_function = torch.nn.MSELoss()  # Mean squared error


# Definition der Parameter des statischen Ersatzsystems
Lb = float(input('Länge des Kragarms [m]: '))
EI = 21
K = 5/6
G = 80
A = 100

#Normierungsfaktor (siehe Kapitel 10.3)
normfactor = 10/((11*Lb**5)/(120*EI))

# ODE als Loss-Funktion, Streckenlast
Ln = 0 #float(input('Länge Einspannung bis Anfang der ' + str(i + 1) + '. Streckenlast [m]: '))
Lq = Lb # float(input('Länge der ' + str(i + 1) + '. Streckenlast [m]: '))
s = str(normfactor)+"*x"#input(str(i + 1) + '. Streckenlast eingeben: ')

def h(x):
    return eval(s)

#Netzwerk System 1
def f(x, net_phi):
    u = net_phi(x)
    _,_,u_xxx = deriv(u,x,3)
    ode = u_xxx + (h(x - Ln))/EI
    return ode

#Netzwerk für System 2
def g(x, net_v, net_phi):
    u = net_v(x)
    z = net_phi(x)
    u_x = deriv(u,x,1)[0]
    _,z_xx = deriv(z,x,2)
    ode = u_x - z/EI + z_xx/(K*A*G)
    return ode

x = np.linspace(0, Lb, 1000)
qx = h(x)* (x <= (Ln + Lq)) * (x >= Ln)

Q0 = integrate.cumtrapz(qx, x, initial=0)

qxx = (qx) * x

M0 = integrate.cumtrapz(qxx, x, initial=0)
def gridSearch(num_layers, layers_size):
    start = time.time()
    net_phi = Net(num_layers, layers_size)
    net_v = Net(num_layers, layers_size)
    net_phi = net_phi.to(device)
    net_v = net_v.to(device)
    mse_cost_function = torch.nn.MSELoss()  # Mean squared error
    optimizer = torch.optim.Adam([{'params': net_phi.parameters()}, {'params': net_v.parameters()}], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, verbose=True, factor=0.75)
    if train:
        mse_cost_function = torch.nn.MSELoss()  # Mean squared error
        y1 = net_v(torch.unsqueeze(myconverter(x, False), 1))
        fig = plt.figure()
        plt.grid()
        ax1 = fig.add_subplot()
        ax1.set_xlim([0, Lb])
        ax1.set_ylim([-10, 0])
        net_out_plot = myconverter(y1)
        line1, = ax1.plot(x, net_out_plot)
        plt.title(f'{num_layers =}, {layers_size =}')
        plt.show(block=False)
        pt_x = torch.unsqueeze(myconverter(x), 1)
        f_anal = (-1 / 120 * normfactor * x ** 5 + 1 / 6 * Q0[-1] * x ** 3 - M0[-1] / 2 * x ** 2) / EI + (
                1 / 6 * normfactor * x ** 3 - Q0[-1] * x) / (K * A * G)

    iterations = 1000000
    for epoch in range(iterations):
        optimizer.zero_grad()  # to make the gradients zero
        x_bc = np.linspace(0, Lb, 500)
        pt_x_bc = torch.unsqueeze(myconverter(x_bc), 1)
        x_collocation = np.random.uniform(low=0.0, high=Lb, size=(250 * int(Lb), 1))

        pt_x_collocation = torch.unsqueeze(myconverter(x_collocation), 1)
        f_out_phi = f(pt_x_collocation, net_phi)
        f_out_v = g(pt_x_collocation, net_v, net_phi)

        # Randbedingungen (siehe Kapitel 10.5)
        net_bc_out_phi = net_phi(pt_x_bc)
        net_bc_out_v = net_v(pt_x_bc)
        phi_x, phi_xx = deriv(net_bc_out_phi, pt_x_bc, 2)

        # RB für Netzwerk 1
        BC6 = phi_xx[0] - Q0[-1]
        BC7 = phi_xx[-1]
        BC8 = phi_x[0] + M0[-1]
        BC9 = phi_x[-1]
        BC10 = net_bc_out_phi[0]

        # RB für Netzwerk 2
        BC1 = net_bc_out_v[0]

        mse_Gamma_phi = errsum(mse_cost_function, 1 / normfactor * BC6, BC7, 1 / normfactor * BC8, BC9, BC10)
        mse_Gamma_v = errsum(mse_cost_function, BC1)
        mse_Omega_phi = errsum(mse_cost_function, f_out_phi)
        mse_Omega_v = errsum(mse_cost_function, f_out_v)

        loss = mse_Gamma_phi + mse_Gamma_v + mse_Omega_phi + mse_Omega_v

        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        with torch.autograd.no_grad():
            if epoch % 10 == 9:
                print(epoch, "Traning Loss:", loss.data)
                plt.grid()
                net_out_v = myconverter(net_v(pt_x))
                err = np.linalg.norm(net_out_v - f_anal, 2)
                print(f'Error = {err}')
                if err < 0.1 * Lb:
                    print(f"Die L^2 Norm des Fehlers ist {err}.\nStoppe Lernprozess")
                    break
                line1.set_ydata(net_out_v)
                fig.canvas.draw()
                fig.canvas.flush_events()


# GridSearch
time_elapsed = []
for num_layers in range(1, 4):
    for _ in range(10):  # Wieviele zufällige layers_size pro num_layers
        layers_size = [np.random.randint(10, 250) for _ in range(num_layers)]
        time_elapsed.append(
            (num_layers, layers_size, gridSearch(num_layers, layers_size)))
        plt.close()

with open(r'random8m.txt', 'w') as fp:
    for item in time_elapsed:
        # write each item on a new line
        fp.write(f'{item} \n')

##

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Total population, N.
N = 1000
# Initial number of infected, vaccinated and recovered individuals, I0, V0 and R0.
I0, R0, V0 = 1, 0, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0 - V0
# Contact rate, beta, vaccination rate, sigma, the universal birth/mortality rate, mu, and mean recovery rate, gamma, (in 1/days).
beta, gamma, sigma, mu = 0.5, 0.1/10, 0.4, 0.2
# A grid of time points (in days)
t = np.linspace(0, 160, 160)

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma, sigma, mu):
    S, I, R, V = y
    dSdt = -mu*S + mu*N - sigma*S - beta*S*I/N
    dIdt = -mu*I + beta*S*I/N - gamma*I
    dRdt = -mu*R + gamma*I + sigma*S
    dVdt = -mu*V + sigma*S/N
    #dCdt = ((beta*S*I)/N)-(S*sigma)-(gamma*I)
    return dSdt, dIdt, dRdt, dVdt

# Initial conditions vector
y0 = S0, I0, R0, V0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma, sigma, mu))
S, I, R, V = ret.T

# Plot the data on three separate curves for S(t), I(t) , V(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, axisbelow=True)
ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R/1000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.plot(t, V/1000, 'c', alpha=0.5, lw=2, label='Vaccinated')
#ax.plot(t, C/1000, 'm', alpha=0.5, lw=2, label='Precancerous')
ax.set_xlabel('Time /days')
ax.set_ylabel('Number (1000s)')
#ax.set_ylim(0,2.0)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()
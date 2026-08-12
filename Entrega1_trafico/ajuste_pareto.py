import numpy as np
import matplotlib.pyplot as plt

datos = np.loadtxt("arrivals.txt")

tiempos = datos[:, 0]
interarribos = np.diff(tiempos)
interarribos = interarribos[interarribos > 0]

# Se ajusta la cola superior de los datos
xmin = np.percentile(interarribos, 90)
cola = interarribos[interarribos >= xmin]

n = len(cola)

alpha = n / np.sum(np.log(cola / xmin))

print("Resultados del ajuste Pareto")
print("----------------------------")
print(f"Total de interarribos: {len(interarribos)}")
print(f"Umbral xmin: {xmin:.8f} s")
print(f"Datos usados en la cola: {n}")
print(f"Parametro alpha: {alpha:.4f}")

if alpha <= 1:
    print("La media teorica de Pareto no es finita.")
elif alpha <= 2:
    print("La media teorica es finita, pero la varianza teorica es infinita.")
else:
    print("La media y la varianza teoricas son finitas.")

x_emp = np.sort(cola)
ccdf_emp = 1 - np.arange(1, n + 1) / n

x_pareto = np.logspace(
    np.log10(xmin),
    np.log10(np.max(cola)),
    300
)

ccdf_pareto = (xmin / x_pareto) ** alpha

plt.figure(figsize=(8, 5))

plt.loglog(
    x_emp,
    ccdf_emp,
    label="CCDF empírica",
    linewidth=1.5
)

plt.loglog(
    x_pareto,
    ccdf_pareto,
    label=f"Pareto ajustada, alpha={alpha:.3f}",
    linewidth=2
)

plt.xlabel("Tiempo entre llegadas (s)")
plt.ylabel("1 - F(x)")
plt.title("Ajuste Pareto de la cola de los tiempos entre llegadas")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("ajuste_pareto.png", dpi=300)
plt.show()




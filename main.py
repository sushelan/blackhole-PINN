from syntheticgen import generate_synthetic_orbit


data = generate_synthetic_orbit()

print('done')
import matplotlib.pyplot as plt

plt.figure()
plt.plot(data.x_true, data.y_true, label="true orbit")
plt.scatter(data.x_obs, data.y_obs, s=15, label="observations")
plt.axis("equal")
plt.legend()
plt.show()

print("True M, a:", data.M_true, data.a_true)
print("r min / max:", data.r_true.min(), data.r_true.max())


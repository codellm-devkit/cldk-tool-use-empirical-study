import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42  # Embed fonts as TrueType
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Times New Roman'

plt.plot([1, 2, 3], [4, 5, 6])
plt.title("Sample Plot")
plt.savefig("figure.pdf")

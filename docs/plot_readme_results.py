"""Replot published, rounded notebook metrics; no inference or training."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / 'docs' / 'images'
IMAGES.mkdir(parents=True, exist_ok=True)
labels = [r'$\eta$', r'$E_{\mathrm{bd}}^{(0)}$', r'$\ell_c$', r'$k$']
targets = ['eta', 'mean_breakdown_field', 'correlation_length', 'weibull_shape']
models = {
    'Random Forest': ([.2276, .1374, 3.5317, 2.5486], [.815, .053, .084, .312]),
    'Residual CNN': ([.2369, .1326, 3.1719, 2.2629], [.8100, .0741, .1912, .4362]),
    'Deep hybrid CNN': ([.2271, .1344, 3.2786, 2.2677], [.8215, .0638, .1707, .4398]),
    'Gaussian mean': ([.2160, .1335, 3.1952, 2.2581], [.8337, .0687, .1540, .4241]),
}
uncertainty = {
    'Individual': (4.2875, [.9426, .9369, .8857, .8976], [1.0503, .5677, 12.5460, 8.8308]),
    'Individual calibrated': (4.1336, [.9519, .9648, .9250, .9488], [1.0880, .6311, 14.5789, 11.0233]),
    'Constant': (4.5411, [.9477, .9410, .8464, .8655], [1.2064, .5801, 12.0439, 8.6730]),
    'Constant calibrated': (4.3970, [.9633, .9669, .9053, .9343], [1.3710, .6353, 14.1758, 10.6676]),
}
colors = ['#8996a5', '#3b82b8', '#dc9343', '#229979']

plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
    'axes.spines.right': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})

fig, ax = plt.subplots(figsize=(10.8, 4.8))
x = np.arange(4)
width = .18
for i, (name, (_, r2)) in enumerate(models.items()):
    ax.bar(x + (i - 1.5) * width, r2, width, label=name, color=colors[i])
ax.set_xticks(x, labels, fontsize=14)
ax.set_ylim(0, .95)
ax.set_ylabel(r'Test $R^2$ (higher is better)')
ax.set_title('Inverse parameter recovery', fontsize=16, pad=38)
ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=4, frameon=False, fontsize=10)
ax.grid(axis='y', alpha=.18)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(IMAGES / 'inverse_test_r2.png', dpi=200, bbox_inches='tight')
plt.close(fig)

with (ROOT/'docs'/'inverse_test_summary.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['model','target','mae','r2'])
    for name,(mae,r2) in models.items():
        for target,m,r in zip(targets,mae,r2): writer.writerow([name,target,m,r])
with (ROOT/'docs'/'gaussian_test_summary.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['model','target','joint_test_nll','coverage95','mean_width95'])
    for name,(nll,coverage,widths) in uncertainty.items():
        for target,c,w in zip(targets,coverage,widths): writer.writerow([name,target,nll,c,w])
print('Created 1 figure and 2 metric tables from rounded notebook outputs.')

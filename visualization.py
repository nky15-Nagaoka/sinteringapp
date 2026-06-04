import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from matplotlib.patches import Circle, Ellipse, Rectangle, FancyArrowPatch

# Japanese font support for Matplotlib.
# If japanize_matplotlib is installed, Japanese labels render correctly.
# If it is not available, labels fall back to English to avoid garbled/tofu text.
try:
    import japanize_matplotlib  # noqa: F401
    JP_FONT_OK = True
except Exception:
    JP_FONT_OK = False


def _label(jp: str, en: str) -> str:
    return jp if JP_FONT_OK else en


def _draw_initial_particles(ax, rng, n=46, title=None):
    ax.set_aspect('equal'); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    ax.add_patch(Rectangle((0,0),1,1,facecolor='white',edgecolor='black',lw=1.2))
    for _ in range(n):
        x,y = rng.random(2)
        r = rng.uniform(0.035,0.085)
        ax.add_patch(Ellipse((x,y), 2*r*rng.uniform(0.8,1.45), 2*r*rng.uniform(0.8,1.35),
                             angle=rng.uniform(0,180), facecolor='0.55', edgecolor='black', lw=0.8))
    ax.text(0.02,0.90, title or _label('(a) 粒子充填体', '(a) Powder compact'), fontsize=13, weight='bold', bbox=dict(facecolor='white', edgecolor='black'))


def _draw_dense_polycrystal(ax, rng, rho=0.96, porosity=0.04, title=None):
    ax.set_aspect('equal'); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    ax.add_patch(Rectangle((0,0),1,1,facecolor='0.55',edgecolor='black',lw=1.2))
    pts = rng.random((26,2))
    bounds = np.array([[-1,-1],[-1,0.5],[-1,2],[0.5,-1],[0.5,2],[2,-1],[2,0.5],[2,2]])
    vor = Voronoi(np.vstack([pts,bounds]))
    for ridge in vor.ridge_vertices:
        if -1 in ridge:
            continue
        v = vor.vertices[ridge]
        # clip visually by only drawing segments intersecting the unit square neighborhood
        if np.all((v >= -0.04) & (v <= 1.04)):
            ax.plot(v[:,0], v[:,1], color='black', lw=0.8)
    # isolated closed pores only; no particles/circles as grains
    pore_count = int(np.clip(24*porosity/0.12, 0, 18))
    for _ in range(pore_count):
        x,y = rng.random(2)
        ax.add_patch(Circle((x,y), rng.uniform(0.006,0.018), facecolor='white', edgecolor='black', lw=0.6))
    ax.text(0.02,0.90, title or _label('(b) 緻密化後', '(b) Dense polycrystal'), fontsize=13, weight='bold', bbox=dict(facecolor='white', edgecolor='black'))
    ax.annotate(_label('粒界', 'grain boundary'), xy=(0.63,0.55), xytext=(0.78,0.67), color='white',
                arrowprops=dict(arrowstyle='->', color='white', lw=1.2), fontsize=11)
    if rho >= 0.97:
        ax.text(0.50,0.04, _label('ほぼ緻密化完了', 'nearly fully dense'), ha='center', fontsize=10, bbox=dict(facecolor='white', edgecolor='black'))
    else:
        ax.text(0.50,0.04, _label('閉気孔が残存', 'closed pores remain'), ha='center', fontsize=10, bbox=dict(facecolor='white', edgecolor='black'))


def draw_transition_schematic(rho, porosity, seed=1):
    """Attached-style before/after schematic: particles -> dense polycrystal."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2), gridspec_kw={'width_ratios':[1,0.35,1]})
    _draw_initial_particles(axes[0], rng, n=48, title=_label('(a) 焼結前', '(a) Before sintering'))
    axes[1].axis('off'); axes[1].set_xlim(0,1); axes[1].set_ylim(0,1)
    axes[1].add_patch(FancyArrowPatch((0.05,0.5),(0.95,0.5),arrowstyle='->',mutation_scale=18,lw=1.8))
    axes[1].text(0.50,0.60, _label('焼結', 'Sintering'), ha='center', va='bottom', fontsize=12)
    _draw_dense_polycrystal(axes[2], rng, rho=rho, porosity=porosity, title=_label('(b) 焼結後', '(b) After sintering'))
    fig.tight_layout()
    return fig


def draw_microstructure(rho, G_um, porosity, liquid_flag=0, seed=1):
    """Density-aware schematic. Late stage is drawn as polygonal grains, not circles."""
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(_label(f'現在の微細構造模式図  ρ={rho:.3f}, 気孔率={porosity:.3f}, G≈{G_um:.2f} µm', f'Current microstructure  rho={rho:.3f}, porosity={porosity:.3f}, G~{G_um:.2f} um'))

    if rho < 0.70:
        _draw_initial_particles(ax, rng, n=44, title=_label('初期: 粒子充填', 'Initial: particle packing'))

    elif rho < 0.84:
        ax.add_patch(Rectangle((0,0),1,1,facecolor='white',edgecolor='black',lw=1.2))
        n = 30
        pts = rng.random((n,2))
        # draw broad necks first so the figure looks connected
        for i in range(n):
            d = np.sum((pts-pts[i])**2, axis=1)
            for j in np.argsort(d)[1:3]:
                ax.plot([pts[i,0], pts[j,0]], [pts[i,1], pts[j,1]], color='0.55', lw=12, alpha=0.55, solid_capstyle='round')
        for i in range(n):
            x,y = pts[i]
            r = rng.uniform(0.050,0.095)
            ax.add_patch(Circle((x,y), r, facecolor='0.55', edgecolor='black', lw=0.7))
        pore_count = max(3, int(20*(1-rho)/0.35))
        for _ in range(pore_count):
            x,y = rng.random(2)
            ax.add_patch(Circle((x,y), rng.uniform(0.012,0.032), facecolor='white', edgecolor='black', lw=0.6))
        ax.text(0.02,0.90, _label('中期: ネック成長・開気孔収縮', 'Intermediate: neck growth / open-pore shrinkage'), fontsize=11, weight='bold', bbox=dict(facecolor='white', edgecolor='black'))

    else:
        # From roughly closed-pore stage onward, draw as a dense polycrystal.
        _draw_dense_polycrystal(ax, rng, rho=rho, porosity=porosity, title=_label('後期: 緻密多結晶', 'Final: dense polycrystal'))

    if liquid_flag:
        ax.text(0.02,0.03, _label('液相/助剤効果 ON', 'Liquid/additive effect ON'), fontsize=10, bbox=dict(facecolor='white', edgecolor='black'))
    return fig

"""
Configure matplotlib for plot in ipython. 
IEEE figure width: 
    one column: 3.5 inches
    two column: 7.16 inches
"""
import matplotlib as mpl

LARGE_SIZE = 9
MEDIUM_SIZE = 7.5 
params ={\
    'backend': 'GTK3Agg',
    
    'font.family': 'sans-serif',
    'font.serif': ['Times', 'Palatino', 'New Century Schoolbook', 'Bookman', 'Computer Modern Roman'],
    'font.sans-serif' : ['Helvetica', 'Avant Garde', 'Computer Modern Sans serif'],
#font.cursive       : Zapf Chancery
#font.monospace     : Courier, Computer Modern Typewriter
    'text.usetex': True,
    'axes.labelsize': LARGE_SIZE,
    'axes.linewidth': .75,
    'figure.subplot.left' : 0.175,
    'figure.subplot.right': 0.95,
    'figure.subplot.bottom': 0.15,
    'figure.subplot.top': .95,
    
    'figure.dpi':150,
    
    'font.size': MEDIUM_SIZE,
    'legend.fontsize': MEDIUM_SIZE,
    'xtick.labelsize': MEDIUM_SIZE,
    'ytick.labelsize': MEDIUM_SIZE,
    'lines.markersize': 2,
    'lines.linewidth':.75,
    'savefig.dpi':600,
    }
GOLDEN_RATION =  1.618

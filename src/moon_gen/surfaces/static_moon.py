import numpy as np

from moon_gen.lib.utils import SurfaceType


def surface(n=200, seed=42) -> SurfaceType:
    '''
    A static moon surface with fixed features.
    Uses a fixed seed for reproducible results.
    '''
    np.random.seed(seed)
    
    # Create coordinate grid
    ax = ay = 50
    x = np.linspace(-ax/2, ax/2, n)
    y = np.linspace(-ay/2, ay/2, n)
    X, Y = np.meshgrid(x, y)
    
    # Create base elevation with subtle rolling hills
    z = 0.5 * np.sin(X / 10) * np.cos(Y / 10)
    
    # Add some fixed craters at specific locations
    crater_positions = [
        (-20, -15, 2.0, 5),   # (x, y, depth, radius)
        (15, 20, 1.5, 4),
        (0, 0, 1.2, 3),
        (-10, 10, 0.8, 2.5),
        (20, -10, 1.0, 3.5),
    ]
    
    for cx, cy, depth, radius in crater_positions:
        distance = np.sqrt((X - cx)**2 + (Y - cy)**2)
        crater = -depth * np.exp(-(distance**2) / (2 * radius**2))
        z += crater
    
    # Add some fine detail with fixed noise
    detail = 0.1 * np.sin(X * 0.5) * np.cos(Y * 0.5)
    z += detail
    
    return x, y, z

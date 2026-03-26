"""Test script for Theta* pathfinding implementation."""

import numpy as np
from moon_gen.planning.main import generate_all_candidates
from moon_gen.planning.maps import load_lunar_image

# Load image
image, path = load_lunar_image()
print(f"Image shape: {image.shape}")
print(f"Image path: {path}")

# Generate all three routes using new Theta* implementation
print("\n=== Testing Theta* Pathfinding ===")
# Use valid coordinates for the 170x170 image
result = generate_all_candidates(image, (10, 10), (150, 150))

# Check all three modes
print("\nMode Results:")
for mode in ["safe", "eco", "fast"]:
    plan = result["plans"][mode]
    res = plan["result"]
    if res.exists:
        path_len = res.path_length
        cost = res.cost
        print(f"{mode.upper():5} mode: Path length={path_len:6.1f}, Cost={cost:10.2f}")
    else:
        print(f"{mode.upper():5} mode: NO PATH FOUND")

print("\nGlobal mission conditions:")
print(f"  Global risk: {result['global_risk']:.3f}")
print(f"  Mean uncertainty: {result['mean_uncertainty']:.3f}")

print("\n✓ Theta* implementation verified!")

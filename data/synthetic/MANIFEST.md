# Synthetic tier generation manifest
- seed: 20260728
- compositions: 10000 ({'nist_perturbed': np.int64(4500), 'lean_sweet': np.int64(1325), 'sour': np.int64(1325), 'rich': np.int64(1325), 'n2_rich': np.int64(1325), 'nist_real': np.int64(200)}); split: {'train': np.int64(9026), 'val': np.int64(974)}
- T range: [250.0, 450.0] K with T >= 1.05 * Tpc_kay
- P range: log-uniform [0.05, 140.0] MPa, capped at Ppr_kay <= 30.0
- z bounds: (0.15, 3.0); stability filter: dP/drho > 0
- points kept: 9,999,878; rejected: {'unstable': 0, 'z_bounds': 122, 'error': 0}
- z stats: {'count': 9999878.0, 'mean': 1.0485, 'std': 0.2774, 'min': 0.2581, '25%': 0.9597, '50%': 0.9934, '75%': 0.9992, 'max': 3.0}
- DETAIL cross-check on 20000 sampled points: median |dz|/z = 0.0148%, p95 = 0.4190%, max = 33.330%
- EOS: GERG-2008 via pyaga8 (validated vs NIST reference to machine precision)

import iceutils as ice
import numpy as np
import os

import sys
sys.path.append('..')
from constants import E_XX_STACKFILE, E_XY_STACKFILE, E_YY_STACKFILE, VX_STACKFILE, VY_STACKFILE


def get_strain_stress():
    vx_stack = ice.Stack('..' + VX_STACKFILE)
    vy_stack = ice.Stack('..' + VY_STACKFILE)

    dx = vx_stack.hdr.dx
    dy = vy_stack.hdr.dy

    e_xx, e_yy, e_xy = [], [], []
    
    for vx, vy in zip(vx_stack._datasets['data'], vy_stack._datasets['data']):
        robust_opts = {'window_size': 250, 'order': 2}

        # Compute stress strain
        strain_dict, stress_dict = ice.compute_stress_strain(vx, vy, dx=dx, dy=dy, grad_method='robust_l2', inpaint=False, **robust_opts)
        e_xx.append(strain_dict['e_xx'])
        e_yy.append(strain_dict['e_yy'])
        e_xy.append(strain_dict['e_xy'])

    if not os.path.exists('..' + E_XX_STACKFILE):
        e_xx_stack = ice.Stack('..' + E_XX_STACKFILE, mode='w', init_tdec=vx_stack.tdec, init_rasterinfo=vx_stack.hdr)
        e_xx_stack.fid.create_dataset('data', data=np.array(e_xx))
        e_xx_stack.fid.close()

    if not os.path.exists('..' + E_YY_STACKFILE):
        e_yy_stack = ice.Stack('..' + E_YY_STACKFILE, mode='w', init_tdec=vx_stack.tdec, init_rasterinfo=vx_stack.hdr)
        e_yy_stack.fid.create_dataset('data', data=np.array(e_yy))
        e_yy_stack.fid.close()

    if not os.path.exists('..' + E_XY_STACKFILE):
        e_xy_stack = ice.Stack('..' + E_XY_STACKFILE, mode='w', init_tdec=vx_stack.tdec, init_rasterinfo=vx_stack.hdr)
        e_xy_stack.fid.create_dataset('data', data=np.array(e_xy))
        e_xy_stack.fid.close()


def analyze_strain_stress():
    strain_dict, stress_dict = get_strain_stress()
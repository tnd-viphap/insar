from dev.stamps.flow import StaMPSStep
from modules.tomo.ps_parms import Parms
import numpy as np


if __name__ == "__main__":
    
    # from modules.snap2stamps.bin._9_0_stamps_prep import StaMPSPrep
    # StaMPSPrep('TOMO', 0.4).process()
    
    # from modules.tomo.input_parm import Input
    
    # slcstack, interfstack = Input(False).run()
    
    # nlines, nwidths, n_interf = interfstack["datastack"].shape
    # n_slc = n_interf + 1

    # reference_ind = list(set(slcstack["filename"]) - set(interfstack["filename"]))[0]
    # reference_idx = slcstack["filename"].index(reference_ind)
    
    # print(reference_idx)
    
    
    # miniStackSize = 5
    # mini_ind = list(range(0, n_slc, miniStackSize))
    
    # print(miniStackSize)
    
    # # Check if reference index is in mini_ind
    # reference_ComSAR_ind = mini_ind.index(reference_idx) if reference_idx in mini_ind else 0
    # if reference_ComSAR_ind == 0:
    #     # Add reference index and sort
    #     mini_ind.append(reference_idx)
    #     mini_ind = sorted(mini_ind)

    #     temp_diff = [miniStackSize] + [mini_ind[i] - mini_ind[i-1] for i in range(1, len(mini_ind))]

    #     one_image_ind = [i for i, diff in enumerate(temp_diff) if diff < 2]

    #     reference_ComSAR_ind = mini_ind.index(reference_idx)

    #     # Two images needed for interferometric phase
    #     if one_image_ind:
    #         for idx in one_image_ind:
    #             if reference_ComSAR_ind != idx:
    #                 mini_ind[idx] = mini_ind[idx] + 1
    #             elif idx > 1:
    #                 mini_ind[idx - 1] = mini_ind[idx - 1] - 1
    #             else:
    #                 mini_ind[idx + 1] = mini_ind[idx + 1] + 1

    # # Ensure last index is not equal to n_slc (need 2 images per mini-stack)
    # if mini_ind[-1] == n_slc - 1:
    #     mini_ind[-1] = mini_ind[-1] - 1

    # # Number of mini stacks
    # numMiniStacks = len(mini_ind)

    # Unified_flag = True
    # if Unified_flag:
    #     Unified_ind = np.arange(mini_ind[0], n_slc)
    #     try:
    #         reference_UnifiedSAR_ind = list(Unified_ind).index(reference_idx)
    #     except ValueError:
    #         reference_UnifiedSAR_ind = -1  # not found

    #     N_unified_SAR = len(Unified_ind)

    # if Unified_flag:
    #     slcstack_ComSAR_filename = [slcstack["filename"][i] for i in Unified_ind]
    #     mask = np.isin(interfstack["filename"], slcstack_ComSAR_filename)
    #     interfstack_ComSAR_filename = [interfstack["filename"][i] for i in np.where(mask)[0]]
    #     print("reference_UnifiedSAR_ind: ", reference_UnifiedSAR_ind)
    #     print("master_index: ", reference_UnifiedSAR_ind)
    #     print(slcstack_ComSAR_filename[reference_UnifiedSAR_ind])
    # else:
    #     slcstack_ComSAR_filename = [slcstack["filename"][i] for i in mini_ind]
    #     mask = np.isin(interfstack["filename"], slcstack_ComSAR_filename)
    #     interfstack_ComSAR_filename = [interfstack["filename"][i] for i in np.where(mask)[0]]
    #     print("reference_ComSAR_ind: ", reference_ComSAR_ind)
    #     print("master_index: ", reference_ComSAR_ind)
    #     print(slcstack_ComSAR_filename[reference_ComSAR_ind])
    # print(slcstack_ComSAR_filename)
    # print(interfstack_ComSAR_filename)

    project_conf = "modules/snap2stamps/bin/project.conf"
    parms = Parms(project_conf)
    parms.load()
    steps = StaMPSStep(parms)

    steps.run(1, 1)
    steps.run(2, 2)

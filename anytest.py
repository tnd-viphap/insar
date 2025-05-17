if __name__ == "__main__":
    
    from modules.snap2stamps.bin._9_0_stamps_prep import StaMPSPrep
    StaMPSPrep('TOMO', 0.4).process()
    
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
    # if reference_idx in mini_ind:
    #     reference_ComSAR_ind = mini_ind.index(reference_idx)
    #     print(reference_ComSAR_ind)
    # else:
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
        
    # print("-> Performing coherence estimation and phase linking...")

    # # Number of mini stacks
    # numMiniStacks = len(mini_ind)
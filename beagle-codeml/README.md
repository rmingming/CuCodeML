## 1. Notice:
   This sub-directory is used to generate BeagleCodeML only for runtime comparision purposes.
   WARNNING: DO NOT USE IT FOR REAL ANALYSIS!

## 2. Generating beagle-codeml:
   + Install the beagle library.(https://github.com/beagle-dev/beagle-lib)
   + Edit *Makefile* in this sub-directory. 
      The following are some relevant configurations you may need to edit:
```
      * CUDA_INSTALL_PATH: your cuda installation path
      * BEAGLE_INSTALL_PATH: your beagle installation path
      * BEAGLE_GPU_ID: the GPU card you want to use (from 1 to #GPUs)
```

      Notice: You also need to modity the beagle libary path in two src files: beagle-codeml.cu and treesub_beagle.c.
      WARNING: DO NOT EDIT ANY OTHER PART OF THE MAKEFILE
   + Execute the command: make

## 3. Running beagle-codeml:
   The following command is used for running beagle-codeml:
```
    your-BeagleCodeML-path/BeagleCodeML
```

   (just like running codeml)  

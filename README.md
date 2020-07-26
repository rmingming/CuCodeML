# CuCodeML Manual

1. Introduction
2. Getting Started
3. License and Warranty

-------------------------------------------------------------------------

## 1. Introduction

CuCodeML is a GPU version of CodeML for the branch-site model.

## 2. Getting Started

The following instructions take you through a sequence of steps to get the
default configuration of CuCodeML up and running.

(a) Prerequisites
- Hardware Requirements
    * GPU: NVIDIA graphics card(s) with CUDA compute capability 2.0 and above
- Software Requirements
    * OS: we have tested on Centos6.4
    * C compiler: gcc is sufficient
    * CUDA suites: CUDA Driver, CUDA Toolkit 

(b) Configuration
- Edit *Makefile* with your favorite editor and edit the following options in
configuration part to fit your need:
    * CUDA_INSTALL_PATH: your cuda installation path
    * MULTICARD_ONLY_GPU: yes if you want to use multiple GPU cards and not cooperate with CPU
    * HYBRID: yes if you want to use multiply GPU cards or cooperate with CPU(single GPU card or multiple cards) (only available if MULTICARD_ONLY_GPU is *no*)
    * SINGLE_GPU_ID: the single GPU card id which you want to use (from 0 to #GPUs-1)(only available if MULTICARD_ONLY_GPU and HYBRID are *no*)
    * DIVIDE_EQUALLY: yes if you want to divide the sites equally among all the GPU cards (yes is suggested if your GPU cards are of the same type) (only avaliable if MULTICARD_ONLY_GPU is *yes*)
    * SSE: yes if your CPU support SSE
    * MAXGPU: how many GPU cards to use
    * MAXCPU: how many CPU cores to use (zero if you want to use *GPU only*)
    
    WARNING: DO NOT EDIT ANY OTHER PART OF THE MAKEFILE    

    Summary:
    If you want a quick configuration, just modifying the CUDA_INSTALL_PATH and SINGLE_GPU_ID configuration is usually enough.
 
    If you just want to use one GPU card and not cooperate with CPU, edit the configuration just like this:
  ```
       MULTICARD_ONLY_GPU = no
       HYBRID = no
       SINGLE_GPU_ID = #GPUID you want to use
       MAXGPU = 1
  ```

    If you want to use one GPU card and cooperate with CPU, edit like this:
  ```
       MULTICARD_ONLY_GPU = no
       HYBRID = yes
       MAXGPU = 1
  ```

    If you want to use multiple GPU cards and not cooperate with CPU, edit like this:
  ```
       MULTICARD_ONLY_GPU = yes
       HYBRID = no
       MAXGPU = #GPUs you want to use ( bigger than 1) 
  ```

    If you want to use multiple GPU cards and cooperate with CPU, edit like this:
  ```
       MULTICARD_ONLY_GPU = no
       HYBRID = yes
       MAXGPU = #GPUs you want to use( bigger than 1)
  ```

(c) Build
-  For all shells:
   `make `

(d) Installation
 -   CuCodeML doesn't need to be installed.

(e) Execution
 -   Run the program with the following command:
     `your-CuCodeML-path/CuCodeML`
 -   The arguments it takes is exactly the same as that codeml takes.

## 3. License and Warranty
CuCodeML is a free, open-source software under GNU GPL v3 license.

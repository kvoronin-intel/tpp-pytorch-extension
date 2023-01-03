#!/bin/bash

# bf16 tests
python -u test_optim.py --test-optim SGD_fb_enhanced          --use-bf16-opt
python -u test_optim.py --test-optim SGD_bf16_enhanced        --use-bf16-opt
python -u test_optim.py --test-optim SGD_bf16fb_enhanced      --use-bf16-opt --without-checkpointing
python -u test_optim.py --test-optim SplitSGD_bf16fb_enhanced --use-bf16-opt

# fp32 tests
python -u test_optim.py --test-optim SGD_fb_enhanced          
python -u test_optim.py --test-optim SGD_bf16_enhanced        
python -u test_optim.py --test-optim SGD_bf16fb_enhanced                     --without-checkpointing
python -u test_optim.py --test-optim SplitSGD_bf16fb_enhanced               

# Each test produces either "Validation failed" or "Validation succeeded" as a part of its output

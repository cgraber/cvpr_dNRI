# Code accompanying "Dynamic Neural Relational Inference"

This codebase accompanies the paper ["Dynamic Neural Relational Inference"](http://openaccess.thecvf.com/content_CVPR_2020/html/Graber_Dynamic_Neural_Relational_Inference_CVPR_2020_paper.html) from CVPR 2020.

This code was written using the following packages:
- PyTorch 1.2.0
- numpy 1.16.4
- transforms3d 0.3.1 (For Motion Capture data processing)

To run this code, you should pip install it in editable mode. This can be done using the following command:

`pip install -e ./`

Scripts train models can be found in the `run_scripts` directory.

Datasets:
- Motion Capture: the datasets can be downloaded from http://mocap.cs.cmu.edu/search.php?subjectnumber=118 
  and http://mocap.cs.cmu.edu/search.php?subjectnumber=35. For subject 35, you need trials 1-16 and 28-34.
  For subject 118, you need trials 1-30.
- Basketball: The data can be accessed here: https://github.com/ezhan94/multiagent-programmatic-supervision

Attribution:
Some portions of this code are based on the code for the paper "Neural Relational Inference for Interacting
Systems." This code can be found at https://github.com/ethanfetaya/NRI

If you use this code or this model in your work, please cite us:

    @inproceedings{dNRI,
      title={Dynamic Neural Relational Inference},
      author={Graber, Colin and Schwing, Alexander},
      booktitle={The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2020},
    }
  
  

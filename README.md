### CycleGAN implementation for the project of GIF-7005

This repository is the implementation of the CycleGAN algorithm, based on:

 Zhu, J.-Y., Park, T., Isola, P., & Efros, A. A. (2020). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (No. arXiv:1703.10593). arXiv. https://doi.org/10.48550/arXiv.1703.10593

### To run the style transfer and evaluation code

1) Start by installing the requirements in `requirements.txt`
2) If you want to use the pretrained weights, which are default weights, simply run `run_generators_on_datasets.py`. It will create an output folder of the style transfer for the input images and generators specified in the code (go change the paths of dataset and weights if necessary). If you simply want to visualize the style transfer, use the script `gui_test.py`. Start by selecting a folder containing test images using the tkinter interface and input an index to test on the image of corresponding index. You can modify the generators by modifying the paths in the script.
3) Once you have the input dataset and the output dataset, you can measure LPIPS using `eval_lpips.py`, which will print the LPIPS score for each generator evaluated (go change the paths of datasets if necessary)
4) To measure the FID, make sure you have installed `pytorch_fid` and run the following command in terminal:
```
python -m pytorch_fid path/to/inputs path/to/outputs
```
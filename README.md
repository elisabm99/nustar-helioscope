# nustar-helioscope
Code to obtain the bound of arXiv:2407.03828

To run the code
1) Download coronal magnetic field model from https://www.predsci.com/corona/jul2019eclipse/data.php
2) Convert magnetic field from HDF format to Numpy’s compressed array format with convert_hdf.py and save the npz files in a folder named npzs/
3) Run conversion_prob.py to get the conversion probability
4) Run bounds.py to make the bound

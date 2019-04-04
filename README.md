# QRdar

This is a repository for a method of using fiducial markers (akin to QR codes) to identify trees in laser scans.

<p align="center"><img src=http://www2.geog.ucl.ac.uk/~ucfaptv/IMG_20180704_095428.jpg width=400></p>

The method uses <a href=https://www.uco.es/investiga/grupos/ava/node/26>AruCo</a> markers attachted to trees, 
these are then scanned as part of normal plot scanning and extracted in post-processing. 

Although the method has currently only be implemented using a RIEGL VZ-400, with some simple modifications, it could be
applicable to any intensity recording instrument e.g. not the ZEB-REVO as this only records position.

### Preparing targets
We printed the targets on waterproof paper (365 gsm weight) using a standard office laser jet printer. These were then attached
to the trees either with nails or using string. Existing tree tags and the tree codes were recorded using an ODK form on a
smartphone. It is a good idea to locate markers where they are clearly visible and facing towrads the centre of the plot.
A guide to deploying targets can be found <a href=https://docs.google.com/document/d/1WuwAQ8iDk_QOwp7p3tvTk5mp5DD1B9hG0njXC-MNlBo/edit?usp=sharing>
here</a>.

### Scanning targets
There are no specific requirements for scanning codes as they should be adequately captured with normal scanning. They have
been tested using angular resolutions of <= 0.04.

### Post processing
Post processing is desscribed in the Jupyter notebook.

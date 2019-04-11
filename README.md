# qrDAR

This is a method for using fiducial markers (akin to QR codes) to identify the location of tagged features in laser scans.

<p align="center"><img src=http://www2.geog.ucl.ac.uk/~ucfaptv/IMG_20180704_095428.jpg width=400></p>

The examples here use <a href=https://www.uco.es/investiga/grupos/ava/node/26>AruCo</a> markers attachted to trees, these are then scanned as part of normal scanning and their location and connected point cloud are extracted in post-processing. 

Although the method has currently only been tested using a RIEGL VZ-400, with some simple modifications, it could be applicable to any intensity recording laser scanner e.g. not the ZEB-REVO as this only records <i>xyz</i>.

### Preparing targets
Targets are printed on waterproof paper using a standard office laser jet printer. The paper has to be reasonably heavy weight (we used 365 gsm weight) so targets don't move in the wind or bend over - the flatter the target the better! Circular retro-reflective stickers are then attached to the markers to allow for automatic identification in post-processing. The paper and stickers are easily sourced online.

It is a good idea to locate markers where they are clearly visible (avoiding occlusion) and, if scanning a regular grid, facing towrads the centre of the plot.

A set of 250 markers can be found here <a href=https://github.com/philwilkes/qrdar/blob/master/markers/aruco_tags_16h3_000-249.pdf>here</a> and a notebook to make your own <a href=https://github.com/philwilkes/qrdar/blob/master/markers/create_markers.ipynb>here</a>. A guide to deploying targets can be found <a href="https://docs.google.com/document/d/1WuwAQ8iDk_QOwp7p3tvTk5mp5DD1B9hG0njXC-MNlBo/edit?usp=sharing">here</a>.

### Scanning targets
There are no specific requirements for scanning codes as they should be adequately captured with normal scanning. They have
been tested using angular resolutions of <= 0.04 where scans were done on a regular 10 m grid.

### Post processing
Post processing is described in the <a href=https://github.com/philwilkes/qrdar/blob/master/find_markers.ipynb>find_markers.ipynb</a>. This includes an option to extract the desired feature.

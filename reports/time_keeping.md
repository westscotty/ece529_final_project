# ECE 529 Project Time Log

## Proposal

- **2024-09-16**: Project brainstorming (1 hour)
- **2024-09-18**: Project brainstorming (1.5 hours)
- **2024-09-20**: Worked on initial proposal ideas (2 hours)
- **2024-09-22**: Narrowed proposal idea to KLT and worked on project plan (2 hours)
- **2024-09-23**: Researched articles and textbooks for image processing sources related to
                  KLT algorithm, and found example videos to use for testing (2.5 hours)
- **2024-09-24**: Finalized proposal with sources identification (3 hours)

## Development and Testing

- **2024-10-05**: Played with cv2 library in python importing videos and converting to grayscale (2 hours)
- **2024-10-06**: Played with cv2 library more (1 hours)
- **2024-10-07**: Created a sobel operator from scratch to use for calculating gradients of the images (2.5 hours)
- **2024-10-08**: Played with cv2 library using built in commands for corner detection on an image (4 hours)
- **2024-10-10**: Developed debugging portions, committed testing scripts to git (2 hours)
- **2024-10-21**: Developed convolution routine from scratch (3 hours)
- **2024-10-28**: Hooked up convolution routine with kernels and made it work with sobel operators and other filters (very generic) (4.5 hours)
- **2024-11-01**: Corner detection algorithm development (shi-tomasi corners) (12.5 hours)
  - Developed shi-tomasi code for both numpy and cv2 implementiations, including error checking code
  - Added plotting to show difference between home grown solution for corner detection vs openCV module
  - Added k-means clustering method for grouping detected points for the purpose of drawing bounding boxes
  - Added several basic image transformations including high/low pass filters, and histogram equalization
  - Added code for drawing uniform plots
  - Added debugging code for evaluating metrics between image operations during intermediate steps

- **2024-11-02**: Polished the shi-tomasi method a little more, and took a stab at implementing the optical flow portion of the lucas-kanade algorithm (4.5 hours)
- **2024-11-04**: Devloped further the lucas-kanade method (openCV portion) and generated a video utilities file to use for generic functin calls. Ready to convert to implementing the lucas-kanade alg from scratch now that the wrapper for the results is flushed out. (4.75 hours)
- **2024-11-05**: Developed algorithm for calcOpticalFlowPyrLK method, using same inputs as the cv2 method. Now I have a fully functional custom klt alg and a cv2 implementation. Next steps are making the custom method run solely on numpy operators, as well as quantifying performance between it, and adding comments throughout to discuss the math behind each operation. (7 hours)
- **2024-11-10**: Working on adding numpy versioning to the lucas-kanade algorithm (now fully numpy convolutions can be used) as well as added a frame skipping routine and a recursive pydown image operation to make the images smaller and therefore faster to calculate on (5.5 hours)
- **2024-11-15**: Revisited code for corner detections, made it work by testing both baseline and custom methods for corners (3 hours)
- **2024-11-18**: Worked on code clean up (2 hours)
- **2024-11-21**: Developed video utilites to better work with code base, more generic now (3 hours)
- **2024-11-22**: Wroked on plotting functions to come with methods to figure out how to best plot the corners are detected (nearest neighbor, k-means clustering, euclidean distance grouping) (4 hours)
- **2024-11-25**: Worked on testing metric ideas(2 hours)
- **2024-12-02**: Updated code for all image operations to be more generic, only working with a single image at a time (2.5 hours)
  - Started final report (1.5 hours)
  - Developed testing metrics for cv2 implementation versus numpy implementation
    - Split focus between corners, persisting corners, and corner location (number of corners over all)
  - Developed testing metric plots (2 hours)
  - Reworked convolution algorithm (1.5 hour)
- **2024-12-03**: Developed method for output images to be in quadrants (3 hours)
  - Developed MC analaysis script for evaluating performance overall(3 hours)
  - Developed psnr, mae, ssim, precison, and recall methods for images to be evaluated against each other (3.5 hours)
- **2024-12-04**: Connected mc script to lucas kanade script in order to generate data and plots that show the results from each seed compared to the others (3.5 hours)
  - Also updated time_keeping sheet

## TODO

- comment more code
- Work on report report detailing the algorithm design and code implementation comparing the code from scratch versus the code from openCV
- Need to write about the convolutio algorithm for the sobel and gaussian operators
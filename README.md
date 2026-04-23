# CloudChamber
Python code to take images of a cloud chamber and to analyse them: optical correction matrix, optical correction and fiducial volume, generation of the background image, raw clustering, merging of fragmented clusters. removing correlated cluster, final results. There are two applications : thoron lifetime and Radon in air measurement. 

## Experimental Setup and Preparation
Before using the code, the experimental setup must be carefully prepared:
- The cloud chamber must be cleaned thoroughly:
- - glass surface
- - internal volume (cloud)
- - electrical wires and components
    To reduce noise and unwanted traces
- The webcam must be correctly positioned:
- - the image is rectangular
- - the full field of view must include the calibration chessboard (damier)
- - check alignment before data taking
- The system must be protected from external light:
- - use a cover on the cloud chamber
- - in practice (Subatech), the cover is not sufficient
- Additional protection is required:
- - paper or cardboard around the setup, especially around the chessboard area
This is critical to reduce background noise ("bruit de fond")

## Configuration data
The code cloudChamberCommonCode.py defined all data that configures a given data taking period and analysis.
This file contains all the parameters used for: image processing, filtering, clustering, merging, and physics analysis.
Main parameters include: 
- Calibration: calibrationFactor = 0.44 (pixel → mm conversion)
- Filtering:
- - seuil = 70, binarization threshold (pixel > seuil → white and pixel < seuil → black)
- - seuilDiff = 20, background subtraction threshold
- - timePeriod = 60, timeStep = 6, background computation to detect only the moving particles
- Clustering
- - clusterSizeThreshold =45, the minimum cluster size to be analyzed
- Merging: 
- - maxLinePointDistance = 4.
- - maxRelativeAngle = 25.
- - maxRelativeDistance = 50.
    To keep clusters that resemble real particle tracks and reject noise
- Correlation removal
- - maxCorrelatedRelativeDistance = 75.
- - maxCorrelatedRelativeAngle = 25.
    To link clusters across images to track particles over time
- Selection:
- - goodCluster(): quality criteria
- - minLength = 15, maxLength = 80 for the track selection
- Fiducial volume
- - interestArea_x1 = 650, interestArea_x2 = 1260,
- - interestArea_y1 = 60, interestArea_y2 = 1030,
- - coronaSize = 25, border exclusion
This file must be adapted for each experimental setup and dataset.

## Image Data Taking
- In src/acq/ directory the python3 script webcam_dacq.py to take a series of images of the webcam (to be used at the end of the day with FULL HD WEBCAM using a linux computer). See specifications of the webcam FULL HD in the doc directory.
- The acquisition rate is controlled using a cycle parameter defined as cycle = FPS / 10. This means that only 1 frame out of every “cycle” frames is stored, resulting in approximately 10 images per second, independently of the camera frame rate.
This reduces data size and limits correlations between consecutive images while keeping a stable acquisition rate.
- Test acquisitions are stored in: src/acq-test-jj-mm-aaaa/
- Each test is saved with its date and used to validate setup, lighting, and camera position.
- Ensure stable lighting conditions (dark environment) to avoid vibrations and verify that the chamber is properly visible.

## Optical Aberration correction and Image Fiducial Area
- In src/acq/ directory the python3 script chessBoard_CameraCalibrationProcess.py performs the optical aberration correction (using the chess board image in data/ImageDamier_FullResolution/ ) and selects the fiducial area of the image in which the radioactive tracks will be reconstructed and analyzed.
- The fiducial area is the region where the tracks are reliable, distortions are minimized and edge effects are avoided.

## Filtering of corrected images
In src/rec/ directory the python3 script
filteringProcess.py  performs :
- the calculation of the background of the images is calculated for a time period of timePeriod images. In the calculation, to avoid correlations, considered image a distance by timeStep images and the pixel are kept if the light different is less and seuilDiff. The background image are named bck_*
- the background contribution is substract form each corrected image and are named diff_*
- The image is then binarized 0 or 255, depending on a seuil value. The binarized images are named bina_*
- The binarized image is filtered. Very small cluster, less than 5x5 are removed, setting pixel values to zero. After filtering the image a named filt_aber_*
- At the end of the filtering, the average occupancy over a certains interval of images (integrationTime expressed as a number of Image). Only images every deltaTimeStep in unit of images are considered to avoid correlations. The error of the averaged pixel occupancy is measured from the statistical fluctuation in the considered interval of averaging. These results is presented as a plot and the data occupancy point can be fitted to a constant or exponential plus constant function. 
- this script can be executed with a filtering option set to zero, on order to performed only the occupancy and fitting process. 
- For the first execution, the filtering option is set to 1 to perform the full image processing (background computation, subtraction, binarization, and filtering). This step is necessary to generate clean data and reduce noise. The option 0 is only used afterward for faster analysis on already processed images (occupancy and fitting only).

## Raw clustering of filtered images
- In src/rec/, the script rawClusteringProcess.py is used to detect and analyze particle tracks by grouping connected pixels into clusters. These clusters are then used to study key properties such as position, angle, length, shape, and cluster size.
- A clusterSizeThreshold is applied, which defines the minimum number of pixels required for a group to be accepted as a valid cluster. This helps remove noise and keep only meaningful tracks.
- To perform clustering, an empty image is prepared for visualization, and all pixels are scanned using a loop. Each pixel has a status: done = 1 means it has already been processed, and done = 0 means it is still unvisited. The algorithm selects the first unvisited pixel and starts growing a cluster from it.
- The cluster is expanded by checking neighboring pixels in all directions (up, down, left, and right). This process continues until no new connected neighbors are found, forming a complete cluster.
- Once a cluster is created, the number of pixels inside it is checked. The algorithm then computes the center position (x, y) of the track, as well as its orientation and shape.
- The shape of the cluster is described using a covariance matrix. This helps distinguish different track types: for example, an alpha track appears as a thin and long structure with a clear main direction.
- For each cluster, the image number, position, and angle (in x and y directions) are stored.
- For visualization, each track is displayed with a green line showing the main direction of the cluster, and a red ellipse representing the uncertainty in its shape.
- The results are saved in a data file called rawClusteringData.dat, and processed images showing the detected clusters are saved with the prefix clus_.

## Merging fragmented clusters
- In src/rec/, the script mergingFragmentedClusterProcess.py is used to merge clusters that belong to the same physical track but were initially detected as separate fragments in rawClusteringData.dat.
- The idea is that a single particle track can appear split into multiple pieces, so this step reconstructs the full track by merging consistent fragments. To decide whether two clusters should be merged, the algorithm checks both their spatial and directional consistency.
- A key step is computing the perpendicular distance from a point to a cluster direction line, given by:
$$
d = | -\sin(\theta)(x - x_0) + \cos(\theta)(y - y_0) |
$$
  A small value means the point is close to the track direction, while a large value means it is not related to that cluster.
- The algorithm loops over all images and builds a clusterMergedList, which contains the final merged clusters. A mergedClusterNumberList is also used to ensure that the same cluster is not merged multiple times.
- For each reference cluster, the algorithm compares it with other clusters. Only clusters that are not already merged and are physically valid are considered. Each cluster is described using its center position (x, y) and its angle, which defines the track direction.
- Two clusters are merged if they satisfy three main conditions:
- - the are close in space: 
$$
d = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}
$$
- - they have similar direction (small angle difference) : 
$$
\Delta \theta = |\theta_1 - \theta_2|
$$
- - and they lie along the same track line based on the perpendicular distance test.
- When these conditions are satisfied, the clusters are merged by combining their pixel lists into a single larger cluster. The merged cluster is then marked so it cannot be reused again.
- Finally, after merging, the properties of the new cluster are recomputed, including its main direction, size, and width, to better represent the full physical track.

## Removing correlated clusters
- In src/rec/, the script removingCorrelatedClusterProcess.py is used to eliminate duplicated detections of the same physical track that appear across consecutive images.
- The algorithm starts from the previously merged clusters and builds a new dictionary called clusterDictRemoved, which stores the final cleaned results. For each image, every cluster is compared not only with clusters in the same frame but also with clusters in the next 7 images. This is done because a physical track can persist across several frames depending on the camera frame rate and the particle lifetime, which helps avoid missing delayed duplicates.
- The process is similar to the merging step: for each reference cluster in a given image, only physically meaningful clusters are considered, and clusters already marked for removal are skipped. Then each reference cluster is compared with candidate clusters in later frames.
- For each comparison, the algorithm recomputes the spatial distance and the angle difference between clusters. If two clusters have nearly the same position and direction, they are considered to represent the same physical track. In this case, the second cluster is marked as a duplicate and stored in RemoveClusterListDict, so it can be removed later.
- A small additional condition is used: if j == -1, the algorithm checks the quality of the clusters, and if the second cluster has better shape information, it is preferred instead of the first one.
- Finally, once duplicates are removed, the physical quantities of the tracks are extracted. The track length, position, and size are computed using a calibration factor. The factor is multiplied by 2 because the track width is interpreted as a Gaussian distribution, where the measured value corresponds to σ (standard deviation), and the full track size is taken as 2σ.

## Final Distribution Analysis
- In src/rec/, the script distributionProcess.py performs the final selection and physical analysis of reconstructed clusters.
- At this stage, only clusters with good quality are kept. Cuts are applied on cluster properties such as minimum valid length and shape consistency. In addition, clusters located near the image borders are removed because edge regions are more noisy and lead to unreliable detections.
- Total number of detected particles: the total number of reconstructed clusters is counted over all processed images:
$$
N_{\text{total}}
$$
- Merged Ratio: the merging efficiency is evaluated using the fraction of clusters that were merged during reconstruction:
  $$
R_{\text{merged}} = \frac{N_{\text{total}} - N_{\text{final}}}{N_{\text{total}}}
  $$
This ratio indicates how fragmented the initial detection was a high value means many fragmented tracks were merged and a low value indicates clean and well-separated detections.
- Cluster rate (particle rate): the particle detection rate is computed as the number of clusters per second:
$$
R_{\text{cluster}} = \frac{N_{\text{total}}}{N_{\text{images}}} \times f_{\text{fps}}
$$
where:Nimages is the number of processed images, ffps is the camera frame rate (images per second)
This gives the physical particle rate in particles per second.
- This final step transforms raw reconstructed tracks into physically meaningful observables. It ensures that only reliable clusters are used for analysis and provides global quantities such as particle rate, merging efficiency, and spatial distributions, which are essential for the final physical interpretation of the experiment.

# depth_from_disparity
Disparity computation for stereo vision

1. The code disparity_map.cpp takes stereo images, rectifies them and then by extracts ORB features and computes the matches using brute-force matcher and then calculates disparity. The left image has been taken as the reference.
2. It then converts the 2d points to 3d points using reprojectTo3d function of opencv whose output is saved in stereo_rect.yml.
3. The depth of the feature points have been computed using the equation 
                                  Depth = (focal length)*(baseline)/disparity
The depth values are stored in the file depth_map.yml

Commands
g++ -std=c++11  disparity_map.cpp -o ./disparity_map `pkg-config --cflags --libs opencv`
./disparity_map

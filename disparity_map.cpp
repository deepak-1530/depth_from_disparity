//disparity map computation from features correspondences.
//only the horizonal distances are being computed as the vertical height will be same.
//SIFT features have been used as they are rotation invariant.
//firstly stereo rectification is performed and then the disparity is computed.

#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/features2d.hpp>
//#include<opencv2/xfeatures2d.hpp>
#include<opencv2/calib3d.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc.hpp>
#include<stdio.h>

using namespace cv;
using namespace std;
//using namespace cv::xfeatures2d;

int main()
{
    Mat img_L,img_R;
   
    //take the images.
    img_L = imread("left2.png",0);
    img_R = imread("right2.png",0);
     
    //stereo rectification using the extrinsic and intrinsic parameters. 
    //this is done to make the epipolar lines parallel and hence correspondences can be computed easily.
    //load the intrinsic and extrinsic features.

    char* rectification_cam_file = "3d_coordinates.yml";
    char* depth_of_features = "depth_features.yml";
      
    cv:: FileStorage fsrc(rectification_cam_file,FileStorage::WRITE);
    cv:: FileStorage fd(depth_of_features, FileStorage::WRITE);
  
    Mat K1,D1,K2,D2,R,T;  //matrices for intrinsic parame

K1 = (Mat_<double>(3,3) << 1300.916421, 0.000000, 693.632446, //left camera matrix
                          0.000000, 1236.693231, 263.159396, 
                          0.000000, 0.000000, 1.000000);
K2 = (Mat_<double>(3,3) << 1300.916421, 0.000000, 693.632446, 0.000000, 1236.693231, 263.159396, 0.000000, 0.000000, 1.000000);
D1 = (Mat_<double>(5,1) <<-0.333135, 0.094357, 0.003903, -0.010715, 0.000000);
D2 = (Mat_<double>(5,1) <<-0.357812, 0.177484, 0.001862, 0.001925, 0.000000);
R = (Mat_<double>(3,3)<<1, 0 ,0 ,0 ,1 ,0 ,0 ,0, 1);
T = (Mat_<double>(3,1)<<3.4332044927271215e+02, -1.0313693129244882e+01, 1.0344859084778136e-13);

  cv::Mat R1, R2, P1, P2, Q; //stereo rectification parameters.
  stereoRectify(K1, D1, K2, D2, img_L.size(), R, T, R1, R2, P1, P2, Q); //function for stereo rectification

  printf("Rectification Done\n");
 
  Mat map1x, map1y, map2x, map2y;
  Mat imgU_L, imgU_R;

  initUndistortRectifyMap(K1, D1, R1, P1, img_L.size(), CV_32FC1, map1x, map1y);
  initUndistortRectifyMap(K2, D2, R2, P2, img_L.size(), CV_32FC1, map2x, map2y);

    remap(img_L, imgU_L, map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
    remap(img_R, imgU_R, map2x, map2y, INTER_LINEAR, BORDER_CONSTANT, Scalar());  
cout<<"mapping done"<<endl;
imshow("rectified Left",imgU_L);
imshow("rectified Right",imgU_R);
waitKey(0);
   //now disparity computation is performed 
  // we first compute the features and then match them using flann based matcher.    
    vector<KeyPoint> kpl,kpr; //feature vectors
    Mat img_L_desc, img_R_desc;

   Ptr<FeatureDetector> detector  = ORB::create();
   detector->detectAndCompute(imgU_L , Mat() ,kpl,img_L_desc);
   detector->detectAndCompute(imgU_R , Mat() ,kpr,img_R_desc);
   
   //Ptr<DescriptorExtractor> descriptor =ORB::create(); //descriptors for feature vectors
   //descriptor->compute(imgU_L,kpl, img_L_desc);
   //descriptor->compute(imgU_R,kpr, img_R_desc);
   if ( img_L_desc.empty() )
   cvError(0,"MatchFinder","1st descriptor empty",__FILE__,__LINE__);    
if ( img_R_desc.empty() )
   cvError(0,"MatchFinder","2nd descriptor empty",__FILE__,__LINE__);
    
   //now matching is done using FLANN based matcher
   BFMatcher matcher(NORM_L2);
   std::vector<DMatch> matches;

   matcher.match(img_L_desc, img_R_desc, matches);
   double max_distance = 100, min_distance=0;

   vector<DMatch> good_matches;
    for(int i=0; i<img_L_desc.rows; i++ )
    if(matches[i].distance<= max(2*max_distance, 0.02))
    {
      good_matches.push_back(matches[i]);
    }

    //drawing only the good matches
    Mat img_matches;
   //cv:: drawMatches(imgU_L,kpl,imgU_R,kpr,good_matches, img_matches, Scalar::all(-1), vector<char>(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    drawMatches( imgU_L, kpl, imgU_R, kpr,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    vector<DMatch>::iterator it;
    
    imshow( "Good Matches", img_matches );
    for( int i = 0; i < (int)good_matches.size(); i++ )
    { printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx ); }
    waitKey(0); 
    vector<Point3f> feature_disparity;
    vector<Point3f> dst; 
    //disparity calculation at best matches
    Mat disparity_img(img_L.size().height, img_L.size().width, CV_8UC1, Scalar(0));
    
//   float xLmin = 0, yLmin = 0, xLmax = 300, yLmax = 300;
//   float xRmin = 0, yRmin = 0, xRmax = 300, yRmax = 300;
     for (it = good_matches.begin(); it != good_matches.end(); ++it) {
        // Get index
        float xL = kpl[it->queryIdx].pt.x;
        float yL = kpl[it->queryIdx].pt.y;
        float xR = kpr[it->trainIdx].pt.x;
        float yR  = kpr[it->trainIdx].pt.y; 
        // Calculate disparity & store in image
  //      if(xL >=xLmin && xL<=xLmax && xR >=xRmin && xR<=xRmax && yL >=yLmin && yL<=yLmax)
        float disparity = abs(xL - xR);
        //cout<<disparity<<endl;
        disparity_img.at<uchar>(yL, xL) = disparity;
        cout<<"disparity"<<disparity<<endl;
        feature_disparity.push_back(Point3f(xL,yL,disparity));
    }

    
    double minValue, maxValue;
    minMaxLoc(disparity_img, &minValue, &maxValue); //0
        double alpha = 255 / (maxValue - minValue); //scale factor, default=1
    double beta = 0;                            //optional beta added to the scaled values

    disparity_img.convertTo(disparity_img, CV_8UC1, alpha, beta);
  cout<<disparity_img.size<<endl;
    imshow("disparity",disparity_img);
    waitKey(0);

//now the 3d reprojection for the coordinates

   // Mat xyz;
    
perspectiveTransform(feature_disparity,dst,Q);
fsrc<<"feature point coordinates-"<<dst;


     //reprojectImageTo3D(disparity_img, xyz, Q, (bool)false,CV_32F);
    // cout<<xyz<<endl;
     //fsrc<<"xyz"<<xyz;
     //imshow("xyz",xyz);
    // waitKey(0);
         //xyz = cv::Mat(dst, true);
     //imshow("xyz",xyz);
     //waitKey(0);
  cout<<"reprojection done"<<endl;
//depth estimation of the feature points
double depth;
  Mat depth_img(img_L.size().height, img_L.size().width, CV_8UC1, Scalar(0));
        
        for(int i=0; i<img_L.rows;i++)
        {
         for(int j=0; j<img_L.cols; j++)
        {
          if(disparity_img.at<uchar>(i,j)!=0)
         depth = (1.05*1.3)/ (disparity_img.at<uchar>(i,j)); //1.05 -approx baseline //1.3 focal length
         depth_img.at<uchar>(i,j) = depth;
         //cout<<depth_img<<endl;
         //imshow("depth",depth_img);
         //waitKey(0);
         fd<<"depth "<<depth;
         
        
         //printf("Depth at feature point(%d,%d):%f",i,j,depth);
         //cout<<"depth is:"<<depth_img.at<uchar>(j,i);
        }
        }
     cout<<"done"<<endl;
     return(0);
}
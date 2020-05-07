#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT) {
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1) {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2) {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check whether point is within current bounding box
            if (smallerBox.contains(pt)) {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check whether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1) { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }
    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait) {
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1) {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2) {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i) {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 0);
    cv::resizeWindow(windowName, 800, 800);
    cv::imshow(windowName, topviewImg);

    if(bWait) {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, 
                              std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches) {

    std::vector<double> euclideanDistance;

    for(auto it1 = kptMatches.begin(); it1 != kptMatches.end(); ++it1) {
        auto currentKeypoint = kptsCurr[it1->trainIdx];

        if(boundingBox.roi.contains(currentKeypoint.pt)) {
            auto previousKeypoint = kptsPrev[it1->queryIdx];

            euclideanDistance.push_back(cv::norm(currentKeypoint.pt - previousKeypoint.pt));
        }
    }

    double distanceMean = std::accumulate(euclideanDistance.begin(), euclideanDistance.end(), 0.0) / euclideanDistance.size();

    for(auto it2 = kptMatches.begin(); it2 != kptMatches.end(); ++it2) {
        auto currentKeypoint = kptsCurr[it2->trainIdx];

        if(boundingBox.roi.contains(currentKeypoint.pt)) {
            auto previousKeypoint = kptsPrev[it2->queryIdx];

            double kptDistance = cv::norm(currentKeypoint.pt - previousKeypoint.pt);
            double distanceTolerance = distanceMean * 1.2;

            if(kptDistance < distanceTolerance) {
                boundingBox.kptMatches.push_back(*it2);
            }
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg) {
    
    // compute distance ratios between all matched keypoints 
    std::vector<double> distanceRatios;

    for(auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1) {

        // get current keypoint and its matched keypoint in the previous frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for(auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2) {

            // minimum required distance
            double minDistance = 100.0;

            // get current keypoint and its matched keypoint in the curren frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distance and distance ratios
            double distanceCurrent = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distancePrevious = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if(distancePrevious > std::numeric_limits<double>::epsilon() && distanceCurrent >= minDistance) {

                double distRatio = distanceCurrent / distancePrevious;
                distanceRatios.push_back(distRatio);

            }
        }
    }

    if(distanceRatios.size() == 0) {
        TTC = NAN;
        return;
    }

    std::sort(distanceRatios.begin(), distanceRatios.end());

    long medIndex = floor(distanceRatios.size() / 2.0);

    double medianDistanceRatio = distanceRatios.size() % 2 == 0 ? 
                                 (distanceRatios[medIndex - 1] + distanceRatios[medIndex + 1]) / 2.0 : distanceRatios[medIndex];

    TTC = - (1.0 / frameRate) / (1 - medianDistanceRatio); 
    cout<<"TTC Camera: "<<TTC<<endl;
}


void clusterHelper(int indice, const std::vector<std::vector<float>> points, std::vector<int> &cluster, 
                   std::vector<bool> &processed, KdTree* tree, float distanceTol) {

    processed[indice] = true;
    cluster.push_back(indice);

    std::vector<int> nearest = tree->search(points[indice], distanceTol);

    for (int id : nearest) {

        if (!processed[id])
            clusterHelper(id, points, cluster, processed, tree, distanceTol);
    }
}

std::vector<std::vector<int>> euclideanCluster(const std::vector<std::vector<float>>& points, KdTree* tree, float distanceTol) {

    std::vector<std::vector<int>> clusters;
    std::vector<bool> processed (points.size(), false);

    int i = 0;
    while(i < points.size()) {

        if (processed[i]) {
          i++;
          continue;
        }

        std::vector<int> cluster;
        clusterHelper(i, points, cluster, processed, tree, distanceTol);

        clusters.push_back(cluster);
        i++;
    }

    return clusters;
}



std::vector<LidarPoint> removeOutliers(const std::vector<LidarPoint> &lidarPoints, float distanceTol) {

    KdTree* tree = new KdTree;

    std::vector<std::vector<float>> points;

    int i = 0;
    for(auto point : lidarPoints) {
        std::vector<float> lidarP {static_cast<float>(point.x), static_cast<float>(point.y), static_cast<float>(point.z)};
        tree->insert(lidarP, i++);
        points.push_back(lidarP);
    }

    std::vector<std::vector<int>> clusters = euclideanCluster(points, tree, distanceTol);

    std::vector<LidarPoint> finalLidar;
    for(auto cluster : clusters) {
        std::vector<LidarPoint> tempLidar;
        for(auto indice : cluster) {
            tempLidar.push_back(lidarPoints[indice]);
        }

        if(tempLidar.size() > finalLidar.size()) {
            finalLidar = std::move(tempLidar);
        }
    }

    return finalLidar;
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev, std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC) {

    // double laneWidth = 4.0; // assume width of ego lane

    // find closest distance to LiDAR Points within ego lane
    double minXPrev = 1e9, minXCurr = 1e9;
    float distanceTol = 0.1;

    // performing Euclidean Clustering to preprocess LiDAR Points
    vector<LidarPoint> newlidarPointsPrev = removeOutliers(lidarPointsPrev, distanceTol);
    vector<LidarPoint> newlidarPointsCurr = removeOutliers(lidarPointsCurr, distanceTol);

    // vectors to store distances
    vector<double> xPrev, xCurr;

    for(auto it1 = newlidarPointsPrev.begin(); it1!=newlidarPointsPrev.end(); ++it1) {
        xPrev.push_back(it1->x);
    }

    for(auto it = newlidarPointsCurr.begin(); it!=newlidarPointsCurr.end(); ++it) {
        xCurr.push_back(it->x);
    }

    // finding the median of the distances
    // sorting into ascending order
    if (xPrev.size() > 0) {
        std::sort(xPrev.begin(), xPrev.end());
        long medIndex = floor(xPrev.size() / 2.0);

        minXPrev = xPrev.size() % 2 == 0 ? (xPrev[medIndex - 1] + xPrev[medIndex + 1]) / 2.0 : xPrev[medIndex];
    }

    if(xCurr.size() > 0) {
        std::sort(xCurr.begin(), xCurr.end());
        long medIndex = floor(xCurr.size() / 2.0);

        minXCurr = xCurr.size() % 2 == 0 ? (xCurr[medIndex - 1] + xCurr[medIndex + 1]) / 2.0 : xCurr[medIndex];
    }

    // compute TTC from both measurements
    TTC = minXCurr * (1.0 / frameRate) / (minXPrev - minXCurr);
    cout<<"TTC LiDAR: "<<TTC<<endl;
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame) {
    
    // looping through Bounding Boxes of Previous Frame
    for(auto itr1 = prevFrame.boundingBoxes.begin(); itr1!= prevFrame.boundingBoxes.end(); ++itr1){

        vector<vector<cv::DMatch>::iterator> keypointMatches;

        for(auto itr2 = matches.begin(); itr2 != matches.end(); ++itr2) {

            // extracting the previous frame keypoint index
            int prevFrameKptIdx = itr2->queryIdx;

            // checking if that particular bounding box containes the previous frame keypoint 
            if(itr1->roi.contains(prevFrame.keypoints.at(prevFrameKptIdx).pt)) {

                // pushing the keypoint matches 
                keypointMatches.push_back(itr2);
            }
        }

        // initializing a multimap 
        multimap<int, int> currentFrameAll;

        // looping through the extracted keypoint matches
        for(auto itr3 = keypointMatches.begin(); itr3 != keypointMatches.end(); ++itr3) {

            // extracting the current frame keypoint index from the extracted keypoint matches
            int currFrameKptIdx = (*itr3)->trainIdx;

            // looping through the current frame bounding boxes 
            for(auto itr4 = currFrame.boundingBoxes.begin(); itr4 != currFrame.boundingBoxes.end(); ++itr4) {

                // checking if current frame bounding box contains that keypoint or not
                if(itr4->roi.contains(currFrame.keypoints.at(currFrameKptIdx).pt)) {

                    // if found, storing the bounding box id 
                    int currBoxID = itr4->boxID;

                    // pushing the bounding box id and also the current frame keypoint index
                    currentFrameAll.insert(pair<int, int>(currBoxID, currFrameKptIdx));
                }
            }
        }

        int max = 0;
        int currFrameBoxID = 10000;

        if(currentFrameAll.size() > 0) {
            for(auto itr5 = currentFrameAll.begin(); itr5 != currentFrameAll.end(); ++itr5) {

                // checking for the count of the bounding boxes 
                if(currentFrameAll.count(itr5->first) > max) {

                    // storing the maximum count of the bounding box
                    max = currentFrameAll.count(itr5->first);
                    currFrameBoxID = itr5->first;
                }
            }

            // storing the previous frame bounding box id and its corresponding current frame bounding box id
            bbBestMatches.insert(pair<int, int>(itr1->boxID, currFrameBoxID));
        }
    }
}

#ifndef SLIDINGWINDOWSURF_H
#define SLIDINGWINDOWSURF_H

#include "opencv2/opencv.hpp"

cv::Rect findBestROI(cv::Mat sceneImage, cv::Mat modelImage, cv::Size patchSize, cv::Point2f& centerRelativeToPatch);

#endif
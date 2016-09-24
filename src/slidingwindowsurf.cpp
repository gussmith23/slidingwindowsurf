#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"


using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;


/// <summary>
/// Return a confidence score for a set of match distances.
/// </summary>
/// <param name="Distances">The match distances.</param>
/// <param name="Mask"></param>
/// <returns></returns>
float computeScore(const vector<DMatch> matches)
{
	float score = 0.0f;

	for (auto it = matches.begin(); it != matches.end(); ++it)
		score += it->distance;

	float final_score = 1.0f - (((float)matches.size() - score) / (float)matches.size());
	return final_score;
}

void surf(Mat modelImage, Mat observedImage, Point2f& centerOfObject, double& confidence)
{
	Ptr<DescriptorExtractor> descriptorExtractor = SURF::create(1000, 4, 3, false, true);

	vector<KeyPoint> modelKeyPoints;
	Mat modelDescriptors;
	vector<KeyPoint> observedKeyPoints;
	Mat observedDescriptors;
	Mat indices;

//	surfExtractor->detectAndCompute(modelImage, Mat(), modelKeyPoints, modelDescriptors);
	//surfExtractor->detectAndCompute(observedImage, Mat(), observedKeyPoints, observedDescriptors);
	descriptorExtractor->detectAndCompute(modelImage, Mat(), modelKeyPoints, modelDescriptors);
	descriptorExtractor->detectAndCompute(observedImage, Mat(), observedKeyPoints, observedDescriptors);

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
	std::vector< DMatch > matches;
	matcher->match(modelDescriptors, observedDescriptors, matches);
	double max_dist = 0; double min_dist = 100;
	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < modelDescriptors.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	std::vector< DMatch > good_matches;
	for (int i = 0; i < modelDescriptors.rows; i++)
	{
		if (matches[i].distance <= 3 * min_dist)
		{
			good_matches.push_back(matches[i]);
		}
	}

	//-- Localize the object
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;
	for (size_t i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(modelKeyPoints[good_matches[i].queryIdx].pt);
		scene.push_back(observedKeyPoints[good_matches[i].trainIdx].pt);
	}
	Mat H = findHomography(obj, scene, RANSAC);
	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> modelCenter = {cvPoint(modelImage.cols / 2, modelImage.rows / 2)};
	std::vector<Point2f> observedCenter(1);
	//perspectiveTransform(modelCenter, observedCenter, H);

	centerOfObject = observedCenter[0];
	confidence = computeScore(good_matches);
}

Rect findBestROI(Mat sceneImage, Mat modelImage, Size patchSize, Point2f& centerRelativeToPatch) {
	Point currentLoc = Point(0, 0);
	tuple<Rect, double> bestROI = tuple<Rect, double>(Rect(), 0.0f);
	Size stepSize = Size((int)((float)patchSize.width / 2.0), (int)((float)patchSize.height / 2.0));

	while (currentLoc.y + patchSize.height < sceneImage.rows)
	{
		if (currentLoc.y + patchSize.height > sceneImage.rows)
			currentLoc.y = sceneImage.rows - patchSize.height;

		while (currentLoc.x + patchSize.width < sceneImage.cols)
		{
			if (currentLoc.x + patchSize.width > sceneImage.cols)
				currentLoc.x = sceneImage.cols - patchSize.width;

			double confidence;
			Rect thisROI = Rect(currentLoc, patchSize);
			Mat roi = sceneImage(thisROI);
			Point2f tmpCenter;
			surf(modelImage, roi, tmpCenter, confidence);
			if (confidence > get<1>(bestROI))
			{
				bestROI = tuple<Rect, double>(thisROI, confidence);
				centerRelativeToPatch = tmpCenter;
			}
			currentLoc.x += stepSize.width;
		}
		currentLoc.x = 0;
		currentLoc.y += stepSize.height;
	}

	return get<0>(bestROI);
}
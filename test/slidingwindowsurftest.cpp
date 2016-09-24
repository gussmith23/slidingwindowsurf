#define CATCH_CONFIG_MAIN  
#include "catch.hpp"
#include "slidingwindowsurf.h"
#include "opencv2/opencv.hpp"

using namespace cv;

TEST_CASE("test") {
	Mat scene = imread("resources/Wegmans-144.jpg");
	Mat model = imread("resources/cereal_reese_puffs_1102.jpg");
	Rect bestROI = findBestROI(scene, model, Size(800, 800), Point2f());
	imshow("out", scene(bestROI));
	waitKey();
	REQUIRE(bestROI == Rect(400, 1200, 800, 800));
}

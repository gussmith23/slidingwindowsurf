#define CATCH_CONFIG_MAIN  
#include "catch.hpp"
#include "slidingwindowsurf.h"
#include "opencv2/opencv.hpp"

using namespace cv;

TEST_CASE("reeses puffs") {
	Mat scene = imread("resources/Wegmans-144.jpg", IMREAD_GRAYSCALE);
	Mat model = imread("resources/cereal_reese_puffs_1102.jpg", IMREAD_GRAYSCALE);
	Rect bestROI = findBestROI(scene, model, Size(800, 800), Point2f());
	REQUIRE(bestROI.x >= 200);
	REQUIRE(bestROI.x <= 1000);
	REQUIRE(bestROI.y >= 1000);
	REQUIRE(bestROI.y <= 1700);
}

TEST_CASE("chex") {
	Mat scene = imread("resources/Wegmans-144.jpg", IMREAD_GRAYSCALE);
	Mat model = imread("resources/chex.jpg", IMREAD_GRAYSCALE);
	Rect bestROI = findBestROI(scene, model, Size(800, 800), Point2f());
	REQUIRE(bestROI.x >= 1300);
	REQUIRE(bestROI.x <= 2200);
	REQUIRE(bestROI.y >= 250);
	REQUIRE(bestROI.y <= 850);
}

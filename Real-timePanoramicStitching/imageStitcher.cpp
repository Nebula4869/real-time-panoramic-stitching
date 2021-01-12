#include <stdio.h>
#include <windows.h>
#include <vector>
#include <opencv2/opencv.hpp>


void stitchImage(const char *imagePaths[], int numImages) {
	cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(cv::Stitcher::PANORAMA);
	std::vector<cv::Mat> imgs;
	cv::Mat pano;
	cv::Mat img;

	for (int i = 0; i < numImages; i++) {
		img = cv::imread(imagePaths[i]);
		if (img.empty()) {
			printf("Can't read%s\n", imagePaths[i]);
			continue;
		}
		imgs.push_back(img);
	}

	long t0 = GetTickCount();
	cv::Stitcher::Status status = stitcher->stitch(imgs, pano);
	long t1 = GetTickCount();
	printf("Time Cost: %dms\n", t1 - t0);

	if (status != cv::Stitcher::OK) {
		printf("Can't stitch images, error code = %d\n", int(status));
	}
	else {
		cv::imwrite("Panorama.jpg", pano);
		cv::namedWindow("Panorama", cv::WINDOW_NORMAL);
		cv::imshow("Panorama", pano);
		cv::waitKey();
	}

	cv::destroyAllWindows();
}
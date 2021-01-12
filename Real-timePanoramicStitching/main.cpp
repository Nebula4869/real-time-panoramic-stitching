#include "cameraStitcher.h"
#include "imageStitcher.h"


void main()
{
	const char *imagePaths[] = { "A.JPG", "B.JPG", "C.JPG", "D.JPG" };
	stitchImage(imagePaths, 4);
	const char cameraIDs[] = { 0, 1 };
	stitchCamera(cameraIDs, 2, 1280, 720);
}
#include <windows.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/warpers.hpp"

#define LOGLN(msg) std::cout << msg << std::endl


#pragma region Algorithim Parasmeters
bool preview = false;
bool try_cuda = false;
double work_megapix = 0.6;
double seam_megapix = 0.1;
double compose_megapix = -1;
float conf_thresh = 1.f;
std::string features_type = "orb";
float match_conf = 0.3f;
std::string matcher_type = "homography";
std::string estimator_type = "homography";
std::string ba_cost_func = "ray";
std::string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;
cv::detail::WaveCorrectKind wave_correct = cv::detail::WAVE_CORRECT_HORIZ;
bool save_graph = false;
std::string save_graph_to;
std::string warp_type = "spherical";
int expos_comp_type = cv::detail::ExposureCompensator::GAIN_BLOCKS;
int expos_comp_nr_feeds = 1;
int expos_comp_nr_filtering = 2;
int expos_comp_block_size = 32;
std::string seam_find_type = "gc_color";
int blend_type = cv::detail::Blender::MULTI_BAND;
int timelapse_type = cv::detail::Timelapser::AS_IS;
float blend_strength = 5;
int range_width = -1;
#pragma endregion


void stitchCamera(const char cameraIDs[], int numCameras, int frameWidth, int frameHeight) {
#pragma region Intermediate Variables
	bool findFeatures = true;

	cv::VideoCapture cap;
	std::vector<cv::VideoCapture> caps;

	float warpedImageScale;
	double workScale, seamScale, composeScale, seamWorkAspect, composeWorkAspect;
	bool isWorkScaleSet, isSeamScaleSet, isComposeScaleSet, errorFlag=false;

	cv::Mat img, fullImg, imgWarped, imgWarped_S, mask, maskWarped, dilatedMask, seamMask;

	cv::Ptr<cv::detail::RotationWarper> warper;
	cv::Ptr<cv::WarperCreator> warperCreator;
	cv::Ptr<cv::detail::ExposureCompensator> compensator;

	std::vector<int> indices;
	std::vector<cv::Point> corners(numCameras);
	std::vector<cv::Size> sizes(numCameras);
	std::vector<cv::Size> fullImgSizes(numCameras);
	std::vector<cv::Mat> cameraImages(numCameras);
	std::vector<cv::UMat> masksWarped(numCameras);
	std::vector<cv::detail::CameraParams> cameras;
#pragma endregion

	// Initialize all cameras
	for (int i = 0; i < numCameras; ++i) {
		cap = cv::VideoCapture(int(cameraIDs[i]));
		cap.set(cv::CAP_PROP_FRAME_WIDTH, frameWidth);
		cap.set(cv::CAP_PROP_FRAME_HEIGHT, frameHeight);
		caps.push_back(cap);
	}

	while (true) {
		long t0 = GetTickCount();

		// Check if all cameras are opened
		for (int i = 0; i < numCameras; ++i) {
			if (!caps[i].isOpened()) {
				LOGLN("Open camera " << int(cameraIDs[i]) << " failed");
				errorFlag = true;
			}
		}
		if (errorFlag) break;

#pragma region Feature extraction and transformation parameter calculation
		// Only executed on the first frame and when Enter is pressed
		if (findFeatures) {
			workScale = 1;
			seamScale = 1;
			composeScale = 1;
			isWorkScaleSet = false;
			isSeamScaleSet = false;
			isComposeScaleSet = false;

			LOGLN("Finding features...");

			cv::Ptr<cv::Feature2D> finder;
			if (features_type == "orb") {
				finder = cv::ORB::create();
			}
			else if (features_type == "akaze") {
				finder = cv::AKAZE::create();
			}
			else {
				LOGLN("Unknown 2D features type: '" << features_type);
				errorFlag = true;
			}
			if (errorFlag) break;

			std::vector<cv::detail::ImageFeatures> features(numCameras);
			std::vector<cv::Mat> images(numCameras);

			seamWorkAspect = 1;

			for (int i = 0; i < numCameras; ++i) {
				caps[i].read(fullImg);
				fullImgSizes[i] = fullImg.size();

				if (fullImg.empty()) {
					LOGLN("Read frame failed");
					errorFlag = true;
				}
				if (errorFlag) break;

				if (work_megapix < 0) {
					img = fullImg;
					workScale = 1;
					isWorkScaleSet = true;
				}
				else {
					if (!isWorkScaleSet) {
						workScale = std::min(1.0, sqrt(work_megapix * 1e6 / fullImg.size().area()));
						isWorkScaleSet = true;
					}
					cv::resize(fullImg, img, cv::Size(), workScale, workScale, cv::INTER_LINEAR_EXACT);
				}
				if (!isSeamScaleSet) {
					seamScale = std::min(1.0, sqrt(seam_megapix * 1e6 / fullImg.size().area()));
					seamWorkAspect = seamScale / workScale;
					isSeamScaleSet = true;
				}

				computeImageFeatures(finder, img, features[i]);
				features[i].img_idx = i;
				LOGLN("Features in image #" << i + 1 << ": " << features[i].keypoints.size());

				cv::resize(fullImg, img, cv::Size(), seamScale, seamScale, cv::INTER_LINEAR_EXACT);
				images[i] = img.clone();
			}
			fullImg.release();
			img.release();

			LOGLN("Pairwise matching");

			std::vector<cv::detail::MatchesInfo> pairwise_matches;
			cv::Ptr<cv::detail::FeaturesMatcher> matcher;
			if (matcher_type == "affine")
				matcher = cv::makePtr<cv::detail::AffineBestOf2NearestMatcher>(false, try_cuda, match_conf);
			else if (range_width == -1)
				matcher = cv::makePtr<cv::detail::BestOf2NearestMatcher>(try_cuda, match_conf);
			else
				matcher = cv::makePtr<cv::detail::BestOf2NearestRangeMatcher>(range_width, try_cuda, match_conf);

			(*matcher)(features, pairwise_matches);
			matcher->collectGarbage();

			// Leave only images we are sure are from the same panorama
			indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
			std::vector<cv::Mat> img_subset;
			std::vector<cv::Size> full_img_sizes_subset;
			for (int i = 0; i < indices.size(); ++i) {
				img_subset.push_back(images[indices[i]]);
				full_img_sizes_subset.push_back(fullImgSizes[indices[i]]);
			}

			images = img_subset;
			fullImgSizes = full_img_sizes_subset;

			// Check if we still have enough images
			if (indices.size() < numCameras) {
				LOGLN("Insufficient stitchable images");
				continue;
			}

			cv::Ptr<cv::detail::Estimator> estimator;
			if (estimator_type == "affine")
				estimator = cv::makePtr<cv::detail::AffineBasedEstimator>();
			else
				estimator = cv::makePtr<cv::detail::HomographyBasedEstimator>();

			if (!(*estimator)(features, pairwise_matches, cameras)) {
				LOGLN("Homography estimation failed");
				continue;
			}

			for (int i = 0; i < cameras.size(); ++i) {
				cv::Mat R;
				cameras[i].R.convertTo(R, CV_32F);
				cameras[i].R = R;
				LOGLN("Initial camera intrinsics #" << indices[i] + 1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
			}

			cv::Ptr<cv::detail::BundleAdjusterBase> adjuster;
			if (ba_cost_func == "reproj") adjuster = cv::makePtr<cv::detail::BundleAdjusterReproj>();
			else if (ba_cost_func == "ray") adjuster = cv::makePtr<cv::detail::BundleAdjusterRay>();
			else if (ba_cost_func == "affine") adjuster = cv::makePtr<cv::detail::BundleAdjusterAffinePartial>();
			else if (ba_cost_func == "no") adjuster = cv::makePtr<cv::detail::NoBundleAdjuster>();
			else {
				LOGLN("Unknown bundle adjustment cost function: '" << ba_cost_func);
				continue;
			}
			adjuster->setConfThresh(conf_thresh);
			cv::Mat_<uchar> refine_mask = cv::Mat::zeros(3, 3, CV_8U);
			if (ba_refine_mask[0] == 'x') refine_mask(0, 0) = 1;
			if (ba_refine_mask[1] == 'x') refine_mask(0, 1) = 1;
			if (ba_refine_mask[2] == 'x') refine_mask(0, 2) = 1;
			if (ba_refine_mask[3] == 'x') refine_mask(1, 1) = 1;
			if (ba_refine_mask[4] == 'x') refine_mask(1, 2) = 1;
			adjuster->setRefinementMask(refine_mask);
			if (!(*adjuster)(features, pairwise_matches, cameras)) {
				LOGLN("Camera parameters adjusting failed");
				continue;
			}

			// Find median focal length

			std::vector<double> focals;
			for (int i = 0; i < cameras.size(); ++i) {
				LOGLN("Camera #" << indices[i] + 1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
				focals.push_back(cameras[i].focal);
			}

			sort(focals.begin(), focals.end());

			if (focals.size() % 2 == 1)
				warpedImageScale = static_cast<float>(focals[focals.size() / 2]);
			else
				warpedImageScale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

			if (do_wave_correct) {
				std::vector<cv::Mat> rmats;
				for (int i = 0; i < cameras.size(); ++i)
					rmats.push_back(cameras[i].R.clone());
				waveCorrect(rmats, wave_correct);
				for (int i = 0; i < cameras.size(); ++i)
					cameras[i].R = rmats[i];
			}

			LOGLN("Warping images (auxiliary)... ");

			std::vector<cv::UMat> images_warped(numCameras);

			std::vector<cv::UMat> masks(numCameras);

			// Prepare images masks
			for (int i = 0; i < numCameras; ++i) {
				masks[i].create(images[i].size(), CV_8U);
				masks[i].setTo(cv::Scalar::all(255));
			}

			// Warp images and their masks
			if (warp_type == "plane")
				warperCreator = cv::makePtr<cv::PlaneWarper>();
			else if (warp_type == "affine")
				warperCreator = cv::makePtr<cv::AffineWarper>();
			else if (warp_type == "cylindrical")
				warperCreator = cv::makePtr<cv::CylindricalWarper>();
			else if (warp_type == "spherical")
				warperCreator = cv::makePtr<cv::SphericalWarper>();
			else if (warp_type == "fisheye")
				warperCreator = cv::makePtr<cv::FisheyeWarper>();
			else if (warp_type == "stereographic")
				warperCreator = cv::makePtr<cv::StereographicWarper>();
			else if (warp_type == "compressedPlaneA2B1")
				warperCreator = cv::makePtr<cv::CompressedRectilinearWarper>(2.0f, 1.0f);
			else if (warp_type == "compressedPlaneA1.5B1")
				warperCreator = cv::makePtr<cv::CompressedRectilinearWarper>(1.5f, 1.0f);
			else if (warp_type == "compressedPlanePortraitA2B1")
				warperCreator = cv::makePtr<cv::CompressedRectilinearPortraitWarper>(2.0f, 1.0f);
			else if (warp_type == "compressedPlanePortraitA1.5B1")
				warperCreator = cv::makePtr<cv::CompressedRectilinearPortraitWarper>(1.5f, 1.0f);
			else if (warp_type == "paniniA2B1")
				warperCreator = cv::makePtr<cv::PaniniWarper>(2.0f, 1.0f);
			else if (warp_type == "paniniA1.5B1")
				warperCreator = cv::makePtr<cv::PaniniWarper>(1.5f, 1.0f);
			else if (warp_type == "paniniPortraitA2B1")
				warperCreator = cv::makePtr<cv::PaniniPortraitWarper>(2.0f, 1.0f);
			else if (warp_type == "paniniPortraitA1.5B1")
				warperCreator = cv::makePtr<cv::PaniniPortraitWarper>(1.5f, 1.0f);
			else if (warp_type == "mercator")
				warperCreator = cv::makePtr<cv::MercatorWarper>();
			else if (warp_type == "transverseMercator")
				warperCreator = cv::makePtr<cv::TransverseMercatorWarper>();

			if (!warperCreator) {
				LOGLN("Can't create the following warper '" << warp_type);
				continue;
			}

			warper = warperCreator->create(static_cast<float>(warpedImageScale * seamWorkAspect));

			for (int i = 0; i < numCameras; ++i) {
				cv::Mat_<float> K;
				cameras[i].K().convertTo(K, CV_32F);
				float swa = (float)seamWorkAspect;
				K(0, 0) *= swa; K(0, 2) *= swa;
				K(1, 1) *= swa; K(1, 2) *= swa;

				corners[i] = warper->warp(images[i], K, cameras[i].R, cv::INTER_LINEAR, cv::BORDER_REFLECT, images_warped[i]);
				sizes[i] = images_warped[i].size();

				warper->warp(masks[i], K, cameras[i].R, cv::INTER_NEAREST, cv::BORDER_CONSTANT, masksWarped[i]);
			}

			std::vector<cv::UMat> images_warped_f(numCameras);
			for (int i = 0; i < numCameras; ++i)
				images_warped[i].convertTo(images_warped_f[i], CV_32F);

			LOGLN("Compensating exposure...");

			compensator = cv::detail::ExposureCompensator::createDefault(expos_comp_type);
			if (dynamic_cast<cv::detail::GainCompensator*>(compensator.get())) {
				cv::detail::GainCompensator* gcompensator = dynamic_cast<cv::detail::GainCompensator*>(compensator.get());
				gcompensator->setNrFeeds(expos_comp_nr_feeds);
			}

			if (dynamic_cast<cv::detail::ChannelsCompensator*>(compensator.get())) {
				cv::detail::ChannelsCompensator* ccompensator = dynamic_cast<cv::detail::ChannelsCompensator*>(compensator.get());
				ccompensator->setNrFeeds(expos_comp_nr_feeds);
			}

			if (dynamic_cast<cv::detail::BlocksCompensator*>(compensator.get())) {
				cv::detail::BlocksCompensator* bcompensator = dynamic_cast<cv::detail::BlocksCompensator*>(compensator.get());
				bcompensator->setNrFeeds(expos_comp_nr_feeds);
				bcompensator->setNrGainsFilteringIterations(expos_comp_nr_filtering);
				bcompensator->setBlockSize(expos_comp_block_size, expos_comp_block_size);
			}

			compensator->feed(corners, images_warped, masksWarped);

			LOGLN("Finding seams...");

			cv::Ptr<cv::detail::SeamFinder> seam_finder;
			if (seam_find_type == "no")
				seam_finder = cv::makePtr<cv::detail::NoSeamFinder>();
			else if (seam_find_type == "voronoi")
				seam_finder = cv::makePtr<cv::detail::VoronoiSeamFinder>();
			else if (seam_find_type == "gc_color") {
				seam_finder = cv::makePtr<cv::detail::GraphCutSeamFinder>(cv::detail::GraphCutSeamFinderBase::COST_COLOR);
			}
			else if (seam_find_type == "gc_colorgrad") {
				seam_finder = cv::makePtr<cv::detail::GraphCutSeamFinder>(cv::detail::GraphCutSeamFinderBase::COST_COLOR_GRAD);
			}
			else if (seam_find_type == "dp_color")
				seam_finder = cv::makePtr<cv::detail::DpSeamFinder>(cv::detail::DpSeamFinder::COLOR);
			else if (seam_find_type == "dp_colorgrad")
				seam_finder = cv::makePtr<cv::detail::DpSeamFinder>(cv::detail::DpSeamFinder::COLOR_GRAD);
			if (!seam_finder) {
				LOGLN("Can't create the following seam finder '" << seam_find_type);
				continue;
			}

			seam_finder->find(images_warped_f, corners, masksWarped);

			// Release unused memory
			images.clear();
			images_warped.clear();
			images_warped_f.clear();
			masks.clear();

			LOGLN("Compositing...");

			composeWorkAspect = 1;
			findFeatures = false;
		}
#pragma endregion

#pragma region Image blending and display
		cv::Ptr<cv::detail::Blender> blender;
		for (int img_idx = 0; img_idx < numCameras; ++img_idx) {
			LOGLN("Compositing image #" << indices[img_idx] + 1);

			// Read image and resize it if necessary
			caps[img_idx].read(fullImg);
			cameraImages[img_idx] = fullImg;

			if (!isComposeScaleSet) {
				if (compose_megapix > 0)
					composeScale = std::min(1.0, sqrt(compose_megapix * 1e6 / fullImg.size().area()));
				isComposeScaleSet = true;

				// Compute relative scales
				composeWorkAspect = composeScale / workScale;

				// Update warped image scale
				warpedImageScale *= static_cast<float>(composeWorkAspect);
				warper = warperCreator->create(warpedImageScale);

				// Update corners and sizes
				for (int i = 0; i < numCameras; ++i) {
					// Update intrinsics
					cameras[i].focal *= composeWorkAspect;
					cameras[i].ppx *= composeWorkAspect;
					cameras[i].ppy *= composeWorkAspect;

					// Update corner and size
					cv::Size sz = fullImgSizes[i];
					if (std::abs(composeScale - 1) > 1e-1) {
						sz.width = cvRound(fullImgSizes[i].width * composeScale);
						sz.height = cvRound(fullImgSizes[i].height * composeScale);
					}

					cv::Mat K;
					cameras[i].K().convertTo(K, CV_32F);
					cv::Rect roi = warper->warpRoi(sz, K, cameras[i].R);
					corners[i] = roi.tl();
					sizes[i] = roi.size();
				}
			}
			if (abs(composeScale - 1) > 1e-1)
				cv::resize(fullImg, img, cv::Size(), composeScale, composeScale, cv::INTER_LINEAR_EXACT);
			else
				img = fullImg;
			fullImg.release();
			cv::Size img_size = img.size();

			cv::Mat K;
			cameras[img_idx].K().convertTo(K, CV_32F);

			// Warp the current image
			warper->warp(img, K, cameras[img_idx].R, cv::INTER_LINEAR, cv::BORDER_REFLECT, imgWarped);

			// Warp the current image mask
			mask.create(img_size, CV_8U);
			mask.setTo(cv::Scalar::all(255));
			warper->warp(mask, K, cameras[img_idx].R, cv::INTER_NEAREST, cv::BORDER_CONSTANT, maskWarped);

			// Compensate exposure
			compensator->apply(img_idx, corners[img_idx], imgWarped, maskWarped);

			imgWarped.convertTo(imgWarped_S, CV_16S);
			imgWarped.release();
			img.release();
			mask.release();

			cv::dilate(masksWarped[img_idx], dilatedMask, cv::Mat());
			cv::resize(dilatedMask, seamMask, maskWarped.size(), 0, 0, cv::INTER_LINEAR_EXACT);
			maskWarped = seamMask & maskWarped;

			if (!blender) {
				blender = cv::detail::Blender::createDefault(blend_type, try_cuda);
				cv::Size dst_sz = cv::detail::resultRoi(corners, sizes).size();
				float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
				if (blend_width < 1.f)
					blender = cv::detail::Blender::createDefault(cv::detail::Blender::NO, try_cuda);
				else if (blend_type == cv::detail::Blender::MULTI_BAND) {
					cv::detail::MultiBandBlender* mb = dynamic_cast<cv::detail::MultiBandBlender*>(blender.get());
					mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
					LOGLN("Multi-band blender, number of bands: " << mb->numBands());
				}
				else if (blend_type == cv::detail::Blender::FEATHER) {
					cv::detail::FeatherBlender* fb = dynamic_cast<cv::detail::FeatherBlender*>(blender.get());
					fb->setSharpness(1.f / blend_width);
					LOGLN("Feather blender, sharpness: " << fb->sharpness());
				}
				blender->prepare(corners, sizes);
			}

			// Blend the current image
			blender->feed(imgWarped_S, maskWarped, corners[img_idx]);
		}

		cv::Mat result, result_mask;
		blender->blend(result, result_mask);
		result.convertTo(result, CV_8U);

		long t1 = GetTickCount();
		cv::putText(result, "FPS: " + std::to_string(int(1000 / (t1 - t0))), cv::Point(100, 100), cv::FONT_HERSHEY_COMPLEX, 1, (0, 255, 0));
		cv::imshow("Panaromic", result);

		int key_num = cv::waitKey(1);
		if (key_num == 13) {
			findFeatures = true;
		}
		else if (key_num == 27) {
			break;
		}
#pragma endregion
	}
}
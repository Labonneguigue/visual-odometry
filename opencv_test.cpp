#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>

class DetectorExtractorMatcher
{
public:

    void extractFeatures(cv::Mat& image, std::vector<cv::KeyPoint>& kps)
    {
        static const int cDefaultKeypointCount = 1500; ///< OpenCV default feature count
        static const int cImageBorderThreshold = 31; ///< Features are not detected on the border of the image, this specifies the border size
        static const int cPyramidFirstLevel = 0; ///< Needs to be set to 0
        static const int cBriefWta_k = 2; ///< Number of points used to compute one attribute of the BRIEF descriptor. Setting it to 2 means that we are comparing the brightness of 2 pixels
        static const int cKeypointScoreType = cv::ORB::HARRIS_SCORE;  ///< Either Harris score or FAST score. Harris produces better (more stable) keypoints, but it is slower
        static const int cKeypointFastThreshold = 20; ///< FAST corner detector threshold
        static const float cPyramidScaleFactor = 1.2F; ///< Scale in each image pyramid
        static const int   cPyramidLevelCount = 3; ///< Number of levels in the image pyramid
        static const int   cBriefPatchSize = 31; ///< Size of the image patch on which the descriptor is computed
        static const float cMaximumPatchSize = (cBriefPatchSize + 1U) * std::pow(cPyramidScaleFactor, static_cast<float>(cPyramidLevelCount + 1U)) + 1U; // Maximum size of a orb feature patch as if applied to level 0

        static cv::Ptr<cv::ORB> orb = cv::ORB::create(cDefaultKeypointCount
                                                    , cPyramidScaleFactor
                                                    , cPyramidLevelCount
                                                    , cImageBorderThreshold
                                                    , cPyramidFirstLevel
                                                    , cBriefWta_k
                                                    , cKeypointScoreType
                                                    , cBriefPatchSize
                                                    , cKeypointFastThreshold);

        orb->detect(image, keypoints[1]);
        kps = keypoints[1];
    }

    void matchFeatures()
    {

    }

private:

    std::vector<cv::KeyPoint> keypoints[2];
    std::vector<cv::DMatch>   matches;
    cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_HAMMING);
};

int main()
{
	cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);
	cv::Size winSize(200, 300);
    bool SET_CAMERA = false;
    // Open vid reader
    int delay = 20;
    cv::VideoCapture capture;
    if (SET_CAMERA) {
        capture.open(0); // open the default camera
    } else {
        capture.open("/home/uid22654/videos_code/test.mp4");
    }
    if (!capture.isOpened()) {
        std::cout << "cannot read video!\n";
        return -1;
    }

    cv::Mat image;
    std::vector<cv::KeyPoint> kps;
    DetectorExtractorMatcher dem;

    while (true) {
        if (!capture.read(image)) {
            break;
        }

        dem.extractFeatures(image, kps);

        for (auto& kp : kps)
        {
            cv::circle(image, kp.pt, 2, cv::Scalar(0, 0, 255));
        }

		cv::imshow("frame", image);
		if (cv::waitKey(delay) >= 0) {
			break;
		}

    }
    capture.release();
    return 0;
}

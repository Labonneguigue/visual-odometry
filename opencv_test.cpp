#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <bitset>

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

        static cv::Ptr<cv::ORB> orb = cv::ORB::create();/*cDefaultKeypointCount
                                                    , cPyramidScaleFactor
                                                    , cPyramidLevelCount
                                                    , cImageBorderThreshold
                                                    , cPyramidFirstLevel
                                                    , cBriefWta_k
                                                    , cKeypointScoreType
                                                    , cBriefPatchSize
                                                    , cKeypointFastThreshold);*/

        orb->detectAndCompute(image, cv::Mat(), keypoints, descriptors[1]);
        kps = keypoints;
    }

    void matchFeatures()
    {
        if (descriptors[0].size() == 0) std::copy(descriptors[1].begin(), descriptors[1].end(), descriptors[0].begin());

    }

private:

    std::vector<cv::KeyPoint> keypoints;
    std::vector<unsigned char> descriptors[2];
    std::vector<cv::DMatch>   matches;
    cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_HAMMING);
};

int main()
{
	cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);
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

        cv::Mat gray;

        //cv::cvtColor(image, gray, CV_BGR2GRAY);

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

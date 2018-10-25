#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <bitset>

class DetectorExtractorMatcher
{
public:

    void extractFeatures(cv::Mat& image, std::vector<cv::KeyPoint>& kps_0, std::vector<cv::KeyPoint>& kps_1, cv::Mat& desc)
    {
        static const int cDefaultKeypointCount = 3000; ///< OpenCV default feature count
        static const int cImageBorderThreshold = 31; ///< Features are not detected on the border of the image, this specifies the border size
        static const int cPyramidFirstLevel = 0; ///< Needs to be set to 0
        static const int cBriefWta_k = 2; ///< Number of points used to compute one attribute of the BRIEF descriptor. Setting it to 2 means that we are comparing the brightness of 2 pixels
        static const int cKeypointScoreType = cv::ORB::FAST_SCORE;  ///< Either Harris score or FAST score. Harris produces better (more stable) keypoints, but it is slower
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

        descriptors[0] = descriptors[1];
        descriptors[1] = cv::Mat();
        std::swap(keypoints[0],keypoints[1]);
        keypoints[1].clear();
        // std::vector<cv::Point2d> pts;
        // cv::goodFeaturesToTrack(image, pts, cDefaultKeypointCount, 0.01, 7);
        // for (auto & pt : pts) keypoints[1].push_back(cv::KeyPoint(pt, cBriefPatchSize));
        // orb->compute(image, keypoints[1], descriptors[1]);
        orb->detectAndCompute(image, cv::Mat(), keypoints[1], descriptors[1]);
        kps_0 = keypoints[0];
        kps_1 = keypoints[1];
        desc = descriptors[1];
    }

    void matchFeatures(std::vector<std::vector<cv::DMatch>>& mts)
    {
        matcher.knnMatch(descriptors[0], descriptors[1], mts, 2);
        //flann_matcher.knnMatch(descriptors[0], descriptors[1], mts, 2);
    }

private:

    std::vector<cv::KeyPoint> keypoints[2];
    cv::Mat descriptors[2];
    cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_HAMMING2);
    //cv::FlannBasedMatcher flann_matcher = cv::FlannBasedMatcher();
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
    std::vector<cv::KeyPoint> kps[2];
    DetectorExtractorMatcher dem;
    std::vector< cv::DMatch > good_matches;
    double min_dist = 30;

    while (true) {
        if (!capture.read(image)) {
            break;
        }

        cv::Mat gray, descriptors;
        cv::cvtColor(image, gray, CV_BGR2GRAY);

        dem.extractFeatures(image, kps[0], kps[1], descriptors);

        std::vector<std::vector<cv::DMatch>> mts;
        dem.matchFeatures(mts);

        good_matches.clear();
        std::vector<cv::Point2d> matched_prev, matched_curr;
        for (auto & mt : mts)
        {
            for (auto & m : mt)
                if( m.distance < min_dist )
                {
                    good_matches.push_back( m );
                    matched_prev.push_back(kps[0][m.queryIdx].pt);
                    matched_curr.push_back(kps[1][m.trainIdx].pt);
                    break;
                }
        }

        assert(matched_curr.size() == matched_prev.size());
        cv::Mat F;
        std::set<int> outliers;
        static const double SAMPSON_ERROR_THRESHOLD = 50.0;
        if (matched_curr.size() > 8)
        {
            F = cv::findFundamentalMat(matched_prev, matched_curr, CV_FM_7POINT, 0.02);
            for (int i = 0 ; i < matched_curr.size() ; ++i)
            {
                double error = cv::sampsonDistance(cv::Mat(matched_curr[i]), cv::Mat(matched_prev[i]), F);
                if (std::isnan(error) || error > SAMPSON_ERROR_THRESHOLD) outliers.emplace(i);
            }
        }



        std::cout << "Keypoints: " << kps[1].size() << " Matches: " << mts.size() << " good matches: " << good_matches.size() << " outliers: " << outliers.size() << "\n";

        for (int i = 0 ; i < matched_curr.size() ; ++i)
        {
            if (outliers.find(i) == outliers.end()) cv::line(image, matched_curr[i], matched_prev[i], cv::Scalar(0,0,255), 1, 1, 0);
        }

        cv::imshow("frame", image);
        if (cv::waitKey(delay) >= 0) {
            break;
        }

    }
    capture.release();
    return 0;
}


        // for (auto& mt : good_matches)
        // {
        //     cv::line(image, kps[0][mt.queryIdx].pt, kps[1][mt.trainIdx].pt, cv::Scalar(0,0,255), 1, 1, 0);
        // }

        // for (auto& kp : kps)
        // {
        //     cv::circle(image, kp.pt, 2, cv::Scalar(0, 0, 255));
        // }
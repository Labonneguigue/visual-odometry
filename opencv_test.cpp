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

template<typename T>
T kph2ms(T kph)
{
    return (static_cast<T>(1000.0)*kph)/static_cast<T>(3600.0);
}

int main()
{
	cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);
    const bool SET_CAMERA = false;
    // Open vid reader
    const int delay = 20;
    const int fps = 20; // video shot at 20 fps
    constexpr float delta_t = 1.0 / static_cast<float>(fps);

    cv::VideoCapture capture;
    if (SET_CAMERA) {
        capture.open(0); // open the default camera
    } else {
        //capture.open("/home/uid22654/videos_code/test.mp4");
        capture.open("/home/uid22654/videos_code/train.mp4");
    }
    if (!capture.isOpened()) {
        std::cout << "cannot read video!\n";
        return -1;
    }

    std::ifstream ground_truth("/home/uid22654/videos_code/train_gt.txt");

    cv::Mat image;
    std::vector<cv::KeyPoint> kps[2];
    DetectorExtractorMatcher dem;
    double min_dist = 30;

    // OpenPlus 3: 29mm | LePro 3: 4.04->26.81
    double focal = 8.0;
    cv::Point2d pp (image.cols/2, image.rows/2);

    long double cum_coeff = 0.0;
    int frames = 0;

    while (true) {
        if (!capture.read(image)) {
            break;
        }

        ++frames;
        std::string line;
        std::getline(ground_truth, line);
        double speed = kph2ms(std::atof(line.c_str()));
        std::cout << "Speed: " << speed << "\n";

        // Convert image to grayscale because keypoint detection works in 1 dimension
        cv::Mat gray, descriptors;
        cv::cvtColor(image, gray, CV_BGR2GRAY);

        // Detect keypoint locations and extract their descriptors
        dem.extractFeatures(image, kps[0], kps[1], descriptors);
        std::vector<std::vector<cv::DMatch>> matches;
        dem.matchFeatures(matches);

        std::vector< cv::DMatch > good_matches;
        std::vector<cv::Point2d> matched_prev, matched_curr, outlier_prev, outlier_curr;
        for (auto & match : matches)
        {
            const float ratio = 0.70; // As in Lowe's paper; can be tuned
            if (match[0].distance < ratio * match[1].distance)
            {
                good_matches.push_back( match[0] );
                matched_prev.push_back(kps[0][match[0].queryIdx].pt);
                matched_curr.push_back(kps[1][match[0].trainIdx].pt);
            }
        }

        assert(matched_curr.size() == matched_prev.size());
        cv::Mat F, R, t;
        int outliers_count = 0;
        static const double SAMPSON_ERROR_THRESHOLD = 50.0;
        // If matches, then computation of the Fundamental matrix using ransac and
        // outlier rejection
        if (matched_curr.size() > 8)
        {
            std::vector<uchar> inliers(matched_prev.size(),0);
            F = cv::findFundamentalMat(matched_prev, matched_curr, inliers, CV_FM_RANSAC); // 0.02
            std::vector<uchar>::const_iterator itIn= inliers.begin();
            for (int i = 0 ; i < matched_curr.size() ; ++i)
            {
                double error = cv::sampsonDistance(cv::Mat(matched_curr[i]), cv::Mat(matched_prev[i]), F);
                if (std::isnan(error) || error > SAMPSON_ERROR_THRESHOLD || !(*itIn)){
                    ++outliers_count;
                    outlier_curr.push_back(matched_curr[i]);
                    outlier_prev.push_back(matched_prev[i]);
                    matched_curr.erase(matched_curr.begin() + i);
                    matched_prev.erase(matched_prev.begin() + i);
                }
            }

            assert(matched_curr.size() == matched_prev.size());

            // Recover pose: rotation (R) and translation (t) from one frame to the next.
            cv::recoverPose(F, matched_prev, matched_curr, R, t, focal, pp );

            // Print translation (t) matrix
            std::cout << "t: \n";
            for (int i = 0 ; i<t.rows ; ++i)
            {
                double coeff = std::abs( (speed/fps) / t.at<int>(i,0) );
                std::cout << std::setprecision(15) << coeff << "\n";
                if (i == 0) cum_coeff += coeff;
            }
            std::cout << "\n";
        }

        std::cout << "Keypoints: " << kps[1].size() << " Matches: " << matches.size() << " good matches: " << good_matches.size() << " outliers: " << outliers_count << "\n";

        // Plot lines between matched keypoint (inliers)
        for (int i = 0 ; i < matched_curr.size() ; ++i)
        {
            cv::line(image, matched_curr[i], matched_prev[i], cv::Scalar(255,0,0), 1, 1, 0);
        }

        // Plot outliers
        for (int i = 0 ; i < outlier_curr.size() ; ++i)
        {
            cv::line(image, outlier_curr[i], outlier_prev[i], cv::Scalar(0,0,255), 1, 1, 0);
        }

        // Display image with overlays
        cv::imshow("frame", image);
        if (cv::waitKey(delay) >= 0) {
            break;
        }

    }
    capture.release();

    cum_coeff /= frames;
    std::cout << "Cumul coeff: " << std::setprecision(10) << cum_coeff;

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
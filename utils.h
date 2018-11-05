#ifndef UTILS_H
#define UTILS_H

#include "opencv2/opencv.hpp"

namespace utl{

template<typename T>
T kph2ms(T kph)
{
    return (static_cast<T>(1000.0)*kph)/static_cast<T>(3600.0);
}

}

template<typename T>
void print_matrix(cv::Mat& mat)
{
    for (int i = 0 ; i<mat.rows ; ++i)
    {
        for (int j = 0 ; j<mat.cols ; ++j)
        {
            std::cout << std::setprecision(15) << mat.at<T>(i, j) << "   ";
        }
        std::cout << "\n";
    }
}

#endif
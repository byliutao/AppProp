//
// Created by liutao on 23-8-15.
//

#ifndef APPPROP_APPPROP_H
#define APPPROP_APPPROP_H

#define SHOW_RESULT

#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <random>
#include <algorithm>

using namespace std;
using namespace cv;
using namespace Eigen;

class AppProp {
private:
    const int _m = 100;
    const double _alpha_a = 500;
    const double _alpha_s = 100;
    Mat _source_img;
    Mat _user_select_mask;
    Mat _initial_edit_img;
    Mat _final_edit_img;
    int _brightness_increase;
    int _height;
    int _width;
    int _n;
    void initialEditImage();
    void imagePropagating();
    void renderImageWithMask();
    void convertVectorToMask(const Eigen::VectorXd &vector, Mat &mask) const;
    static Vec6d computeFeatureVector(const Mat &image, int row, int col);
public:
    AppProp(Mat &source_img, Mat &user_select_mask, int brightness_increase);

    void getEditResult(Mat &result){
        result = _final_edit_img;
    }

};


#endif //APPPROP_APPPROP_H

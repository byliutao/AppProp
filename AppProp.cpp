//
// Created by liutao on 23-8-15.
//

#include "AppProp.h"

Vec6d AppProp::computeFeatureVector(const Mat &image, int row, int col) {
    Vec6d featureVector;

    int halfNeighborhood = 3 / 2;
    int startRow = max(row - halfNeighborhood, 0);
    int endRow = min(row + halfNeighborhood + 1, image.rows);
    int startCol = max(col - halfNeighborhood, 0);
    int endCol = min(col + halfNeighborhood + 1, image.cols);

    // Extract the valid neighborhood around the pixel
    Mat neighborhood = image(Rect(startCol, startRow, endCol - startCol, endRow - startRow));

    // Calculate average and standard deviation of color in the neighborhood
    Scalar meanColor = mean(neighborhood);
    Scalar stddevColor;
    meanStdDev(neighborhood, meanColor, stddevColor);

    featureVector[0] = meanColor[0];  // L channel
    featureVector[1] = meanColor[1];  // a channel
    featureVector[2] = meanColor[2];  // b channel
    featureVector[3] = stddevColor[0];
    featureVector[4] = stddevColor[1];
    featureVector[5] = stddevColor[2];

    return featureVector;
}

void AppProp::imagePropagating() {
    Eigen::VectorXd g(_n), w(_n), one_n(_n);
    Eigen::MatrixXd U(_n,_m);
    double lambda;
    int select_pixel_num = 0;


    g.setZero();
    w.setZero();
    one_n.setOnes();
    U.setZero();

    //calculate Matrix W and vector g
    for(int row = 0; row < _source_img.rows; row++){
        for(int col = 0; col < _source_img.cols; col++){
            uchar value = _user_select_mask.at<uchar>(row,col);
            int index = row*(_source_img.cols) + col;
            if(value > 0){
                g(index) = _brightness_increase;
                w(index) = 1;
                select_pixel_num++;
            }
            else{
                g(index) = 0;
                w(index) = 0;
            }
        }
    }

    //test g
#ifdef SHOW_RESULT
    Mat g_mask;
    convertVectorToMask(g,g_mask);
    namedWindow("g_mask",WINDOW_NORMAL);
    imshow("g_mask",g_mask);
    waitKey(0);
#endif

    lambda = select_pixel_num * 1.0 / (_height*_width);

    std::vector<int> availableNumbers(_n);
    for (int i = 0; i < _n; ++i) {
        availableNumbers[i] = i;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(availableNumbers.begin(), availableNumbers.end(), gen);

    std::vector<int> selectedNumbers(availableNumbers.begin(), availableNumbers.begin() + _m);

    //calculate Matrix A and U
    for(int i = 0; i < _n; i++){
        for(int j = 0; j < _m; j++){
            int index_j = selectedNumbers[j];
//            int index_j = j;
            int pixel_i_row = i / _width;
            int pixel_i_col = i - pixel_i_row*_width;
            int pixel_j_row = index_j / _width;
            int pixel_j_col = index_j - pixel_j_row*_width;
            Vec3b pixel_i = _source_img.at<Vec3b>(pixel_i_row,pixel_i_col);
            Vec3b pixel_j = _source_img.at<Vec3b>(pixel_j_row,pixel_j_col);
            Vec6d feature_i = computeFeatureVector(_source_img, pixel_i_row, pixel_i_col);
            Vec6d feature_j = computeFeatureVector(_source_img, pixel_j_row, pixel_j_col);
            Vec2i location_i = Vec2i(pixel_i_row,pixel_i_col);
            Vec2i location_j = Vec2i(pixel_j_row, pixel_j_col);
            double z_ij = exp(-norm(feature_i-feature_j,NORM_L2SQR)/_alpha_a)*exp(-norm(location_i-location_j,NORM_L2SQR)/_alpha_s);
            U(i,j) = z_ij;
        }
    }

    MatrixXd U_tran = U.transpose();
    MatrixXd A = U.block(0, 0, _m, _m);
    if(A.determinant() == 0){
        double regularizationParameter = 1e-2;  // Adjust this value as needed

        // Regularize matrix A by adding a diagonal regularization term
        A = A + regularizationParameter * Eigen::MatrixXd::Identity(_m, _m);
    }
    MatrixXd A_inv = A.inverse();


    VectorXd a = U * (A_inv * (U_tran * (w.asDiagonal() * one_n))); // U A^-1 U^T M W 1_n
    VectorXd b = U * (A_inv * (U_tran * one_n));                    // U A^-1 U^T M 1_n
;
    VectorXd d_inv = a / lambda / 2 + b;
    for (int i = 0; i < _n; i++){
        d_inv(i) = 1.0 / d_inv(i);
    }

    VectorXd c = U * (A_inv * (U_tran * (w.asDiagonal() * g))); // U A^-1 U^T M W g
    MatrixXd U_TD_1 = U_tran * d_inv.asDiagonal();                                  // U^T D^-1
    MatrixXd D_1U = d_inv.asDiagonal() * U;                                      // D^-1 U
    MatrixXd mid = (U_TD_1 * U - A).inverse();                                 // (-A + U^T D^-1 U)^-1

    VectorXd e = d_inv.asDiagonal() * c - D_1U * (mid * (U_TD_1 * c));
    e /= lambda * 2;

    convertVectorToMask(e,_final_edit_mask);
#ifdef SHOW_RESULT
    Mat e_mask;
    convertVectorToMask(e,e_mask);
    namedWindow("e_mask",WINDOW_NORMAL);
    imshow("e_mask",e_mask);
    waitKey(0);
#endif
}

void AppProp::renderImageWithMask() {
    for(int row = 0; row < _source_img.rows; row++){
        for(int col = 0; col < _source_img.cols; col++){
            cout<<(int)_source_img.at<Vec3b>(row,col)[0]<<endl;
            _final_edit_img.at<Vec3b>(row,col) = _source_img.at<Vec3b>(row,col);
            _final_edit_img.at<Vec3b>(row,col)[0] += _final_edit_mask.at<uchar>(row,col);
        }
    }
#ifdef SHOW_RESULT
    Mat show;
    cvtColor(_final_edit_img,show,COLOR_Lab2BGR);
    namedWindow("final_edit_img",WINDOW_NORMAL);
    imshow("final_edit_img",show);
    waitKey(0);
#endif
}

void AppProp::initialEditImage() {
    cout<<_source_img.type()<<endl;
    for(int row = 0; row < _source_img.rows; row++){
        for(int col = 0; col < _source_img.cols; col++){
            CV_Assert(_source_img.at<Vec3b>(row,col)[0] <= 100);
            if(_user_select_mask.at<uchar>(row,col) > 0){
                _initial_edit_img.at<Vec3b>(row,col)[0] = (_source_img.at<Vec3b>(row,col)[0] + _brightness_increase) > 100 ? 100 :
                                                          (_source_img.at<Vec3b>(row,col)[0] + _brightness_increase);
            }
            else{
                _initial_edit_img.at<Vec3b>(row,col)[0] = _source_img.at<Vec3b>(row,col)[0];
            }
        }
    }

#ifdef SHOW_RESULT
    Mat show;
    cvtColor(_initial_edit_img,show,COLOR_Lab2BGR);
    namedWindow("initial_edit_img",WINDOW_NORMAL);
    imshow("initial_edit_img",show);
    waitKey(0);
#endif
}

AppProp::AppProp(Mat &source_img, Mat &user_select_mask, int brightness_increase): _source_img(source_img), _user_select_mask(user_select_mask)
    ,_brightness_increase(brightness_increase) {
    _initial_edit_img = Mat(_source_img.size(),_source_img.type());
    _final_edit_img = Mat(_source_img.size(), _source_img.type());
    cvtColor(_source_img,_source_img,COLOR_BGR2Lab);
    cvtColor(_initial_edit_img,_initial_edit_img,COLOR_BGR2Lab);
    cvtColor(_final_edit_img,_final_edit_img,COLOR_BGR2Lab);
    _height = _source_img.rows;
    _width = _source_img.cols;
    _n = _height * _width;
//    initialEditImage();
    imagePropagating();
    renderImageWithMask();
}

void AppProp::convertVectorToMask(const Eigen::VectorXd &vector, Mat &mask) const {
    mask = Mat(_source_img.size(), CV_8UC1);
    for(int row = 0; row < _height; row++){
        for(int col = 0; col < _width; col++){
            mask.at<uchar>(row,col) = (uchar)vector(row*_width + col) * 2.55;
        }
    }
}

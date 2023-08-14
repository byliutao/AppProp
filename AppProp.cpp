//
// Created by liutao on 23-8-15.
//

#include "AppProp.h"

void AppProp::imagePropagating() {
    Eigen::VectorXd e(_n), g(_n);
    Eigen::SparseMatrix<double> W(_n, _n), D(_n,_n), D_inv(_n,_n);
    Eigen::MatrixXd A(_m,_m), A_inv(_m, _m);
    Eigen::MatrixXd U(_n,_m), U_tran(_n, _m);
    double lambda;
    int select_pixel_num = 0;


    A.setZero();
    D.setZero();
    D_inv.setZero();
    W.setZero();
    e.setZero();

    //calculate Matrix W and vector g
    for(int row = 0; row < _source_img.rows; row++){
        for(int col = 0; col < _source_img.cols; col++){
            uchar value = _user_select_mask.at<uchar>(row,col);
            int index = row*(_source_img.cols) + col;
            if(value > 0){
                g(index) = _brightness_increase;
                W.insert(index,index) = 1;
                select_pixel_num++;
            }
            else{
                g(index) = 0;
                W.insert(index,index) = 0;
            }
        }
    }
    W.finalize();


    //test g
#ifdef SHOW_RESULT
    Mat g_mask;
    convertVectorToMask(g,g_mask);
    namedWindow("g_mask",WINDOW_NORMAL);
    imshow("g_mask",g_mask);
    waitKey(0);
#endif

    lambda = select_pixel_num * 1.0 / (_height*_width);

    //calculate Matrix A and U
    for(int i = 0; i < _n; i++){
        for(int j = 0; j < _m; j++){
            int pixel_i_row = i % _width;
            int pixel_i_col = i - pixel_i_row*_width;
            int pixel_j_row = j % _width;
            int pixel_j_col = j - pixel_j_row*_width;
            Vec3b pixel_i = _source_img.at<Vec3b>(pixel_i_row,pixel_i_col);
            Vec3b pixel_j = _source_img.at<Vec3b>(pixel_j_row,pixel_j_col);
            Vec2i location_i = Vec2i(pixel_i_row,pixel_i_col);
            Vec2i location_j = Vec2i(pixel_j_row, pixel_j_col);

            double z_ij = exp(-norm(pixel_i-pixel_j,NORM_L2)/_alpha_a)*exp(-norm(location_i-location_j,NORM_L2)/_alpha_s);
            if(i < _m){
                A(i,j) = z_ij;
                U(i, j) = z_ij;
            }
            else{
                U(i,j) = z_ij;
            }
        }
    }
    A_inv = A.inverse();
    U_tran = U.transpose();

    //calculate Matrix D
    Eigen::VectorXd oneVector(_n), valueVector(_n);
    oneVector.setOnes();
    valueVector = (U * A);

    valueVector = (((1.0 / (2 * lambda)) * U * A_inv * U_tran * W) + U * A_inv * U_tran) * oneVector;
    for(int i = 0; i < _n; i++){
        D.insert(i,i) = valueVector(i);
    }
    D.finalize();
    D_inv = D;

    //calculate e
    e = (1.0 / (2 * lambda)) * (D_inv - D_inv * U * (-A + U_tran * D_inv * U) * U_tran * D_inv) * (U * A_inv * U_tran) * W * g;

#ifdef SHOW_RESULT
    Mat e_mask;
    convertVectorToMask(e,e_mask);
    namedWindow("e_mask",WINDOW_NORMAL);
    imshow("e_mask",e_mask);
    waitKey(0);
#endif
}

void AppProp::renderImageWithMask() {

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
            mask.at<uchar>(row,col) = (uchar)vector(row*_width + col);
        }
    }
}

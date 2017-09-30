#include<iostream>
#include<vector>
#include<algorithm>
#include<opencv2/opencv.hpp>
#include<opencv2/nonfree/nonfree.hpp>
#include<opencv2/features2d/features2d.hpp>

using namespace std;

int main(int argc,char** argv)
{
    cv::Mat img1=cv::imread(argv[1]);
    cv::Mat img2=cv::imread(argv[2]);

    imshow("img1",img1);
    imshow("img2",img2);

    cv::Mat gray_img1,gray_img2;
    cvtColor(img1,gray_img1,CV_RGB2GRAY);
    cvtColor(img2,gray_img2,CV_RGB2GRAY);
    int minHessian = 800;
    cv::SurfFeatureDetector surfDetector(minHessian);
    vector<cv::KeyPoint> kp1,kp2;
    surfDetector.detect(gray_img1,kp1);
    surfDetector.detect(gray_img2,kp2);

    cv::SurfDescriptorExtractor surfDescriptor;
    cv::Mat imgDes1,imgDes2;
    surfDescriptor.compute(gray_img1,kp1,imgDes1);
    surfDescriptor.compute(gray_img2,kp2,imgDes2);

    //Match Points
    cv::FlannBasedMatcher matcher;
    vector<cv::DMatch> matchPoints;
    matcher.match(imgDes1,imgDes2,matchPoints,cv::Mat());
    sort(matchPoints.begin(),matchPoints.end());

    vector<cv::Point2f> imgPoints1,imgPoints2;
    for(int i=0;i<8;i++)
    {
        imgPoints1.push_back(kp1[matchPoints[i].queryIdx].pt);
        imgPoints2.push_back(kp2[matchPoints[i].trainIdx].pt);
    }

    cv::Mat homo = cv::findHomography(imgPoints1,imgPoints2,CV_RANSAC);

    cv::Mat imageTransform;
    cv::warpPerspective(img1,imageTransform,homo,cv::Size(img2.cols,img2.rows));
    cv::imshow("Transform image",imageTransform);


    cv::waitKey(0);
    return 0;

}
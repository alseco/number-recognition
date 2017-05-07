/*
This program teaches the computer how to 
classify numbers by extracting features of the number such as area and perimeter

by: Aizaya L. Seco and Alyssa De Guzman
date: 10/07/2016
*/
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <stdio.h>
#define SIZE 10

using namespace cv;
using namespace std;

Mat src, dst;
int array[SIZE][2]; //features array [0] area [1] perimeter

void getMode(int features[][2], int size, int greatest, int flag, int value){
	//flag: 0 if area, 1 if perimeter
	int i;
	int mode[greatest];

	for(i=0;i<greatest;i++){
		//initialize
		mode[i] = 0;
	}
	int val = 0;
	for(i=1;i<size;i++){
		val = features[i][flag];
		mode[val]+=1;
	}
	int index=0;
	int maxCount=0;
	i=0;
	while(i<greatest){
		if(mode[i]>=maxCount){
			maxCount=mode[i];
			index=i;
			i++;
		}
		else
			i++;
	}
	
	array[value][flag]=index;
}

void erosion(Mat input){
  int i,j;
  Mat output(input.size(), input.type());

  for(i=0; i < input.rows; i++){
      for(j=0; j< input.cols; j++){

        if(input.at<uchar>(i,j) == 255){
          if(input.at<uchar>(i-1,j)==0 || input.at<uchar>(i,j-1)==0 || input.at<uchar>(i+1,j)==0 || input.at<uchar>(i,j+1)==0){
            output.at<uchar>(i-1,j) = 0;
            output.at<uchar>(i,j-1) = 0;
            output.at<uchar>(i+1,j) = 0;
            output.at<uchar>(i,j+1) = 0;
          	output.at<uchar>(i,j) = 0;
		  }
          else
			  output.at<uchar>(i,j) = 255;
        }
        else{
          output.at<uchar>(i,j) = 0;
        }

      }
  }
  output.copyTo( dst );
}

void subtraction(Mat input, Mat eroded){
  int i,j;
  Mat output(input.size(), input.type());

  for(i=0; i < input.rows; i++){
      for(j=0; j< input.cols; j++){

        if(input.at<uchar>(i,j) == eroded.at<uchar>(i,j)){
          output.at<uchar>(i,j) = 0;
        }else{
			output.at<uchar>(i,j) = 255;
		}

      }
  }
  output.copyTo( dst );
}

int compareFeatures(int features[][2], int value){
	//using euclidean distance
	int i,index;
	double minimum = 10000;
	for(i=0;i<SIZE;i++){
		double base1 = array[i][0] - features[value][0];
		double base2 = array[i][1] - features[value][1];
		double base= 0;
		double distance= 0;
		
		base1= pow(base1,2);
		base2= pow(base2,2);
		base= base1+base2;
		
		distance = sqrt(base);
		if(distance<minimum){
			minimum=distance;
			index=i;
		}
	}
	return index;
}

void extractFeatures(Mat input, int flag,int value){
	//flag is 0 if input is src
	//flag is 1 if input is training data
	//value corresponds to the number in the training data; 10 if source
	//in this function it will first get the area, the the perimeter, then the comapactness
	
	int i,j,k;
	int nLabels;
	const int connectivity = 4;

	Mat labelImage(input.size(),CV_32S);
	Mat stats, centroids; 
	nLabels =  connectedComponentsWithStats(input, labelImage,stats,centroids, connectivity, CV_32S);
	int arr[nLabels];
	int per[nLabels];
	int features[nLabels][2];// [0]area [1]perimeter
	
	//initialization for perimeter
	dst=input.clone();
	erosion(dst);
	subtraction(input,dst);
	
	/*
	for(int i=0; i<nLabels; i++){
	printf("Component %d stats\n", i);
	printf("  CC_STAT_LEFT: %d\n", stats.at<int>(i,CC_STAT_LEFT));
	printf("  CC_STAT_TOP: %d\n", stats.at<int>(i,CC_STAT_TOP));
	printf("  CC_STAT_WIDTH: %d\n", stats.at<int>(i,CC_STAT_WIDTH));
	printf("  CC_STAT_HEIGHT: %d\n", stats.at<int>(i,CC_STAT_HEIGHT));
	printf("  CC_STAT_AREA: %d\n", stats.at<int>(i,CC_STAT_AREA));
	printf("  Centroid (x,y): %f, %f\n\n", centroids.at<double>(i,0), centroids.at<double>(i,1));
    }*/

	for(i=0;i<nLabels;i++){
		//initialize
		arr[i] = 0;
		per[i]= 0;
	}
	
	for(i = 0; i < input.rows; ++i){
	   	for(j = 0; j < input.cols; ++j){
	   		//gets area for every object in the image
	        int label = labelImage.at<int>(i,j);
	        arr[label]+=1;
	    }
	}

	for (i=0;i<nLabels;i++){		
		//gets perimeter for every object in the image
		for(j=(stats.at<int>(i,CC_STAT_TOP));j<(stats.at<int>(i,CC_STAT_TOP)+stats.at<int>(i,CC_STAT_HEIGHT));j++){
			for(k=(stats.at<int>(i,CC_STAT_LEFT));k<(stats.at<int>(i,CC_STAT_LEFT)+stats.at<int>(i,CC_STAT_WIDTH));k++){
				if(dst.at<uchar>(j,k)==255){
					per[i]+=1;
				}
			}
		}
		
	}

	if(flag==0){
		//if src
		for(i=0;i<nLabels;i++){
			int area = arr[i];
			features[i][0]=area;
			int perimeter= per[i];
			features[i][1]= perimeter;			
		}
		
		for(i=1;i<nLabels;i++){
			int result= compareFeatures(features,i);
			if(result==0){
				rectangle(src, Point(stats.at<int>(i,CC_STAT_LEFT),stats.at<int>(i,CC_STAT_TOP)), Point((stats.at<int>(i,CC_STAT_LEFT)+stats.at<int>(i,CC_STAT_WIDTH)),(stats.at<int>(i,CC_STAT_TOP)+stats.at<int>(i,CC_STAT_HEIGHT))), Scalar( 255, 115, 0 ), 2, 8, 0);
			}else if(result==1){
				rectangle(src, Point(stats.at<int>(i,CC_STAT_LEFT),stats.at<int>(i,CC_STAT_TOP)), Point((stats.at<int>(i,CC_STAT_LEFT)+stats.at<int>(i,CC_STAT_WIDTH)),(stats.at<int>(i,CC_STAT_TOP)+stats.at<int>(i,CC_STAT_HEIGHT))), Scalar( 180, 105, 255 ), 2, 8, 0);
			}else if(result==2){
				rectangle(src, Point(stats.at<int>(i,CC_STAT_LEFT),stats.at<int>(i,CC_STAT_TOP)), Point((stats.at<int>(i,CC_STAT_LEFT)+stats.at<int>(i,CC_STAT_WIDTH)),(stats.at<int>(i,CC_STAT_TOP)+stats.at<int>(i,CC_STAT_HEIGHT))), Scalar( 255, 20, 147 ), 2, 8, 0);
			}else if(result==3){
				rectangle(src, Point(stats.at<int>(i,CC_STAT_LEFT),stats.at<int>(i,CC_STAT_TOP)), Point((stats.at<int>(i,CC_STAT_LEFT)+stats.at<int>(i,CC_STAT_WIDTH)),(stats.at<int>(i,CC_STAT_TOP)+stats.at<int>(i,CC_STAT_HEIGHT))), Scalar( 0, 255, 255 ), 2, 8, 0);
			}else if(result==4){
				rectangle(src, Point(stats.at<int>(i,CC_STAT_LEFT),stats.at<int>(i,CC_STAT_TOP)), Point((stats.at<int>(i,CC_STAT_LEFT)+stats.at<int>(i,CC_STAT_WIDTH)),(stats.at<int>(i,CC_STAT_TOP)+stats.at<int>(i,CC_STAT_HEIGHT))), Scalar( 255, 0, 255 ), 2, 8, 0);
			}else if(result==5){
				rectangle(src, Point(stats.at<int>(i,CC_STAT_LEFT),stats.at<int>(i,CC_STAT_TOP)), Point((stats.at<int>(i,CC_STAT_LEFT)+stats.at<int>(i,CC_STAT_WIDTH)),(stats.at<int>(i,CC_STAT_TOP)+stats.at<int>(i,CC_STAT_HEIGHT))), Scalar( 255, 255, 0 ), 2, 8, 0);
			}else if(result==6){
				rectangle(src, Point(stats.at<int>(i,CC_STAT_LEFT),stats.at<int>(i,CC_STAT_TOP)), Point((stats.at<int>(i,CC_STAT_LEFT)+stats.at<int>(i,CC_STAT_WIDTH)),(stats.at<int>(i,CC_STAT_TOP)+stats.at<int>(i,CC_STAT_HEIGHT))), Scalar( 0, 255, 0 ), 2, 8, 0);
			}else if(result==7){
				rectangle(src, Point(stats.at<int>(i,CC_STAT_LEFT),stats.at<int>(i,CC_STAT_TOP)), Point((stats.at<int>(i,CC_STAT_LEFT)+stats.at<int>(i,CC_STAT_WIDTH)),(stats.at<int>(i,CC_STAT_TOP)+stats.at<int>(i,CC_STAT_HEIGHT))), Scalar( 0, 0, 255 ), 2, 8, 0);
			}else if(result==8){
				rectangle(src, Point(stats.at<int>(i,CC_STAT_LEFT),stats.at<int>(i,CC_STAT_TOP)), Point((stats.at<int>(i,CC_STAT_LEFT)+stats.at<int>(i,CC_STAT_WIDTH)),(stats.at<int>(i,CC_STAT_TOP)+stats.at<int>(i,CC_STAT_HEIGHT))), Scalar( 255, 0, 0 ), 2, 8, 0);
			}else if(result==9){
				rectangle(src, Point(stats.at<int>(i,CC_STAT_LEFT),stats.at<int>(i,CC_STAT_TOP)), Point((stats.at<int>(i,CC_STAT_LEFT)+stats.at<int>(i,CC_STAT_WIDTH)),(stats.at<int>(i,CC_STAT_TOP)+stats.at<int>(i,CC_STAT_HEIGHT))), Scalar( 255, 165, 0 ), 2, 8, 0);
			}
		}
	}
	else if(flag==1){
		//if training data
		features[0][0]=0;
		int greatest=0;
		for(i=1;i<nLabels;i++){
			int area=arr[i];
			features[i][0]=area;
			if(area>greatest){
				greatest=area;
			}
		}
		getMode(features,nLabels,greatest, 0, value); //mode of area
		
		features[1][0]=0;
		greatest =0;
		for(i=1;i<nLabels;i++){
			int perimeter=per[i];
			features[i][1]=perimeter;
			if(perimeter>greatest){
				greatest=perimeter;
			}
		}
		getMode(features,nLabels,greatest, 1, value); //mode of perimeter
	}

}

int main(int, char *argv[]){
	Mat src_inv, zero, one, two, three, four, five, six, seven, eight, nine;
	int i;
	src = imread( argv[1]);

	if( src.empty() )
    { return -1; }
	cvtColor(src, src_inv, CV_BGR2GRAY);	
  	threshold(src_inv, src_inv, 200, 255,THRESH_BINARY_INV);
	
	zero = imread("0.jpg");
	cvtColor(zero, zero, CV_BGR2GRAY);
	threshold(zero, zero, 200, 255, THRESH_BINARY_INV);
	extractFeatures(zero, 1,0);
	
	one = imread("1.jpg");
	cvtColor(one, one, CV_BGR2GRAY);
	threshold(one, one, 200, 255, THRESH_BINARY_INV);
	extractFeatures(one,1,1);
	
	two = imread("2.jpg");
	cvtColor(two, two, CV_BGR2GRAY);
	threshold(two, two, 200, 255, THRESH_BINARY_INV);
	extractFeatures(two,1,2);
	
	three = imread("3.jpg");
	cvtColor(three, three, CV_BGR2GRAY);
	threshold(three, three, 200, 255, THRESH_BINARY_INV);
	extractFeatures(three,1,3);
	
	four = imread("4.jpg");
	cvtColor(four, four, CV_BGR2GRAY);
	threshold(four, four, 200, 255, THRESH_BINARY_INV);
	extractFeatures(four,1,4);
	
	five = imread("5.jpg");
	cvtColor(five, five, CV_BGR2GRAY);
	threshold(five, five, 200, 255, THRESH_BINARY_INV);
	extractFeatures(five,1, 5);
	
	six = imread("6.jpg");
	cvtColor(six, six, CV_BGR2GRAY);
	threshold(six, six, 200, 255, THRESH_BINARY_INV);
	extractFeatures(six,1, 6);

	seven = imread("7.jpg");
	cvtColor(seven, seven, CV_BGR2GRAY);
	threshold(seven, seven, 200, 255, THRESH_BINARY_INV);
	extractFeatures(seven,1, 7);

	eight = imread("8.jpg");
	cvtColor(eight, eight, CV_BGR2GRAY);
	threshold(eight, eight, 200, 255, THRESH_BINARY_INV);
	extractFeatures(eight,1, 8);

	nine = imread("9.jpg");
	cvtColor(nine, nine, CV_BGR2GRAY);
	threshold(nine, nine, 200, 255, THRESH_BINARY_INV);
	extractFeatures(nine,1, 9);
	
	printf("Area:\n");
	for(i=0;i<10;i++){
		printf("%d  ",array[i][0]);
	}
	printf("\nPerimeter:\n");
	for(i=0;i<10;i++){
		printf("%d  ",array[i][1]);
	}

	extractFeatures(src_inv,0,10);
	imshow("src",src);
	imshow("src_inv",src_inv);
	imwrite("output.jpg",src);
	
	/// Wait until user exit program by pressing a key
  	waitKey(0);
    return 0;
}

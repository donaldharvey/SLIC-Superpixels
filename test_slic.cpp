/*
 * test_slic.cpp.
 *
 * Written by: Pascal Mettes.
 *
 * This file creates an over-segmentation of a provided image based on the SLIC
 * superpixel algorithm, as implemented in slic.h and slic.cpp.
 */
 
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <float.h>
#include <fstream>
#include <iostream>
using namespace std;

#include "slic.h"

string get_base_filename(const char* filename)
{
    string fName(filename);
    size_t pos = fName.rfind(".");
    if(pos == string::npos)  // No extension.
        return fName;

    if(pos == 0)    //. is at the front. Not an extension.
        return fName;

    return fName.substr(0, pos);
}

int main(int argc, char *argv[]) {
    /* Load the image and convert to Lab colour space. */
    string root_name = get_base_filename(argv[1]);

    IplImage *image = cvLoadImage(argv[1], 1);
    IplImage *lab_image = cvCloneImage(image);
    cvCvtColor(image, lab_image, CV_BGR2Lab);
    
    /* Yield the number of superpixels and weight-factors from the user. */
    int w = image->width, h = image->height;
    int nr_superpixels = atoi(argv[2]);
    int nc = atoi(argv[3]);

    double step = sqrt((w * h) / (double) nr_superpixels);
    
    /* Perform the SLIC superpixel algorithm. */
    Slic slic;
    slic.generate_superpixels(lab_image, step, nc);
    slic.create_connectivity(lab_image);
    
    // /* Display the contours and show the result. */
    // slic.display_contours(image, CV_RGB(0,0,0));
    // cvShowImage("result", image);
    // cvWaitKey(0);
    // cvSaveImage(argv[4], image);

    vector<CvPoint> *contours = slic.generate_contours(image);
    vec2di *clusters = slic.get_clusters();

    bool* contoursOut = NULL;
    int* regionsOut = NULL;

    contoursOut = new bool[w*h];
    regionsOut = new int[w*h];

    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            contoursOut[j*w + i] = 0;
            regionsOut[(j*w)+i] = (*clusters)[i][j] + 1;
        }
    }

    for (int i = 0; i < (int)contours->size(); i++) {
        contoursOut[((*contours)[i].y * w) + (*contours)[i].x] = 1;
    }

    ofstream contoursFile;
    contoursFile.open(root_name + "_contours.dat", ios::binary | ios::out);

    ofstream regionsFile;
    regionsFile.open(root_name + "_regions.dat", ios::binary | ios::out);

    for (int i=0; i < w*h; i++) {
        contoursFile << contoursOut[i];
        regionsFile.write((char*)&(regionsOut[i]), sizeof(int));
    }

    contoursFile.close();
    regionsFile.close();
}

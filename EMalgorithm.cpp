/*Program to use EM algorithm to cluster each image from MINST dataset into ten groups*/
/*Author: Siffi Singh */
/*Dated: 9/11/2017 */

/*Standard Headers */
#include<iostream>
#include<cmath>
#include<cstdio>
#include<cstring>
#include<vector>
#include<algorithm>
#include <fstream>
using namespace std;

/*Function to read files from MNIST dataset*/
int ReverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int) ch1 << 24) + ((int) ch2 << 16) + ((int) ch3 << 8) + ch4;
}
void ReadMNIST(int NumberOfImages, int DataOfAnImage, vector < vector < double > > & arr, char s[]) {
        arr.resize(NumberOfImages, vector < double > (DataOfAnImage));
        ifstream file(s, ios::binary);
        if (file.is_open()) {
            int magic_number = 0;
            int number_of_images = 0;
            int n_rows = 0;
            int n_cols = 0;
            file.read((char * ) & magic_number, sizeof(magic_number));
            magic_number = ReverseInt(magic_number);
            file.read((char * ) & number_of_images, sizeof(number_of_images));
            number_of_images = ReverseInt(number_of_images);
            if (DataOfAnImage != 1) {
                file.read((char * ) & n_rows, sizeof(n_rows));
                n_rows = ReverseInt(n_rows);
                file.read((char * ) & n_cols, sizeof(n_cols));
                n_cols = ReverseInt(n_cols);
            }
            for (int i = 0; i < number_of_images; ++i) {
                if (DataOfAnImage != 1) {
                    for (int r = 0; r < n_rows; ++r) {
                        for (int c = 0; c < n_cols; ++c) {
                            unsigned char temp = 0;
                            file.read((char * ) & temp, sizeof(temp));
                            arr[i][(n_rows * r) + c] = (double) temp;
                        }
                    }
                } else {
                    unsigned char temp = 0;
                    file.read((char * ) & temp, sizeof(temp));
                    arr[i][0] = (double) temp;
                }
            }
        }
    }
    /* Binarization of pixel values, convertign each pixel to either 0 or 1*/
int convert_pixel(double x) {
        if (x > 127)
            return 1;
        else
            return 0;
    }
    /*Driver Function*/
int main() {
    /* Variable declarations */
    vector < vector < double > > ar;
    char s1[] = "train-images.idx3-ubyte";
    char s2[] = "train-labels.idx1-ubyte";
    vector < vector < double > > training_images;
    vector < vector < double > > training_labels;
    cout << "--------------------------------------------------------------------------\n" << endl;
    /*pixels: 784 = 28*28*/
    ReadMNIST(60000, 784, training_images, s1); //Storing training_images of size [60000, 784]
    for (int i = 0; i < 60000; i++) {
        for (int j = 0; j < 784; j++) {
            training_images[i][j] = convert_pixel(training_images[i][j]);
        }
    }
    ReadMNIST(60000, 1, training_labels, s2); //Storing training_labels of size [60000, 1]
    for (int i = 0; i < 60000; i++) {}
    /*Calculating count for final check*/
    int count[10] = {
        0
    }, count_calculated[10] = {
        0
    };
    for (int i = 0; i < 60000; i++) {
        if (training_labels[i][0] == 0)
            count[0]++;
        else if (training_labels[i][0] == 1)
            count[1]++;
        else if (training_labels[i][0] == 2)
            count[2]++;
        else if (training_labels[i][0] == 3)
            count[3]++;
        else if (training_labels[i][0] == 4)
            count[4]++;
        else if (training_labels[i][0] == 5)
            count[5]++;
        else if (training_labels[i][0] == 6)
            count[6]++;
        else if (training_labels[i][0] == 7)
            count[7]++;
        else if (training_labels[i][0] == 8)
            count[8]++;
        else if (training_labels[i][0] == 9)
            count[9]++;
    }
    
    /* Expectation-Step of EM algorithm*/
    
    /*Initializing initial parameter values*/
    double lambda[10], P[784], P2[10][784] = {
        0.5
    };
    for (int i = 0; i < 10; i++) {
        lambda[i] = 0.1; //Step - 1 : EM algorithm, intializing parameters
    }
    for (int i = 0; i < 784; i++) {
        P[i] = 0.5; //Step - 1 : EM algorithm, intializing parameters
    }
    double X[784];
    /*taking each data point one by one*/
    for (int i = 0; i < 60000; i++) {
        for (int j = 0; j < 784; j++) {
            X[i] = training_images[i][j];
        }
        /*Finding out P(X|P0, P1, P784)= Product of (for each i=0 to 784, i.e. each pixel) Pi^Xi * (1-Pi)^(1-Xi) */
        double prod = 1;
        for (int j = 0; j < 784; j++) {
            prod = prod * (pow(P[j], X[j])) * (pow((1 - P[j]), (1 - X[j])));
        }
        double prod_class[10] = {
            0
        };
        double sum = 0;
        /*Finding out P(X|P's and lambda's)= Sum of (for each class) lambda(class k)* (Product of (for each i=0 to 784, i.e. each pixel) Pi^Xi * (1-Pi)^(1-Xi)) */
        for (int j = 0; j < 10; j++) {
            double computed = 1;
            for (int k = 0; k < 784; k++) {
                computed = computed * (pow(P2[j][k], X[k])) * (pow((1 - P2[j][k]), (1 - X[k])));
            }
            prod_class[j] += (lambda[j] * computed);
            sum = sum + prod_class[j];
        }
        /*At this point the log of the loss function for this data is summation of (for all n=1-N i.e. N(max = 60000) and current value of N) * log(P(X|P's and lambda's) */
        double responsibility[10], max = 0;
        int pos = 0;
        /* Computing responsibility as the division of lambda(k) * P(Xn|Pk) and the summation (of all classes k) lambda(k) * P(Xn|Pk) */
        for (int j = 0; j < 10; j++) //for each class we find the responsibility
        {
            responsibility[j] = prod_class[j] / sum;
            if (max > responsibility[j]) {
                max = responsibility[j];
                pos = j;
            }
        }
        
        /* Maximization-Step of EM algorithm*/
        
        count_calculated[pos]++;
        /* Updating the lambdas */
        for (int j = 0; j < 10; j++) {
            lambda[j] = count_calculated[j] / (i + 1);
        }
        double P_prev[10][784];
        /*Storing for comparision*/
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 784; k++) {
                P_prev[j][k] = P2[j][k];
            }
        }
        /* Calculating the parameters now, P0, P1, p784 for each class now */
        cout << "Current Parameters are: " << endl;
        for (int j = 0; j < 10; j++) {
            if (count_calculated[j] != 0) {
                for (int k = 0; k < 784; k++) {
                    P2[j][k] = ((1 / count_calculated[j]) * responsibility[j] * X[k]);
                    //cout<<P2[j][k]<<" ";
                }
            }
            //cout<<endl;
        }
        int flag = 1;
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 784; k++) {
                if (P2[j][k] != P_prev[j][k])
                    flag = 0;
            }
        }
        if (flag == 1) {
            cout << "Parameters converged after " << i + 1 << "th iteration!" << endl;
            break;
        }
    }
    int D[2][10], val;
    double max2 = 0;
    cout << "--------------------------------------------------------------------------\n" << endl;
    /* Computing the confusion matrix */
    for (int k = 0; k < 10; k++) {
        cout << "For digit : " << k << endl;
        for (int j = 0; j < 10; j++) {
            D[0][j] = abs(count[j] - count_calculated[j]);
            D[1][j] = max(count[j], count_calculated[j]) - abs(count[j] - count_calculated[j]);
            if (max2 > D[0][j]) {
                max2 = D[0][j];
                val = j;
            }
        }
        cout << "digit 'k' is associated to cluster no.: " << val << endl;
        cout << "Sensitivity: " << (double) D[0][k] / (D[0][k] + D[1][k]) << endl;
        cout << "Specificity: " << (double) D[1][k] / (D[0][k] + D[1][k]) << endl;
    }
    return 0;
}



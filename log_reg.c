#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
  int N;
  int p;
  double *W;
  double *b;
} LogReg;

void construct(LogReg*, int, int);
void LogReg_train(LogReg*, int*, int, double);
void LogReg_sigmoid(double*);
void LogReg_predict(LogReg*, int*, double*);
void log_reg(void);

void construct(LogReg *this, int N, int p) {
    int i, j;
    this->N = N;
    this->p = p;
    this->W = (double *)malloc(sizeof(double) * p);
    this->b = (double *)malloc(sizeof(double));
    
    for(j = 0; j < p; ++j) {
        this->W[j] = 0;
    }
    *this->b = 0;
    
}



void LogReg_train(LogReg *this, int *x, int y, double log_reg) {
    int i, j;
    double y_hat;
    double err;

    //simply y = W*X 
    y_hat = 0;
    for(j = 0; j < this->p; ++j) {
        y_hat += this->W[j] * x[j];
    }
    y_hat += *this->b;

    // since we are doing a binary classification we will use the sigmoid function
    LogReg_sigmoid(&y_hat);

    err = y - y_hat;
    for(j = 0; j < this->p; ++j) {
        this->W[j] += log_reg * err * x[j] / this->N;
    }
    *this->b += log_reg * err / this->N;
    
    // if you want to trace the weights 
    // printf("----------------------\n");
    // 
    // for(j = 0; j < this->p; ++j){
    //    printf("%f\t", this->W[j]);
    // }
    // 
    // printf("\n------------------------");
    
}

void LogReg_sigmoid(double* x) {
    *x = 1/(1+exp(-*x));
}

void LogReg_predict(LogReg *this, int *x, double *y) {
    int i, j;
    *y = 0;
    for(j = 0; j < this->p; ++j) {
        *y += this->W[j] * x[j];
    }
    *y += *this->b;
    
}


void log_reg(void) {
    int i, j, epoch;
    double learning_rate = 0.1;  
    int n_epochs = 750;          
    int train_n = 20;             
    int test_n = 7;              
    int p = 6;                

    int train_X[20][6] = {
    {1, 2, 3, 4, 5, 6},
    {2, 3, 4, 5, 6, 7},
    {3, 4, 5, 6, 7, 8},
    {4, 5, 6, 7, 8, 9},
    {5, 6, 7, 8, 9, 10},
    {6, 7, 8, 9, 10, 11},
    {7, 8, 9, 10, 11, 12},
    {8, 9, 10, 11, 12, 13},
    {9, 10, 11, 12, 13, 14},
    {10, 11, 12, 13, 14, 15},

    {11, 12, 13, 14, 15, 16},
    {12, 13, 14, 15, 16, 17},
    {13, 14, 15, 16, 17, 18},
    {14, 15, 16, 17, 18, 19},
    {15, 16, 17, 18, 19, 20},
    {16, 17, 18, 19, 20, 21},
    {17, 18, 19, 20, 21, 22},
    {18, 19, 20, 21, 22, 23},
    {19, 20, 21, 22, 23, 24},
    {20, 21, 22, 23, 24, 25}
  };

  int train_Y[20] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};


  LogReg clf;
  construct(&clf, train_n, p);

  for(epoch = 0; epoch < n_epochs; epoch++) {
      for(i = 0; i < train_n; ++i) {
          LogReg_train(&clf, train_X[i], train_Y[i], learning_rate);
      }  
  }
  int test_X[7][6] = {
      {1, 2, 3, 4, 5, 6},
      {19, 20, 21, 22, 23, 24},
      {2, 3, 4, 5, 6, 7},
      {8, 9, 10, 11, 12, 13},
      {10, 11, 12, 13, 14, 15},

      {11, 12, 13, 14, 15, 16},
      {12, 13, 14, 15, 16, 17}
  };

  double test_Y[2];

  
  for(i = 0; i < test_n; ++i) {
      LogReg_predict(&clf, test_X[i], &test_Y[i]);
      LogReg_sigmoid(&test_Y[i]);
      printf("Predicted probability: %f, Class: %d\n", test_Y[i], test_Y[i] > 0.5);
      printf("\n");
  }

  free(clf.W);
  free(clf.b);
}


int main(void) { 
    log_reg();
    return 0;
}
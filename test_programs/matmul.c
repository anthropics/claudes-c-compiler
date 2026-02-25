#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define N 256
static double A[N][N], B[N][N], C[N][N];
void matmul(void) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
}
int main(void) {
    srand(42);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = (double)rand() / RAND_MAX;
            B[i][j] = (double)rand() / RAND_MAX;
        }
    clock_t start = clock();
    for (int iter = 0; iter < 5; iter++)
        matmul();
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("matmul %dx%d x5: %.3f seconds\n", N, N, elapsed);
    printf("C[0][0] = %.6f\n", C[0][0]); // prevent dead code elimination
    return 0;
}

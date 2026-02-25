#include <stdio.h>
#include <stdlib.h>
long long fib(int n) {
    if (n <= 1) return n;
    long long a = 0, b = 1;
    for (int i = 2; i <= n; i++) {
        long long t = a + b;
        a = b;
        b = t;
    }
    return b;
}
int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 40;
    printf("fib(%d) = %lld\n", n, fib(n));
    return 0;
}

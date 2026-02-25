#include <stdio.h>
#include <string.h>
#include <time.h>
#define LIMIT 10000000
static char sieve[LIMIT + 1];
int run_sieve(void) {
    memset(sieve, 1, sizeof(sieve));
    sieve[0] = sieve[1] = 0;
    for (int i = 2; (long long)i * i <= LIMIT; i++) {
        if (sieve[i]) {
            for (int j = i * i; j <= LIMIT; j += i)
                sieve[j] = 0;
        }
    }
    int count = 0;
    for (int i = 2; i <= LIMIT; i++)
        if (sieve[i]) count++;
    return count;
}
int main(void) {
    clock_t start = clock();
    int count = 0;
    for (int i = 0; i < 3; i++)
        count = run_sieve();
    clock_t end = clock();
    printf("sieve(%d) x3: %d primes, %.3f seconds\n",
           LIMIT, count, (double)(end - start) / CLOCKS_PER_SEC);
    return 0;
}

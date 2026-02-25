#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
int count_words(const char *s) {
    int count = 0, in_word = 0;
    while (*s) {
        if (isspace((unsigned char)*s)) { in_word = 0; }
        else if (!in_word) { in_word = 1; count++; }
        s++;
    }
    return count;
}
void reverse_words(char *s) {
    int len = strlen(s);
    // Reverse entire string
    for (int i = 0, j = len - 1; i < j; i++, j--) {
        char t = s[i]; s[i] = s[j]; s[j] = t;
    }
    // Reverse each word
    int start = 0;
    for (int i = 0; i <= len; i++) {
        if (i == len || s[i] == ' ') {
            for (int a = start, b = i - 1; a < b; a++, b--) {
                char t = s[a]; s[a] = s[b]; s[b] = t;
            }
            start = i + 1;
        }
    }
}
int main(void) {
    char buf[4096];
    // Generate test data
    const char *words[] = {"the","quick","brown","fox","jumps","over","lazy","dog"};
    int pos = 0;
    for (int i = 0; i < 500; i++) {
        const char *w = words[i % 8];
        int wlen = strlen(w);
        if (pos + wlen + 1 >= 4095) break;
        if (pos > 0) buf[pos++] = ' ';
        memcpy(buf + pos, w, wlen);
        pos += wlen;
    }
    buf[pos] = '\0';

    clock_t start = clock();
    long total_words = 0;
    for (int iter = 0; iter < 100000; iter++) {
        total_words += count_words(buf);
        char tmp[4096];
        memcpy(tmp, buf, pos + 1);
        reverse_words(tmp);
    }
    clock_t end = clock();
    printf("strprocess: %.3f seconds, total_words=%ld\n",
           (double)(end - start) / CLOCKS_PER_SEC, total_words);
    return 0;
}

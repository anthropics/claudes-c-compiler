#!/bin/bash
# CCC Benchmark Harness
# Establishes baseline and tracks improvement progress
# Usage: ./benchmark_harness.sh [--baseline | --compare]

set -euo pipefail

CCC="${CCC:-./target/release/ccc}"
GCC="${GCC:-gcc}"
RESULTS_DIR="benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="${RESULTS_DIR}/run_${TIMESTAMP}.json"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

mkdir -p "$RESULTS_DIR" test_programs benchmark_bins

# ============================================================
# SECTION 1: Test Program Generation
# ============================================================

create_test_programs() {
    echo -e "${YELLOW}[*] Creating test programs...${NC}"

    # 1. Hello World (Issue #1 test)
    cat > test_programs/hello.c << 'EOF'
#include <stdio.h>
int main(void) {
    printf("Hello from CCC!\n");
    return 0;
}
EOF

    # 2. Fibonacci (basic correctness + optimization test)
    cat > test_programs/fib.c << 'EOF'
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
EOF

    # 3. Matrix multiply (loop optimization stress test)
    cat > test_programs/matmul.c << 'EOF'
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
EOF

    # 4. String processing (pointer-heavy code, tests sign extension / regalloc)
    cat > test_programs/strprocess.c << 'EOF'
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
EOF

    # 5. Sieve of Eratosthenes (array access patterns, loop opts)
    cat > test_programs/sieve.c << 'EOF'
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
EOF

    echo -e "${GREEN}[+] Test programs created in test_programs/${NC}"
}

# ============================================================
# SECTION 2: Compilation & Measurement
# ============================================================

compile_and_measure() {
    local src="$1"
    local name=$(basename "$src" .c)
    local compiler="$2"
    local flags="$3"
    local label="$4"
    local outbin="benchmark_bins/${name}_${label}"

    # Compile with timing
    local compile_start=$(date +%s%N)
    if $compiler $flags -o "$outbin" "$src" -lm 2>/dev/null; then
        local compile_end=$(date +%s%N)
        local compile_ms=$(( (compile_end - compile_start) / 1000000 ))
        local bin_size=$(stat -c%s "$outbin" 2>/dev/null || echo 0)

        # Run the binary for runtime measurement (3 runs, take median-ish)
        local runtime="N/A"
        if [ -x "$outbin" ]; then
            local total=0
            local runs=3
            for i in $(seq 1 $runs); do
                local run_start=$(date +%s%N)
                timeout 30 "$outbin" > /dev/null 2>&1 || true
                local run_end=$(date +%s%N)
                local run_ms=$(( (run_end - run_start) / 1000000 ))
                total=$((total + run_ms))
            done
            runtime=$((total / runs))
        fi

        echo "${label}|${name}|PASS|${compile_ms}|${bin_size}|${runtime}"
    else
        echo "${label}|${name}|FAIL|0|0|N/A"
    fi
}

# ============================================================
# SECTION 3: Main Benchmark Run
# ============================================================

run_benchmarks() {
    echo -e "${YELLOW}[*] Running benchmarks...${NC}"
    echo ""
    printf "%-14s %-12s %-6s %-12s %-12s %-12s\n" \
           "COMPILER" "TEST" "STATUS" "COMPILE(ms)" "SIZE(bytes)" "RUNTIME(ms)"
    echo "------------------------------------------------------------------------"

    local json_entries=""

    for src in test_programs/*.c; do
        local name=$(basename "$src" .c)

        # CCC (no optimization flags — they're all the same currently)
        result=$(compile_and_measure "$src" "$CCC" "" "ccc")
        IFS='|' read -r label tname status ctime size runtime <<< "$result"
        printf "%-14s %-12s %-6s %-12s %-12s %-12s\n" "$label" "$tname" "$status" "$ctime" "$size" "$runtime"
        json_entries="${json_entries}{\"compiler\":\"ccc\",\"test\":\"$tname\",\"status\":\"$status\",\"compile_ms\":$ctime,\"binary_size\":$size,\"runtime_ms\":\"$runtime\"},"

        # GCC -O0
        result=$(compile_and_measure "$src" "$GCC" "-O0" "gcc-O0")
        IFS='|' read -r label tname status ctime size runtime <<< "$result"
        printf "%-14s %-12s %-6s %-12s %-12s %-12s\n" "$label" "$tname" "$status" "$ctime" "$size" "$runtime"
        json_entries="${json_entries}{\"compiler\":\"gcc-O0\",\"test\":\"$tname\",\"status\":\"$status\",\"compile_ms\":$ctime,\"binary_size\":$size,\"runtime_ms\":\"$runtime\"},"

        # GCC -O2
        result=$(compile_and_measure "$src" "$GCC" "-O2" "gcc-O2")
        IFS='|' read -r label tname status ctime size runtime <<< "$result"
        printf "%-14s %-12s %-6s %-12s %-12s %-12s\n" "$label" "$tname" "$status" "$ctime" "$size" "$runtime"
        json_entries="${json_entries}{\"compiler\":\"gcc-O2\",\"test\":\"$tname\",\"status\":\"$status\",\"compile_ms\":$ctime,\"binary_size\":$size,\"runtime_ms\":\"$runtime\"},"

        echo ""
    done

    # Remove trailing comma and save JSON
    json_entries="${json_entries%,}"
    cat > "$RESULT_FILE" << EOJSON
{
    "timestamp": "$TIMESTAMP",
    "ccc_binary": "$CCC",
    "gcc_binary": "$GCC",
    "results": [$json_entries]
}
EOJSON

    echo -e "${GREEN}[+] Results saved to ${RESULT_FILE}${NC}"
}

# ============================================================
# SECTION 4: Comparison Report
# ============================================================

compare_runs() {
    echo -e "${YELLOW}[*] Comparing benchmark runs...${NC}"
    local latest=$(ls -t "$RESULTS_DIR"/run_*.json 2>/dev/null | head -1)
    local baseline=$(ls -t "$RESULTS_DIR"/run_*.json 2>/dev/null | tail -1)

    if [ "$latest" = "$baseline" ]; then
        echo "Only one run found. Run benchmarks at least twice to compare."
        return
    fi

    echo ""
    echo "Baseline: $baseline"
    echo "Latest:   $latest"
    echo ""
    echo "Use 'jq' or Python to diff the JSON files for detailed comparison."
    echo "Quick check:"
    echo ""
    echo "--- CCC results (baseline) ---"
    grep '"compiler":"ccc"' "$baseline" | head -5
    echo ""
    echo "--- CCC results (latest) ---"
    grep '"compiler":"ccc"' "$latest" | head -5
}

# ============================================================
# MAIN
# ============================================================

create_test_programs

case "${1:-benchmark}" in
    --baseline|benchmark)
        run_benchmarks
        ;;
    --compare)
        compare_runs
        ;;
    *)
        echo "Usage: $0 [--baseline | --compare]"
        exit 1
        ;;
esac

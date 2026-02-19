/* Test for printf formats.  Formats using extensions to the standard
   should be rejected in strict pedantic mode. But allowed by -Wno-pedantic-ms-format.
*/
/* Origin: Kai Tietz <kai.tietz@onevision.com> */
/* { dg-do compile { target { *-*-mingw* } } } */
/* { dg-options "-std=iso9899:1999 -pedantic -Wformat -Wno-pedantic-ms-format" } */

#define USE_SYSTEM_FORMATS
#include "format.h"

enum en1 { A=0, B=1 };
typedef enum { _A=0, _B=1 } en2;

void
foo (int i, long l, long long ll, size_t z, enum en1 e1, en2 e2)
{
  printf ("%I32d", i);
  printf ("%I32d", l);
  printf ("%I32d", e1);
  printf ("%I32d", e2);
  printf ("%I64x", ll);
  printf ("%Ix", z);
}

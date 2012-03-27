#include <time.h>

time_t now;
struct tm* xyz;
struct tm abc;

main()
{
  now = time(NULL);
  xyz = gmtime(&now);
  abc = *xyz;

  printf("%ld, %d\n", abc.tm_sec, abc.tm_sec);
}

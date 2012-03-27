#include <ftw_param.h>

char *test = "testval";

int main(int argc, char *argv[])
{

  setCommandLineParameters(argc, argv);

  getStringParam("-test", &test);

  printf("test = %s\n", test);

  return 0;
}


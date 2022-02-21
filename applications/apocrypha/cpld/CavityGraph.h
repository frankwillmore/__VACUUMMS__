/* CavityGraph.h */

#include <stdio.h>

#ifndef DEF__CAVITYGRAPH
#define DEF__CAVITYGRAPH

extern double cavity_uniqueness_threshold;
extern double cscale;
extern double box_x;
extern double box_y;
extern double box_z;
extern int verbose;

// struct cs_cavity represents a vertex
struct s_cavity
{
  double x;
  double y;
  double z;
  double d;
};

// struct s_pair represents an edge
struct s_pair
{
  int i;
  int j;
};

class CavityGraph
{
  public:

    CavityGraph(char *input_file_name);

    int getNumberOfPairs();
    int getNumberOfCavities();
    int getNumberOfUnmarkedAdjacentCavities(int cavity_number);
    int *getListOfUnmarkedAdjacentCavities(int cavity_number);
    int cavityIsUnique(double x, double y, double z);
    double getCavitySeparation(int cavity1, int cavity2);

  protected:

    int number_of_pairs;
    int number_of_cavities; 
    struct s_cavity *cavities;
    struct s_pair *pairs;

    void generatePairList();
    int isAPair(int i, int j);
};

#endif

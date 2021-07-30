/* CavityPath.h */

#include "Cavity.h"

#ifndef DEF__CAVITYPATH
#define DEF__CAVITYPATH

class CavityPath
{
  public:

    CavityPath(int first_cavity);
    ~CavityPath();

    CavityPath* duplicate();
    void append(int cavity_number);
    void listAllCavities();
    Cavity *getFirstCavity();
    Cavity *getLastCavity();

    CavityPath *getNextPath();

    static int getNumberOfPaths();
    static CavityPath *getFirstPath();
  
  protected:

    Cavity *first_cavity_in_path;
    Cavity *last_cavity_in_path;
    CavityPath *prev_path;
    CavityPath *next_path;
    int number_of_cavities_in_path;

    static CavityPath *first_path;
    static CavityPath *last_path;
    static int number_of_paths;
};
#endif

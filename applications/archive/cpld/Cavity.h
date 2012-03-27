/* Cavity.h */

#ifndef DEF__CAVITY
#define DEF__CAVITY

class Cavity
{
  public:

    Cavity(int cavity_no);
    
    int getCavityNumber();
    Cavity *getNextCavity();
    void setNextCavity(Cavity *next);
    Cavity *getPreviousCavity();
    void setPreviousCavity(Cavity *next);

  protected:

    Cavity *next_cavity;
    Cavity *previous_cavity;
    int cavity_number;
};

#endif

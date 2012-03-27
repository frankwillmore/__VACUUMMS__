/* Cavity.cpp */

#include <stdlib.h>
#include <stdio.h>
#include "Cavity.h"

Cavity::Cavity(int cavity_no)
{
//printf("creating Cavity %d: %d\n", cavity_no, this);
  cavity_number = cavity_no;
}

int Cavity::getCavityNumber()
{
  return cavity_number;
}

Cavity *Cavity::getNextCavity()
{
  return next_cavity;
}

void Cavity::setNextCavity(Cavity *next)
{
  next_cavity = next;
}

Cavity *Cavity::getPreviousCavity()
{
  return previous_cavity;
}

void Cavity::setPreviousCavity(Cavity *prev)
{
   previous_cavity = prev;
}

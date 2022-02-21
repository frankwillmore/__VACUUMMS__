/* CavityPath.cpp */

#include <stdlib.h>
#include <stdio.h>

#include "CavityPath.h"

CavityPath* CavityPath::first_path = NULL;
CavityPath* CavityPath::last_path = NULL;
int CavityPath::number_of_paths = 0;

CavityPath::CavityPath(int first_cavity_no)
{
  // create the first Cavity object
  first_cavity_in_path = new Cavity(first_cavity_no);
  last_cavity_in_path = first_cavity_in_path;

  // put the CavityPath object in the list
  if (number_of_paths == 0)
  { 
    first_path = this;
    last_path = this;
    prev_path = NULL;
    next_path = NULL;
  }
  else 
  {
    last_path->next_path = this;
    next_path = NULL;
    prev_path = last_path;
    last_path = this;
  }

  number_of_paths++;
  number_of_cavities_in_path = 1;
//printf("constructing %d, number_of_paths = %d\n", this, number_of_paths);
}

CavityPath *CavityPath::getFirstPath()
{
  return first_path;
}

CavityPath *CavityPath::getNextPath()
{
  return next_path;
}

CavityPath::~CavityPath()
{
  if (prev_path != NULL) prev_path->next_path = next_path;
  if (next_path != NULL) next_path->prev_path = prev_path;

  number_of_paths--;
// printf("destructing %d, number_of_paths = %d\n", this, number_of_paths);
}

CavityPath *CavityPath::duplicate()
{
  CavityPath *duplicate = new CavityPath(first_cavity_in_path->getCavityNumber());
  for (Cavity *c_ptr = first_cavity_in_path->getNextCavity(); c_ptr != NULL; c_ptr = c_ptr->getNextCavity())
  {
    duplicate->append(c_ptr->getCavityNumber());
  }
    
  return duplicate;
}
    
void CavityPath::append(int cavity_number)
{
  Cavity *c_ptr = new Cavity(cavity_number);
  c_ptr->setPreviousCavity(last_cavity_in_path);
  last_cavity_in_path->setNextCavity(c_ptr);
  last_cavity_in_path = c_ptr;

  number_of_cavities_in_path++;
//printf("num in path %d\n", number_of_cavities_in_path);
}

void CavityPath::listAllCavities()
{
  for (Cavity *c_ptr = first_cavity_in_path; c_ptr != NULL; c_ptr = c_ptr->getNextCavity()) printf("%d-", c_ptr->getCavityNumber());
  printf("\n");
}

int CavityPath::getNumberOfPaths()
{
  return number_of_paths;
}

Cavity *CavityPath::getFirstCavity()
{
  return first_cavity_in_path;
}

Cavity *CavityPath::getLastCavity()
{
  return last_cavity_in_path;
}

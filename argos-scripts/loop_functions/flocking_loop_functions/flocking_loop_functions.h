#ifndef FLOCKING_LOOP_FUNCTIONS_H
#define FLOCKING_LOOP_FUNCTIONS_H

#include <argos3/core/simulator/loop_functions.h>
#include <argos3/plugins/simulator/entities/light_entity.h>
#include <argos3/core/simulator/entity/positional_entity.h>

using namespace argos;

class CFlockingLoopFunctions : public CLoopFunctions {
   
public:

   CLightEntity* m_pcLight;
   CPositionalEntity* m_pcPosLight;

public:

   CFlockingLoopFunctions() :
   m_pcLight(NULL),
   m_pcPosLight(NULL) {}

   virtual ~CFlockingLoopFunctions() {}

   virtual void Init(TConfigurationNode& t_tree);

   virtual void Reset();

   virtual void PostStep();

private:

};

#endif
